from collections import OrderedDict
import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer






class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts,tokenized_prompts):
        x = self.model.encode_text(prompts,True,tokenized_prompts)

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.RPLMED.N_CTX
        ctx_init = cfg.TRAINER.RPLMED.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.RPLMED.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        # Create a temporary CUDA model for computing teacher features
        # (the main biomedclip_model is still on CPU at this stage)
        biomedclip_model_temp,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            num_temp = cfg.TRAINER.RPLMED.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDCOOP_TEMPLATES[classname][i]) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)  # [C, N, D]
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.RPLMED.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute Mean Teacher features
        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # Save unfiltered teacher features for L1 loss
        fixed_embeddings_mean = fixed_embeddings.mean(dim=1)
        fixed_embeddings_mean = fixed_embeddings_mean / fixed_embeddings_mean.norm(dim=-1, keepdim=True)
        
        # Initialize teacher_features (used during inference)
        teacher_features = fixed_embeddings_mean
        
        zero_shot_logits = logit_scale * image_features @ teacher_features.t()
        
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # ========== Teacher Filtering ==========
            # BiomedCoOp-style batch-conditional selective ensemble
            # Dynamically select high-quality Teacher Templates (only for KL distillation)
            with torch.no_grad():
                # fixed_embeddings: [C, N, D]
                C, N, D = fixed_embeddings.shape
                
                # 1) Compute batch-conditional score for each template
                scores = torch.zeros(N, device=image_features.device)
                for j in range(N):
                    text_j = fixed_embeddings[:, j, :]  # [C, D]
                    logits_j = logit_scale * (image_features @ text_j.t())  # [B, C]
                    max_logits = torch.max(logits_j, dim=1).values  # [B]
                    scores[j] = torch.mean(max_logits)  # scalar
                
                # 2) Robust statistical filtering: median + MAD
                s_med = torch.median(scores)
                mad = torch.median(torch.abs(scores - s_med))
                tau = getattr(self.cfg.TRAINER.RPLMED, "TAU", 2.0)
                
                # Keep templates falling within [median - tau*MAD, median + tau*MAD]
                mask = torch.abs(scores - s_med) <= tau * (mad + 1e-6)
                
                # 3) Prevent empty mask (in extreme cases)
                if mask.sum() == 0:
                    mask = torch.ones(N, dtype=torch.bool, device=image_features.device)
                
                # 4) Construct mean teacher using filtered templates
                selected = fixed_embeddings[:, mask, :]  # [C, M, D], M = sum(mask)
                teacher_features = selected.mean(dim=1)  # [C, D]
                teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                
                # Update zero_shot_logits for distillation
                zero_shot_logits = logit_scale * image_features @ teacher_features.t()

            loss_ce = F.cross_entropy(logits, label)
            
            # Confidence-Aware KL Distillation: Prediction distribution alignment weighted by teacher confidence
            # Compute entropy of teacher predictions (uncertainty)
            teacher_probs = F.softmax(zero_shot_logits, dim=1)  # [B, C]
            teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=1)  # [B]
            
            # Convert entropy to confidence weights: Low entropy -> High confidence -> High weight
            # Use exponential decay: weight = exp(-entropy)
            # Weight is 1 when entropy is 0, smaller weight for larger entropy
            confidence_weights = torch.exp(-teacher_entropy)  # [B]
            
            # Normalize weights to have mean 1, maintaining loss scale consistency
            confidence_weights = confidence_weights / (confidence_weights.mean() + 1e-8)  # [B]
            
            # Compute KL divergence per sample (no reduce)
            kl_per_sample = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='none',
                log_target=True
            ).sum(dim=1)  # [B]
            
            # Apply confidence weights
            loss_kl = (kl_per_sample * confidence_weights).mean() * self.cfg.TRAINER.RPLMED.KL_LAMBDA
            
            # L1 Loss: Align student text features with unfiltered teacher features
            # Use fixed_embeddings_mean (mean of all templates) as anchor
            loss_l1 = F.l1_loss(text_features, fixed_embeddings_mean, reduction='mean') * self.cfg.TRAINER.RPLMED.L1_LAMBDA

            total_loss = loss_ce + loss_kl + loss_l1
            
            return logits, total_loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class RPLMED_BiomedCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.RPLMED.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        if cfg.TRAINER.RPLMED.PREC == "fp32" or cfg.TRAINER.RPLMED.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner.ctx"]

        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.RPLMED.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.RPLMED.PREC
        if prec == "amp":
            with autocast():
                logits, loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss = model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            
            # Ignore class-dependent embeddings (different num_classes between base and novel)
            if "prompt_learner.fixed_embeddings" in state_dict:
                del state_dict["prompt_learner.fixed_embeddings"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
