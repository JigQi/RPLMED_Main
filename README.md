# RPL-Med: Reliability-Oriented Prompt Distillation for Few-Shot Biomedical Image Classification

RPL-Med (**R**eliability-Oriented **P**rompt Distillation for **Med**ical Images) addresses a key problem in few-shot biomedical image classification: LLM-generated textual descriptions used as teacher knowledge may not be uniformly reliable for every medical image. The framework introduces three components:

- **MAD-Based Teacher Selection** — Median Absolute Deviation filtering dynamically removes outlier LLM-generated templates per batch, ensuring only well-aligned textual descriptions contribute to distillation.
- **Confidence-Aware Distillation** — Entropy-weighted KL divergence down-weights high-uncertainty teacher predictions, preventing blind imitation of ambiguous supervision.
- **L1 Semantic Anchoring** — An L1 constraint stabilizes learnable prompts against the dynamic teacher, preventing prompt drift during training.

During inference, only the lightweight learned student prompts are retained — no teacher branch, no additional computation overhead.

## Project Structure

```
RPLMed/
├── assets/
│   ├── INSTALL.md            # Installation guide
│   ├── DATASETS.md           # Dataset preparation instructions
│   └── RUN.md                # Training & evaluation guide
├── clip/                     # CLIP / PubMedCLIP / PMC-CLIP backends
├── configs/
│   ├── datasets/             # Dataset configs (11 datasets)
│   └── trainers/             # Trainer configs (RPLMed, CoOp, CoCoOp, ...)
├── Dassl.pytorch/            # Dassl.pytorch library
├── datasets/                 # Dataset classes & loaders
├── open_clip/                # OpenCLIP integration for BiomedCLIP
├── scripts/                  # Training/evaluation bash scripts
├── trainers/
│   ├── RPLMED/               # Proposed method
│   ├── CoOp/                 # Context Optimization
│   ├── CoCoOp/               # Conditional Context Optimization
│   ├── KgCoOp/               # Knowledge-guided CoOp
│   ├── ProGrad/              # Prompt-aligned Gradient
│   ├── BiomedCoOp/           # Biomedical CoOp (baseline)
│   ├── ClipAdapter/          # CLIP Adapter
│   ├── TipAdapter/           # Tip-Adapter
│   ├── LP/ LP2/              # Linear Probe variants
│   └── Zeroshot/             # Zero-shot baseline
├── main.py                   # Entry point for few-shot evaluation (standalone)
├── train.py                  # Main training script (uses Dassl.pytorch)
├── parse_test_res.py         # Parse & average results across seeds
├── interpret_prompt.py       # Interpret learned prompts to nearest words
├── combine_acc_columns.py    # Combine accuracy CSVs across datasets
└── requirements.txt          # Python dependencies
```

## Installation

```bash
# 1. Create conda environment
conda create -n rplmed python=3.10 -y
conda activate rplmed

# 2. Install PyTorch 2.0.1
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Dassl.pytorch
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

For detailed installation instructions, see [assets/INSTALL.md](assets/INSTALL.md).

## Data Preparation

Place all datasets under a `data/` directory:

```
data/
├── BTMRI/
├── BUSI/
├── CHMNIST/
├── COVID_19/
├── CTKidney/
├── DermaMNIST/
├── KneeXray/
├── Kvasir/
├── LungColon/
├── OCTMNIST/
└── RETINA/
```

|    Modality    |   Organ   |   Dataset   | Classes |     Train/Val/Test     |
| :------------: | :--------: | :---------: | :-----: | :---------------------: |
|       CT       |   Kidney   |  CTKidney  |    4    |  6,221 / 2,487 / 3,738  |
|  Dermatoscopy  |    Skin    | DermaMNIST |    7    |  7,007 / 1,003 / 2,005  |
|   Endoscopy   |   Colon   |   Kvasir   |    8    |   2,000 / 800 / 1,200   |
|     Fundus     |   Retina   |   RETINA   |    4    |   2,108 / 841 / 1,268   |
| Histopathology | Lung/Colon |   LC25000   |    5    | 12,500 / 5,000 / 7,500 |
| Histopathology | Colorectal |   CHMNIST   |    8    |  2,496 / 1,000 / 1,504  |
|      MRI      |   Brain   |    BTMRI    |    4    |  2,854 / 1,141 / 1,717  |
|      OCT      |   Retina   |  OCTMNIST  |    4    | 97,477 / 10,832 / 1,000 |
|   Ultrasound   |   Breast   |    BUSI    |    3    |     389 / 155 / 236     |
|     X-Ray     |   Chest   | COVID-QU-Ex |    4    | 10,582 / 4,232 / 6,351 |
|     X-Ray     |    Knee    |  KneeXray  |    5    |   5,778 / 826 / 1,656   |

All datasets are available on [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/tree/main).

For detailed dataset preparation, see [assets/DATASETS.md](assets/DATASETS.md).

## Training and Evaluation

### Few-Shot

```bash
CUDA_VISIBLE_DEVICES=<GPU> bash scripts/RPLMed/few_shot.sh <data_dir> <dataset>
```

### Base-to-Novel Generalization

```bash
CUDA_VISIBLE_DEVICES=<GPU> bash scripts/RPLMed/base2new.sh <data_dir> <dataset>
```

### Averaging Results Over 3 Seeds

```bash
python parse_test_res.py --directory output/<dataset>/shots_16/RPLMED_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

For detailed instructions and baseline methods, see [assets/RUN.md](assets/RUN.md).

## Interpreting Learned Prompts

After training, you can interpret what the learned context vectors mean:

```bash
python interpret_prompt.py --fpath <path_to_prompt_checkpoint> --topk 5
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
