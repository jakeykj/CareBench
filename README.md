# When Does Multimodal Learning Help in Healthcare? A Benchmark on EHR and Chest X-Ray Fusion (CareBench)

## Abstract
Machine learning holds promise for advancing clinical decision support, yet it remains unclear when multimodal learning truly helps in practice, particularly under modality missingness and fairness constraints. In this work, we conduct a systematic benchmark of multimodal fusion between Electronic Health Records (EHR) and chest X-rays (CXR) on standardized cohorts from MIMIC-IV and MIMIC-CXR, aiming to answer four fundamental questions: when multimodal fusion improves clinical prediction, how different fusion strategies compare, how robust existing methods are to missing modalities, and whether multimodal models achieve algorithmic fairness. Our study reveals several key insights. Multimodal fusion improves performance when modalities are complete, with gains concentrating in diseases that require complementary information from both EHR and CXR. While cross-modal learning mechanisms capture clinically meaningful dependencies beyond simple concatenation, the rich temporal structure of EHR introduces a strong modality imbalance that architectural complexity alone cannot overcome. Under realistic missingness, multimodal benefits rapidly degrade unless models are explicitly designed to handle incomplete inputs. Moreover, multimodal fusion does not inherently improve fairness, with subgroup disparities mainly arising from unequal sensitivity across demographic groups. To support reproducible and extensible evaluation, we further release a flexible benchmarking toolkit that enables plug-and-play integration of new models and datasets. Together, this work provides actionable guidance on when multimodal learning helps, when it fails, and why, laying the foundation for developing clinically deployable multimodal systems that are both effective and reliable.

![CareBench Pipeline](assets/pipeline.png)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/CareBench.git
cd CareBench
conda create -n carebench python=3.12
conda activate carebench

# Install PyTorch first (adjust based on your CUDA version)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FastMoE (required for some models like flexmoe)
pip install git+https://github.com/laekov/fastmoe.git@v1.1.0

# Install other dependencies
pip install -r requirements.txt
```

### Dataset Preparation

CareBench is built on **MIMIC-IV** (EHR) and **MIMIC-CXR-JPG** (chest X-rays).  
You need to (1) obtain the raw data from PhysioNet, (2) run the provided preprocessing notebook to build the benchmark EHR folder, and (3) point the training scripts to that folder and to your CXR images.

#### 1. Obtain raw MIMIC data

1. Complete PhysioNet credentialing and accept the data use agreement.
2. Download the following datasets locally:
   - **MIMIC-IV v2.2** (at least `mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`, and optionally `mimiciv_note`).
   - **MIMIC-CXR-JPG v2.0.0** (the JPEG image files and `mimic-cxr-2.0.0-metadata.csv`).
3. Load MIMIC-IV into a PostgreSQL database:
   - Install PostgreSQL (if not already installed): `sudo apt install postgresql postgresql-contrib`
   - Create a database and user (e.g., database `mimiciv22`, user `postgres`)
   - Import MIMIC-IV CSV files into PostgreSQL following the official MIMIC-IV setup guide
   - Configure database connection in `mimiciv_benchmark.ipynb` (Cell 2) by setting environment variables or modifying the connection string:
     ```python
     DBNAME = 'mimiciv22'  # Your database name
     DBUSER = 'postgres'   # Your database user
     DBPASS = 'your_password'
     DBHOST = 'localhost'
     DBPORT = '5432'
     ```

#### 2. Generate the EHR benchmark folder with `mimiciv_benchmark.ipynb`

The notebook `mimiciv_benchmark.ipynb` contains the full EHR preprocessing pipeline:

- Connects to the MIMIC-IV PostgreSQL database (`mimiciv_hosp.*`, `mimiciv_icu.*`, `mimiciv_derived.*`).
- Builds an adult ICU cohort with basic quality filters and 48-hour ICU stay.
- Extracts and merges 20+ tables of time-series signals (vitals, labs, urine output, etc.).
- Computes **phenotype** labels from ICD codes using `icd_9_10_definitions.yaml`.
- Associates ICU stays with available CXRs (`valid_cxrs`) based on `mimic-cxr-2.0.0-metadata.csv`.
- Aggregates signals into **1-hour timesteps for the first 48 hours** after ICU admission.
- Generates per-feature statistics and feature lists.
- Splits the cohort into **5 folds** (train/val/test) and writes all outputs to a dated folder under `DataProcessing/benchmark_data/`.

By default, the notebook uses (Cell 1):

```text
root = "./benchmark_dataset/DataProcessing"
date = <today's YYMMDD>
main_dir = root/benchmark_data/<date>/
```

**Note**: The notebook will create the `benchmark_dataset` directory in the same directory as the notebook if it doesn't exist.

After running the notebook end-to-end, you should obtain a folder like (under `./benchmark_dataset`):

```text
benchmark_dataset/DataProcessing/benchmark_data/250827/
├── merged/                      # Per-stay time-series CSVs used by CareBench
│   ├── {stay_id_1}.csv
│   ├── {stay_id_2}.csv
│   └── ...
├── chartlab/                    # Intermediate per-stay CSVs (not required for training)
├── benchmark_stat/              # Per-feature statistics (intermediate)
├── demographics.csv             # Demographic features per stay
├── stays_meta_with_labels.csv   # Cohort metadata + mortality/phenotype/LoS labels
└── splits/
    ├── features.yaml            # Defines `chartlab_feature` (EHR feature list)
    ├── fold1/
    │   ├── stays_train.csv
    │   ├── stays_val.csv
    │   ├── stays_test.csv
    │   └── train_stats.yaml     # Per-feature stats used for normalization/imputation
    ├── fold2/
    │   └── ...
    └── ...
```

This folder is what CareBench expects as **EHR root**.  
Set `--ehr_root` to the corresponding dated directory, for example (from the project root):

```bash
--ehr_root ./benchmark_dataset/DataProcessing/benchmark_data/250827
```

#### 3. Configure EHR and PKL paths in CareBench

The default paths in `arguments.py` may need to be updated to match your data location. You can either:

- **Option 1**: Update the default paths in `arguments.py` (lines 95-102) to use `./benchmark_dataset/...`
- **Option 2**: Override paths via command line (recommended for flexibility)

The paths you'll need to configure:

- **EHR root**: `./benchmark_dataset/DataProcessing/benchmark_data/250827` (`--ehr_root`)
- **EHR PKL dir**: `./benchmark_dataset/DataProcessing/benchmark_data/250827/data_pkls` (`--pkl_dir`)

On the **first** training run, `datasets/dataset.py` will read raw CSVs from `--ehr_root/merged`, compute normalized tensors and masks, and cache them as PKLs under `--pkl_dir`.  
On later runs, loading will be fast because it reads from the PKL cache.

If you used a different date or location in the notebook, either:
- Update the defaults in `arguments.py`, or
- Override from the command line:

```bash
python main.py --model drfuse --task mortality --fold 1 \
  --ehr_root ./benchmark_dataset/DataProcessing/benchmark_data/<date> \
  --pkl_dir  ./benchmark_dataset/DataProcessing/benchmark_data/<date>/data_pkls
```

#### 4. Prepare CXR images and metadata

CareBench uses **MIMIC-CXR-JPG** for image inputs:

- **Metadata CSV**: use the official `mimic-cxr-2.0.0-metadata.csv` from PhysioNet.
- **Resized JPEGs**: you should resize all used CXR JPEGs (e.g., 256 on the short side and center-crop 224×224, as done in `datasets/dataset.py`) and store them as:

```text
<resized_cxr_root>/
├── {dicom_id_1}.jpg
├── {dicom_id_2}.jpg
└── ...
```

The code only expects `{dicom_id}.jpg` files (flat directory is fine) and uses:
- `valid_cxrs` and `subject_id` from `splits/fold*/stays_*.csv` to select which DICOM IDs to load.
- `mimic-cxr-2.0.0-metadata.csv` for CXR timestamps and basic metadata.

In `arguments.py`, you can place CXR data under `./benchmark_dataset` as well, for example:

- **Resized CXR root**: `./benchmark_dataset/mimic_cxr_resized` (`--resized_cxr_root`)
- **CXR metadata CSV**: `./benchmark_dataset/mimic-cxr-2.0.0-metadata.csv` (`--image_meta_path`)

You can either organize your data to match these paths or override them, for example:

```bash
python main.py --model drfuse --task mortality --fold 1 \
  --ehr_root ./benchmark_dataset/DataProcessing/benchmark_data/<date> \
  --pkl_dir ./benchmark_dataset/DataProcessing/benchmark_data/<date>/data_pkls \
  --resized_cxr_root ./benchmark_dataset/mimic_cxr_resized \
  --image_meta_path ./benchmark_dataset/mimic-cxr-2.0.0-metadata.csv
```

Once these paths are correctly set, you can directly use the training and testing commands below.

### Train a Model

```bash
# Method 1: Using train config file (recommended)
# Train config files are in configs/train_configs/ and contain full training configurations
python main.py --train_config configs/train_configs/drfuse_mortality.yaml

# Override parameters via command line
python main.py --train_config configs/train_configs/drfuse_mortality.yaml \
    --lr 0.0001 --batch_size 32 --epochs 100

# Method 2: Using command line arguments
# The system will automatically load the model config from configs/{model}.yaml
python main.py --model drfuse --task mortality --fold 1 --batch_size 16 --lr 0.0001

# Method 3: Using custom model config path
python main.py --model drfuse --config_path configs/custom_drfuse.yaml --task mortality --fold 1
```

**Note**: 
- `--train_config`: Points to a training config file in `configs/train_configs/` that contains complete training settings (model, task, hyperparameters, etc.)
- `--config_path`: Overrides the default model config path (`configs/{model}.yaml`) with a custom YAML file
- When using `--model`, the system automatically loads `configs/{model}.yaml` as the base model configuration

### Test a Model

```bash
# Method 1: Test with manually specified checkpoint
python main.py --model drfuse --task mortality --mode test \
    --checkpoint_path experiments/drfuse/version_xxx/checkpoints/best.ckpt \
    --fold 1

# Method 2: Auto-find best checkpoint from experiments directory (recommended)
# The system will automatically find the best checkpoint based on model/task/fold/seed
# Single seed
python main.py --model inforeg --task phenotype --mode test \
    --fold 1 --seed 42 \
    --experiments_dir experiments-m-m \
    --compute_fairness --save_predictions

# Multiple seeds (will test all seeds sequentially)
python main.py --model inforeg --task phenotype --mode test \
    --fold 1 --seed 42 123 1234 \
    --experiments_dir experiments-m-m \
    --compute_fairness --save_predictions

# Method 3: Test multiple seeds using config file
# Create a test config file with seeds: [42, 123, 1234] and experiments_dir
python main.py --train_config configs/train_configs/inforeg_phenotype_test.yaml

# Test with fairness evaluation
python main.py --model drfuse --task phenotype --mode test \
    --checkpoint_path path/to/checkpoint.ckpt \
    --fold 1 --compute_fairness --save_predictions
```

**Note**: When using `--experiments_dir`, the system will:
- Automatically find the experiment folder based on `model`, `task`, `fold`, and `seed`
- Search for all checkpoints in the `checkpoints/` directory
- Select the checkpoint with the highest PRAUC (or ACC for LOS task) value from the filename
- Use that checkpoint for testing



---

## 📊 Supported Tasks

| Task | Description | Type | Classes |
|------|-------------|------|---------|
| **mortality** | ICU mortality prediction | Binary | 1 |
| **phenotype** | Phenotype prediction | Multi-label | 25 |
| **los** | Length of stay prediction | Multi-class | 7 |

**Note**: The `--task` parameter accepts one of: `mortality`, `phenotype`, or `los`.

---

## 🤖 Supported Models
- **Uni-modal Baselines**: 
  - LSTM
  - ResNet50
  - Transformer

- **Complete-Modality Multimodal Fusion Methods**:  
  - Late Fusion.
  - [UTDE: Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling](https://arxiv.org/abs/2210.12156)   
  - [DAFT: Distilling Adversarially Fine-tuned Models for Better OOD Generalization](https://arxiv.org/abs/2208.09139)  
  - [MMTM: Multimodal Transfer Module for CNN Fusion](https://arxiv.org/abs/1911.08670) 
  - [AUG: Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](https://arxiv.org/abs/2502.20120)
  - [InfoReg: Adaptive Unimodal Regulation for Balanced Multimodal Information Acquisition](https://arxiv.org/abs/2503.18595)



- **Missing-Modality Multimodal Fusion Methods**:  
  - [HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data](https://arxiv.org/abs/2311.09115)  
  - [Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts](https://arxiv.org/abs/2410.08245)  
  - [DrFuse: Disentangled Representation Fusion for Missing Modality Learning](https://arxiv.org/abs/2403.06197)  
  - [UMSE: Learning Missing Modal Electronic Health Records with Unified Multi-modal Data Embedding and Modality-Aware Attention](https://arxiv.org/abs/2305.02504)
  - [ShaSpec: Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling](https://arxiv.org/abs/2307.14126)  
  - [M3Care: Learning with Missing Modalities in Multimodal Healthcare Data](https://arxiv.org/abs/2210.17292) 
  - [MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images](https://arxiv.org/abs/2207.07027)  
  - [SMIL: Multimodal Learning with Severely Missing Modality](https://arxiv.org/abs/2103.05677)  

---

## ⚙️ Model Hyperparameters and Configurations

For detailed hyperparameter search spaces and best configurations for all models, please refer to [MODEL_PARAMS_README.md](MODEL_PARAMS_README.md).

This document provides:
- **Hyperparameter Search Spaces**: Overview of tunable hyperparameters for each model that underwent Bayesian optimization
- **Best Configurations**: Complete hyperparameter settings for each model across different tasks (mortality, phenotype, length of stay) and cohorts (base cohort, matched subset)
- **Fixed Parameters**: Common training parameters (learning rate, batch size, epochs, etc.) that are consistent across all models

The hyperparameters were optimized using Bayesian optimization, and the best configurations are reported for reproducibility. Models that did not require hyperparameter search (e.g., LateFusion, DAFT, baseline models) have all their parameters documented as fixed values.

---

## 📁 Project Structure

```
CareBench/
├── main.py                      # Main entry point
├── arguments.py                 # Argument parsing
├── requirements.txt             # Dependencies
│
├── configs/                     # Configuration files
│   ├── {model}.yaml            # Model base configs
│   └── train_configs/          # Training configs
│
├── models/                      # Model implementations
│   ├── base/                   # Base classes
│   ├── drfuse/
│   ├── medfuse/
│   └── ...
│
├── datasets/                    # Dataset loaders
│   └── dataset.py
│
├── utils/                       # Utility functions
│   ├── fairness_metrics.py
│   ├── feature_saver.py
│   └── ver_name.py
│
├── robustness_scripts/         # Robustness evaluation scripts
│   └── train_*_robustness.sh   # Model-specific robustness training scripts
│
└── bayesian_search/            # Hyperparameter search scripts
    └── bayesian_search_*.sh    # Model-specific hyperparameter search scripts
```
---

## 🆕 Adding a New Model

Follow these detailed steps to add a new model from scratch:

#### Step 1: Create Model Directory Structure

Create a new directory for your model:
```bash
mkdir -p models/mymodel
```

The directory should contain:
- `mymodel_lightning.py` - Main model implementation (required)
- `mymodel_components.py` - Model-specific components (optional, if you have separate modules)
- `__init__.py` - Module initialization file (required)

#### Step 2: Implement the Model Class

Create `models/mymodel/mymodel_lightning.py` with a class that inherits from `BaseFuseTrainer`. See detailed code example below with full implementation including `forward()` method, task handling, and optimizer configuration.

**Key Requirements:**
- Must inherit from `BaseFuseTrainer`
- Must implement `forward()` method that returns a dict with `'predictions'` and `'loss'` keys
- `forward()` receives `data_dict` with keys: `'ehr_ts'`, `'cxr_imgs'`, `'labels'`, `'has_modality'`
- Handle different tasks (`mortality`, `phenotype`, `los`) with appropriate loss functions
- Optionally override `configure_optimizers()` for custom optimizer/scheduler

**Complete Example:**

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry

@ModelRegistry.register('mymodel')
class MyModel(BaseFuseTrainer):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = self.hparams.task
        
        # Set task-specific number of classes
        if self.task == 'phenotype':
            self.num_classes = self.hparams.num_classes  # 25
        elif self.task == 'mortality':
            self.num_classes = 1  # Binary
        elif self.task == 'los':
            self.num_classes = 7  # 7 classes
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
        self._init_model_components()
    
    def _init_model_components(self):
        self.ehr_encoder = nn.Sequential(
            nn.Linear(self.hparams.input_dim, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout)
        )
        self.predictor = nn.Linear(self.hparams.hidden_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss() if self.task == 'los' else nn.BCEWithLogitsLoss()
    
    def forward(self, data_dict):
        ehr_data = data_dict['ehr_ts']
        labels = data_dict['labels']
        ehr_features = self.ehr_encoder(ehr_data.mean(dim=1))  # Average over time
        logits = self.predictor(ehr_features)
        
        if self.task == 'los':
            loss = self.criterion(logits, labels.long().squeeze())
            predictions = torch.softmax(logits, dim=-1)
        else:
            loss = self.criterion(logits, labels.float())
            predictions = torch.sigmoid(logits)
        
        return {'predictions': predictions, 'loss': loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=getattr(self.hparams, 'wd', 0.0))
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/prauc'}}
```

#### Step 3: Register Model in `models/__init__.py`

Add your model import:
```python
from .mymodel import MyModel
__all__ = [..., 'MyModel']
```

**Note:** The `@ModelRegistry.register('mymodel')` decorator registers your model, but you still need to import it in `__init__.py` so Python can discover it.

#### Step 4: Create Configuration File

Create `configs/mymodel.yaml`:

```yaml
model_name: mymodel
hidden_size: 256
dropout: 0.1
input_dim: 49  # EHR feature dimension
num_classes: 25
task: phenotype  # Options: 'mortality', 'phenotype', 'los'
batch_size: 16
lr: 0.0001
epochs: 50
patience: 10
matched: false
use_demographics: false
pretrained: true
compute_fairness: true
```

**Important:** `model_name` must match the registry name (`'mymodel'`).

#### Step 5: Test Your Model

```bash
# Quick test 
python main.py --model mymodel --task mortality --fold 1 --epochs 1
```

#### Step 6: Train Your Model

```bash
# Using config file (recommended)
python main.py --train_config configs/train_configs/mymodel_mortality.yaml

# Or using command-line arguments
python main.py --model mymodel --task mortality --fold 1 --batch_size 16 --lr 0.0001 --epochs 50
```

#### Step 7: Test Trained Model

```bash
python main.py --model mymodel --task mortality --mode test \
    --checkpoint_path experiments/mymodel/version_xxx/checkpoints/best.ckpt --fold 1
```

#### Tips and Best Practices

1. **Handle Missing Modalities**: Check `data_dict['has_modality']` for missing CXR/EHR data
2. **Task-Specific Losses**: Use `CrossEntropyLoss` for `los`, `BCEWithLogitsLoss` for `mortality`/`phenotype`
3. **Device Management**: `BaseFuseTrainer` handles device placement automatically
4. **Fairness Evaluation**: Enable `compute_fairness` in config for demographic fairness analysis

For more examples, see `models/drfuse/drfuse.py`.

---

## 📝 Citation

If you use CareBench in your research, please cite:

```bibtex
@article{yin2026carebench,
  title={When Does Multimodal Learning Help in Healthcare? A Benchmark on EHR and Chest X-Ray Fusion},
  author={Yin, Kejing and Xu, Haizhou and Yao, Wenfang and Liu, Chen and Chen, Zijie and Cheung, Yui Haang and Cheung, William K and Qin, Jing},
  journal={arXiv preprint arXiv:2602.23614},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


