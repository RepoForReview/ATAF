# ATAF

Anonymous code repository for the manuscript submitted to EAAI.

Repository: [RepoForReview/ATAF](https://github.com/RepoForReview/ATAF).

## 1. Project structure

The commands below refer to the expanded project layout used for the revised experiments:

```text
ATAF/
├── main.py                       # Training/evaluation entry point
├── requirements.txt              # Python dependency versions
├── loss.py                       # Classification/contrastive-loss utilities
├── utils.py                      # Training utilities
├── paired_bh.py                  # Paired t-tests and BH-adjusted p-values
├── backbone/                     # HGRNet and competing recognition models
├── tl_strategies/                # Transfer-learning methods
├── data/                         # Dataset readers and local dataset directories
├── run/                          # Subject lists and experiment launch scripts
├── checkpoints/                  # Source-model state dictionaries
```

Run training commands from the repository root. Relative paths in the loaders and training scripts are resolved from that directory.

## 2. Software environment

Use a dedicated Python environment. The installation example below targets Python 3.11 on Linux with an NVIDIA GPU. CPU execution is available through `--device cpu`, but GPU timing and GPU memory measurements require CUDA.

```bash
git clone https://github.com/RepoForReview/ATAF.git
cd ATAF
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

The supplied `requirements.txt` specifies:

| Package | Version |
|---|---|
| PyTorch | 2.10.0+cu130 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| einops | 0.8.2 |
| fvcore | 0.1.5.post20221221 |
| calflops | 0.3.2 |

The CUDA wheel requires a compatible NVIDIA driver. For CPU-only installation, install the corresponding CPU PyTorch build and omit the CUDA-specific PyTorch line when installing the remaining requirements. Official PyTorch wheels are listed in the [PyTorch CUDA 13.0 index](https://download.pytorch.org/whl/cu130/torch/).


```bash
python -m pip install openpyxl pygame
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip freeze > environment-resolved.txt
```

The manuscript reports an AMD EPYC 7542 CPU and an NVIDIA GeForce RTX 4070 Ti SUPER GPU. Hardware-dependent runtime/memory measurements should be repeated on the same configuration for numerical comparisons. Preserve `environment-resolved.txt` and the GPU/driver information with each experiment release.

## 3. Data preparation

Obtain Hyser, Ninapro DB3 and Senic from their original dataset providers, following the corresponding access and use conditions and the dataset references in the manuscript. The loaders expect the following local organization:

```text
data/
├── Hyser/
│   ├── subject2_session1.mat
│   ├── subject2_session2.mat
│   └── ...
├── Ninapro/
│   ├── DB3_s1/
│   │   ├── S1_E1_A1.mat
│   │   └── S1_E2_A1.mat
│   └── ...
└── Senic/
    ├── h0/
    │   ├── emg_p0_r0_fist.csv
    │   └── ...
    └── ...
```

### Dataset interfaces

- **Hyser:** `subject<ID>_session<1|2>.mat` must contain `data` and `label`. The loader expects trial arrays in `data[0]` and one-based gesture labels in `label[0]`. These files are the loader's prepared input format; the training command does not convert an arbitrary raw Hyser download into this format. The preparation/repacking step must retain the session, trial and gesture identities described in the manuscript.
- **Ninapro DB3:** retain `emg`, `rerepetition` and `restimulus` in the exercise MAT files. The code reads exercises E1 and E2 and retains the 21-class label set used in the experiments.
- **Senic:** the loader reads CSV files without a header from `h<ID>/`, with filenames matching `emg_p<position>_r<repetition>_<gesture>.csv`. Gesture names are `rest`, `eversion`, `fist`, `open_hand`, `pinch_forefinger`, `pinch_middlefinger`, `two` and `varus`.

### Participants

| Dataset | IDs used by the launch scripts | Subjects |
|---|---|---:|
| Hyser | 2, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 16 |
| Ninapro | 1, 2, 3, 4, 5, 6, 9, 10, 11 | 9 |
| Senic | 0–35 | 36 |

Hyser subjects 1, 3, 5 and 11 are excluded because their recordings do not contain the complete set of 34 gestures used in this study. Subject IDs refer to the original/local indexing shown above; they are not renumbered consecutive participant labels.

### Processing and splitting

The dataset-specific implementations are in:

- `data/Hyser_processed_load.py`
- `data/Ninapro_processed_load.py`
- `data/Senic_processed_load.py`
- `data/inter_dataset_provide.py`

Signals are converted to 1,000 Hz and model inputs contain 200 samples per window. Ninapro is initially segmented at its original 2,000-Hz sampling rate using 400-sample windows and then resampled. Dataset-specific filters, rest handling and normalization are implemented in the above files.

For Hyser, the final available trial of each gesture in each session is retained for testing; remaining trials provide the training/validation data. Ninapro uses repetitions 1–5 for training/validation and repetition 6 for testing. Senic uses repetitions 0–1 for training/validation and repetition 2 for testing. Training/validation splitting uses `random_state=42`. Transfer experiments use the target-domain training partition for adaptation and retain the target test partition for evaluation. The target-data sampling routine selects approximately 10% of the target training samples, balanced across classes according to the implementation.

## 4. Shared experimental settings

| Setting | Configuration |
|---|---|
| Global training seed | 6718 (`--random_seed`) |
| Training epochs | 50 |
| Batch size | 128 |
| Optimizer | AdamW |
| Initial learning rate, intra-subject | 0.001 |
| Initial learning rate, transfer | 0.01 |
| LR scheduler | StepLR, step size 20, gamma 0.8 |
| Backbone checkpoint selection | Lowest validation loss |
| HGRNet channels/classes | Hyser: 256/34; Ninapro: 12/21; Senic: 8/8 |
| GDAdapter-A default | Two blocks; learnable lambda initialized at 1 |

In the inter-subject setting, each source subject is paired with the next target subject in the launch-script list, with wraparound for the final subject. The same subject pairs and target-data protocol must be used across all compared methods.

In the inter-device setting, each method uses the same source/target pairs for each of five repetitions. Each repetition contains nine pairs, giving 45 runs per direction. The `run/` scripts print the repetition number and participant lists before running the methods. Preserve these printed pairing manifests with the results. Hyser–Senic directions also sample nine subjects per dataset; they do not involve a Ninapro target or source.

### Method names and training scope

| Manuscript method | `--tl_strategy` | Inter-subject trainable components | Inter-device trainable components |
|---|---|---|---|
| w/o STL | `no_adapter` | No task-specific adaptation | Not used in Table 3 |
| FT-KNN / FT-SVM / FT-RF | `ml` | Classifiers fitted to target-domain features | Not used in Table 3 |
| FPT | `fpt` | Full network | Full network |
| LP | `lp` | Classifier | Target classifier and channel alignment |
| BT | `bt` | Bias parameters | Target classifier, channel alignment and bias parameters |
| Conv-Adapter | `cadapter` | Convolutional adapter | Convolutional adapter, target classifier and channel alignment |
| GDAdapter-A | `gdadapter` | GDAdapter | GDAdapter, target classifier and channel alignment |


## 5. Reproduce recognition experiments

### 5.1 Intra-subject benchmark:

Example for one Hyser subject:

```bash
python main.py --adapt_mode intra-subject --dataset Hyser \
  --source_parti 2 --backbone HGRNet --device cuda:0 --random_seed 6718
```

Repeat for all subjects using the corresponding script after changing its GPU ID:

```bash
bash run/intra_subject_hyser.sh
bash run/intra_subject_ninapro.sh
bash run/intra_subject_senic.sh
```

For competing backbones, replace `--backbone HGRNet` with `EUNet`, `MKCNN`, `FINCNN`, `CEMGNet`, `TraHGR`, `STCNet`, `EMGHandNet` or `HTD`. Each algorithm must be evaluated on the same participant list and data partitions.

### 5.2 Inter-subject benchmark

First train the source HGRNet checkpoints as above. An example source-to-target adaptation is:

```bash
python main.py --adapt_mode inter-subject --dataset Hyser \
  --source_parti 2 --target_parti 4 --backbone HGRNet \
  --tl_strategy gdadapter --device cuda:0 --random_seed 6718
```

The full participant-pair lists are in:

```bash
bash run/inter_subject_hyser.sh
bash run/inter_subject_ninapro.sh
bash run/inter_subject_senic.sh
```

For each pair, repeat the command with `no_adapter`, `ml`, `fpt`, `lp`, `bt` and `cadapter` as appropriate. The current inter-subject shell scripts select `gdadapter`; change the method argument or repeat the commands for the baselines. FPT accuracy is also reported separately in Table 2.

### 5.3 Inter-device benchmark

Here `--dataset` denotes the source dataset; `--target_device` denotes the target dataset.

```bash
python main.py --adapt_mode inter-device --dataset Hyser --target_device Ninapro \
  --source_parti 2 --target_parti 1 --backbone HGRNet \
  --tl_strategy gdadapter --device cuda:0 --random_seed 6718
```

This single command illustrates one pair. Use the scripts below for the five-repetition protocol, after updating the GPU IDs:

```bash
bash run/hyser_2_ninapro.sh
bash run/hyser_2_senic.sh
bash run/ninapro_2_hyser.sh
bash run/ninapro_2_senic.sh
bash run/senic_2_hyser.sh
bash run/senic_2_ninapro.sh
```

Each script loops over `fpt`, `bt`, `lp`, `cadapter` and `gdadapter` on identical pairs.

## 6. Statistics and result tables

`paired_bh.py` provides the function `paired_bh(data, reference)`:

```python
import pandas as pd
from paired_bh import paired_bh

# First row: method names. Each subsequent row: one matched subject/run.
data = pd.read_excel("results/inter_device_Hyser_to_Ninapro.xlsx")
data = data[["FPT", "LP", "BT", "Conv-Adapter", "GDAdapter-A"]]
result = paired_bh(data, reference="GDAdapter-A")
print(result.to_string(index=False, float_format=lambda x: f"{x:.10g}"))
```

The Excel path in this example is the result table you prepare from the experiments; the training commands do not automatically create it. If the table contains a subject/pair ID column, read it with `index_col=0` or remove that column before testing. Do not include mean/SD summary rows.

The function performs two-sided paired t-tests against the reference and applies BH correction jointly to **all comparisons in that call**. It returns raw p-values, adjusted p-values, t statistics, the number of pairs and signed Cohen's dz. The sign is comparator minus reference: shorter completion time relative to Scratch produces negative dz.

A directly runnable call is included at the end of `paired_bh.py`:

```bash
python paired_bh.py
```

To calculate descriptive statistics from the same matched table:

```python
summary = data.agg(["mean", "std"]).T  # std uses sample SD (ddof=1)
print(summary)
```

## 7. Checkpoints

Source-model checkpoints are expected at:

```text
checkpoints/intra-subject/<Dataset>/HGRNet/P<source_ID>/best_model.pth
```

For example:

```text
checkpoints/intra-subject/Hyser/HGRNet/P2/best_model.pth
checkpoints/intra-subject/Ninapro/HGRNet/P1/best_model.pth
checkpoints/intra-subject/Senic/HGRNet/P0/best_model.pth
```

## 8. Permanent availability

The anonymous repository will remain accessible during peer review. Upon publication, a versioned snapshot of the code and associated reproducibility materials will be deposited in a permanent archive. Its persistent identifier will be added to the final manuscript and this README. The archived snapshot will identify the source-code revision and the associated result/checkpoint versions.
