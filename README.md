# Dual-Stage Pruning for Human Activity Recognition on Microcontrollers

Reference implementation for the experiments in **"Dual-Stage Pruning for Human Activity Recognition on Microcontrollers"** by Quang Thong Phan and Kristof Van Laerhoven.

The project trains a lightweight depthwise-separable 1D CNN for wearable Human Activity Recognition (HAR), then applies structured pruning in two sequential stages:

1. **Input-bin pruning:** a straight-through Gumbel-Softmax mask selects informative FFT, DCT, or Integer Haar Wavelet (IHW) bins.
2. **Channel pruning:** a second Gumbel-Softmax mask selects convolutional channels. The selected model is physically compacted and fine-tuned.

The implementation supports the UCI-HAR and WEAR datasets, Leave-One-Subject-Out (LOSO) evaluation, post-training TFLite quantization, and the local part of the ST Edge AI Cloud benchmarking workflow.

> This repository does **not** include datasets, trained checkpoints, ST Edge AI Cloud credentials, or generated firmware. The hardware measurements reported in the paper were produced by submitting exported models to the public ST Edge AI Cloud testbench.

## Paper at a glance

The paper evaluates time-domain (`no`), FFT (`fft`), DCT (`dct`), and IHW (`ihw`) inputs with a CNN of approximately 37k parameters. Each training run uses LOSO folds: one training subject is held out for validation and the provided test split is evaluated for every fold.

| Dataset | Paper's balanced dual-pruning configuration | Quantized macro F1 | STM32F401 latency | RAM / ROM |
| --- | --- | ---: | ---: | ---: |
| WEAR | IHW-DP | 71.85 +/- 0.65% | 23.10 +/- 3.65 ms | 10.84 +/- 0.96 KB / 91.74 +/- 10.32 KB |
| UCI-HAR | DCT-DP | 90.28 +/- 1.37% | 24.72 +/- 6.58 ms | 11.22 +/- 1.79 KB / 86.02 +/- 12.22 KB |

These are the paper's W8A16 measurements on an STM32F401 at 84 MHz. Results from a local run can differ with package versions, hardware, selected folds, and sparsity settings.

## Setup

Python 3.10+ is recommended. Create an isolated environment and install the pinned experiment dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The training entry points import Weights & Biases (`wandb`). Log in before enabling online experiment tracking:

```bash
wandb login
```

For a non-interactive UCI five-stage run, set `WANDB_MODE=disabled`. This is preferable to passing `--wandb False` to that particular script, whose boolean option is legacy and does not parse the string `False` as expected.

## Dataset preparation

Datasets are intentionally excluded from version control. Place them at the repository root using the following layouts; the scripts resolve these paths automatically.

### UCI-HAR

Download and extract the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Rename or move the extracted `UCI HAR Dataset` directory to `uci-har`:

```text
uci-har/
  train/
    Inertial Signals/
    subject_train.txt
    y_train.txt
  test/
    Inertial Signals/
    subject_test.txt
    y_test.txt
```

The original split is retained. Training folds are created from `uci-har/train/subject_train.txt`; the original `test` split is used for evaluation.

### WEAR

Obtain the WEAR data from its authors or project distribution, then arrange the pre-split subject CSV files as follows:

```text
wear/
  train/
    subject_train.txt
    sbj_<subject-id>.csv
    ...
  test/
    subject_test.txt
    sbj_<subject-id>.csv
    ...
```

`subject_<split>.txt` contains the available subject IDs; every listed ID must have a corresponding `sbj_<id>.csv`. The loader forms 2-second windows at 50 Hz (100 samples) with 50% overlap and maps the 18 original labels to the eight activity classes used in the paper.

## Reproduce experiments

All entry points accept `--preprocessing {fft,dct,ihw,no}`:

- `fft` - one-sided FFT magnitude
- `dct` - absolute DCT-II coefficients
- `ihw` - Integer Haar Wavelet coefficients, emulating the MCU-style int16 transform
- `no` - raw time-domain signal

The paper's main method is the five-stage pipeline:

| Stage | Operation | Output |
| --- | --- | --- |
| 1 | Train the unmasked `SeparableConvCNN` | baseline checkpoint |
| 2 | Learn the input-bin mask | input-mask checkpoint |
| 3 | Slice retained bins and retrain | pruned-input checkpoint |
| 4 | Learn convolution-channel masks on the pruned input | channel-mask checkpoint |
| 5 | Physically compact selected channels and fine-tune | compact dual-pruned checkpoint |

Run all LOSO folds by omitting `--subjects`. Start with one fold when validating the setup; UCI subject IDs are typically `1`-`30`, while WEAR IDs depend on the supplied split.

```bash
# UCI-HAR: DCT dual pruning, one validation fold
WANDB_MODE=disabled python uci_main_loso_five_stage.py \
  --preprocessing dct \
  --subjects 1 \
  --sparsity_weight_bin 0.10 \
  --sparsity_weight_channel 0.20 \
  --performance

# WEAR: IHW dual pruning, one validation fold
python wear_main_loso_five_stage.py \
  --preprocessing ihw \
  --subjects 0 \
  --sparsity_weight_bin 0.40 \
  --sparsity_weight_channel 0.25 \
  --wandb False \
  --performance
```

The default configuration trains 60 epochs at each stage (300 stage-epochs per fold), with Adam, learning rate `1e-3`, batch size 64, dropout 0.4, and Gumbel temperature scheduled from 10 to 1. Use `--epochs_stage1` through `--epochs_stage5` to shorten a setup check; do not treat a shortened run as a reproduction of the reported results.

### Other experiment variants

| Goal | UCI-HAR entry point | WEAR entry point |
| --- | --- | --- |
| Unpruned baseline | `uci_main_loso_baseline.py` | `wear_main_loso_baseline.py` |
| Input pruning only (three stages) | `uci_main_loso_input_pruning.py` | `wear_main_loso_input_pruning.py` |
| Channel pruning only (three stages) | `uci_main_loso_channel_pruning.py` | `wear_main_loso_channel_pruning.py` |
| Dual pruning (five stages) | `uci_main_loso_five_stage.py` | `wear_main_loso_five_stage.py` |

For example, run UCI input pruning on selected folds with:

```bash
python uci_main_loso_input_pruning.py \
  --preprocessing dct \
  --subjects 1,2,3 \
  --sparsity_weight_bin 0.15 \
  --wandb False \
  --performance
```

Use `--stage<N>_model_path` to reuse a checkpoint and skip that stage. A `{subject}` placeholder is supported for Stage 1 paths, which is useful when reusing a separate checkpoint per LOSO fold.

## Outputs

Each run creates these directories as needed:

- `models/` - checkpoints, including the physically pruned Stage 3 and Stage 5 models.
- `log/` - per-run settings, per-fold results, and hard input masks. These logs are needed by quantization to slice input bins consistently.
- `wandb/` - local Weights & Biases run files when tracking is enabled.

The primary five-stage logs are `log/uci_loso_five_stage_results_<preprocessing>.txt` and `log/wear_loso_five_stage_results_<preprocessing>.txt`.

## TFLite export and post-training quantization

The quantization scripts export a PyTorch checkpoint through ONNX and `onnx2tf`, then generate and evaluate these TFLite configurations:

- `W8A16_FLOAT_IO`
- `W8A16_INT_IO`
- `W8A8_INT_IO`

For Stage 3 and Stage 5 models, pass the five-stage log so the exporter can recover the learned hard input mask. The sample below exports one UCI-HAR DCT dual-pruned fold after the five-stage training command above:

```bash
python uci_quantize_loso_tflite_ptq.py \
  --stage 5 \
  --preprocessing dct \
  --subjects 1 \
  --mask-log-file log/uci_loso_five_stage_results_dct.txt \
  --tflite-output-path models/tflite/uci/dct-dp \
  --wandb False
```

Use the corresponding `wear_quantize_loso_tflite_ptq.py` command for WEAR. Exported files are named `subject<id>_<configuration>.tflite`, and quantization metrics are written to `log/*_ptq_stage<stage>_*.txt`.

The scripts perform local conversion and TFLite evaluation. To reproduce the paper's hardware figures, upload the exported W8A16 model to the [ST Edge AI Cloud](https://stedgeai-dc.st.com/) and profile it on an STM32F401 target (84 MHz, 512 KB flash, 96 KB SRAM). Firmware generation and testbench submission are external to this repository.

## Repository map

```text
lib/
  model.py              CNN, Gumbel masks, and compact pruned models
  uci_train.py          UCI-HAR data loading and training pipelines
  wear_train.py         WEAR data loading and training pipelines
  ml_lib.py             shared training, masking, and pruning utilities
  signal_lib.py         signal-processing helpers
*_main_loso_*.py        experiment entry points
*_quantize_loso_*.py    ONNX -> TensorFlow -> TFLite PTQ and evaluation
scripts/                historical sweep and batch-run examples
img/                    pruning illustrations
```

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{phan2026dualstage,
  author    = {Quang Thong Phan and Kristof Van Laerhoven},
  title     = {Dual-Stage Pruning for Human Activity Recognition on Microcontrollers},
  booktitle = {Proceedings of the 11th International Workshop on Sensor-Based Activity Recognition and Artificial Intelligence},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```
