# CNN-GRU Pulse Waveform Classification

A PyTorch research prototype for **11-class classification of three-channel pulse waveforms**. The project separates packet decoding, signal preprocessing, model definition, and dataset preparation from an earlier experimental script.

**Status:** initial source release. The original environment has not been recovered, and this release has not been validated end to end. No new training or evaluation run was performed for this release. There is currently no verified held-out performance result to report.

## Implemented components

- Decode hexadecimal recordings and check frame checksums.
- Reconstruct three waveform channels from the original packet format.
- Apply Savitzky-Golay smoothing and peak-based segmentation.
- Interpolate each window to 500 time points and normalize each channel.
- Define a Conv1d + GRU classifier with 11 output classes.
- Export samples with deterministic class labels, recording IDs, and within-recording window indices.

## Model

```text
Input: [batch, 500, 3]
          |
Conv1d (3 -> 64) + ReLU
          |
Three MaxPool1d layers
          |
GRU (input size 64, hidden size 256)
          |
Last recurrent output -> Linear (256 -> 11)
          |
Class logits
```

The classifier returns logits for use with cross-entropy loss. The default architecture follows the supplied prototype; it is not presented as a novel model architecture.

## Repository layout

```text
src/
  models.py           CNN-GRU model
  data.py             PyTorch Dataset and DataLoader
  preprocessing.py    Packet decoding and waveform preprocessing
  prepare_data.py     Dataset export and summary generation
legacy/
  gru_exm.py          Earlier experimental script, retained for reference
config.example.json   Portable configuration template
requirements.txt      Dependencies for the modular code
```

## Dataset layout

The original recordings and model checkpoints are not distributed in this repository. Supply recordings locally in the following structure:

```text
data/raw/
  class_name_1/
    recording_1.txt
  class_name_2/
    recording_2.txt
  ...
```

Each nonempty line is expected to contain 19 whitespace-separated hexadecimal byte tokens. Decoding follows the layout in the original script; other recording formats require an adapted parser. Folder names determine class labels, sorted by name, and the mapping is saved with the prepared dataset.

## Setup and data preparation

The following commands document the intended workflow; they have not been verified in the original environment. Dependency versions are not pinned yet.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config.example.json config.local.json
```

Edit `raw_data_dir` in `config.local.json` to point to your local recordings, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
```

Dataset preparation is designed to produce the following local files under `artifacts/dataset/`:

- `pulse_windows.npz`: features, labels, recording IDs, and window indices.
- `class_to_index.json`: the class-label mapping.
- `records.json`: local recording metadata and sample counts.
- `summary.json`: dataset dimensions and per-class counts.

`config.local.json`, data, generated artifacts, and weights are excluded by `.gitignore`.

## Evaluation status and limitations

The inspected local dataset contains 18 TXT recordings across 11 classes. Some classes have only one recording. Overlapping windows from a recording are related samples, so a random window split can produce overly optimistic results.

- The data-preparation script does not create train/test splits.
- Recording IDs identify files, not necessarily independent subjects.
- Full 11-class evaluation on independent recordings is constrained by the single-recording classes.
- A temporal split must remove overlapping windows and be reported as within-recording evaluation.
- Training and standalone evaluation entry points are still being separated from the legacy prototype.
- Accuracy, macro-F1, confusion matrices, and baseline comparisons will be reported only after an appropriate evaluation is performed.

The new deterministic class mapping may differ from the legacy script's directory-order mapping. Existing checkpoints require a verified mapping and architecture before reuse.

## Development history

The legacy script was retained as supplied. The modular model, dataset preparation, and documentation are a refactor prepared in November 2026. Legacy plotting constants and training-loss plots are not verified test metrics.
