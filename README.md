# nisqalib

This is a Python package for [NISQA](https://github.com/gabrielmittag/NISQA).

## Note

This repo includes the fork of the original NISQA repo as a submodule, "nisqa."

## Installation

Install torch according to your environment.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/kale4eat/nisqalib.git
cd nisqalib
git submodule update --init --recursive
pip install -e .
```

## Usage

```python
import nisqalib

# sinbgle file
result = nisqalib.predict_file("nisqa", "/path/to/wav/file.wav")
mos_pred = result["mos_pred"].values[0]

# directory
result = nisqalib.predict_file(
    "nisqa",
    "/path/to/folder/with/wavs",
    "/path/to/dir/with/results")

mos_pred_mean = result["mos_pred"].mean()
```
