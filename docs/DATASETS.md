# Dataset setup

Raw datasets are not included in this repository. Obtain them from the
official sources below and comply with their licenses and terms.

`tools/prepare_dataset.py` validates paths and creates local links or a small
path configuration. It never downloads or copies raw data.

## BallsCF and BlocktowerCF

Source: Filtered-CoPhy / CoPhy.

- Project: https://filteredcophy.github.io
- Official archive: https://zenodo.org/records/5906002

The experiments use the four-object, same-mass subsets at 112x112. The loader
expects each selected episode to contain:

```text
<scenario-root>/<episode-id>/ab/rgb.mp4
```

Register each scenario:

```bash
python tools/prepare_dataset.py \
  --dataset ballsCF \
  --source /path/to/CoPhy_112

python tools/prepare_dataset.py \
  --dataset blocktowerCF \
  --source /path/to/CoPhy_112
```

The release includes only the small split lists and background plates required
to reproduce the paper protocol. It does not include CoPhy videos.

## OBJ3D

Source: the G-SWM OBJ3D benchmark.

- Dataset: https://huggingface.co/datasets/taldatech/OBJ3D
- Original project: https://github.com/zhixuan-lin/G-SWM

Expected layout:

```text
OBJ3D/
  train/<episode-id>/test_1.png ... test_100.png
  val/<episode-id>/test_1.png ... test_100.png
  test/<episode-id>/test_1.png ... test_100.png
```

Register it:

```bash
python tools/prepare_dataset.py \
  --dataset obj3d \
  --source /path/to/OBJ3D
```

The command creates `data/obj3d` as a local symbolic link.

## Simb

Source: Region Proposal Interaction Networks (RPIN).

- Official instructions:
  https://github.com/HaozhiQi/RPIN/blob/master/docs/SimB.md
- Project repository: https://github.com/HaozhiQi/RPIN

Follow the official instructions to obtain and prepare:

```text
simb/
  train.hkl
  test.hkl
  train/<video-id>/...
  train/<video-id>.pkl
  test/<video-id>/...
  test/<video-id>.pkl
```

Then register it:

```bash
python tools/prepare_dataset.py \
  --dataset simb \
  --source /path/to/simb
```

The command links the user-provided directory into the locations expected by
the perception and trajectory pipelines.

## Verification only

Use `--check-only` when you do not want to create links or configuration:

```bash
python tools/prepare_dataset.py \
  --dataset obj3d \
  --source /path/to/OBJ3D \
  --check-only
```

## Redistribution policy

Do not commit raw images, videos, arrays, or archives to this repository.
Do not upload mirrors unless the corresponding dataset license explicitly
permits redistribution. Citations in a paper or README do not by themselves
grant redistribution rights.
