# Learning Visual Dynamics via Unsupervised Object-centric Prediction

## 💡 Introduction

PyTorch implementation of **Learning Visual Dynamics via Unsupervised Object-centric Prediction**. This framework learns object-centric representations from videos and predicts visual dynamics through contextual and relational modeling.

The model first learns object states and image reconstruction. A dynamics module combines object states, spatial context, and motion history to predict future states, which are decoded into video frames or evaluated as trajectories.

## 📑 Contents

- [Installation](#installation)
- [Dataset preparation](#datasets)
- [Training](#training)
- [Evaluation and checkpoints](#evaluation)
- [Repository layout](#layout)
- [Acknowledgements](#acknowledgements)
- [Citation and license](#citation)

<a id="installation"></a>

## 🛠️ Installation

```bash
git clone https://github.com/return-sleep/object-centric-visual-dynamics.git
cd object-centric-visual-dynamics

conda env create -f environment.yml
conda activate ocvd
```

The supplied environment specifies Python 3.7, PyTorch 1.11, torchvision 0.12, and CUDA 11.3. Python dependencies are listed in `requirements.txt`.

<a id="datasets"></a>

## 📦 Dataset preparation

Obtain datasets from their original providers and follow [the preparation instructions](docs/DATASETS.md). The configuration files currently cover:

| Dataset | Task | Conditioning frames | Prediction frames |
|---|---|---:|---:|
| Balls | Video prediction | 3 | 25 |
| Blocktower | Video prediction | 1 | 25 |
| SimB | Trajectory prediction | 4 | 20 |
| OBJ3D | Video prediction | 6 | 14 for training; 44 for evaluation |

Command-line identifiers are `ballsCF`, `blocktowerCF`, `simb`, and `obj3d`. 

Register the local dataset before running training or evaluation:

```bash
python tools/prepare_dataset.py --dataset ballsCF --source /path/to/CoPhy_112
```

<a id="training"></a>

## 🚀 Training

Preview the two-stage workflow:

```bash
python tools/run.py --dataset ballsCF --mode reproduce --dry-run
```

Train perception and dynamics, then evaluate the selected model:

```bash
python tools/run.py --dataset ballsCF --mode reproduce --gpu 0
```

Dataset-specific training arguments are stored in `configs/`. The default seed is 1024; training outputs are written under `runs/<dataset>/`.

<a id="evaluation"></a>

## 🧪 Evaluation and checkpoints

Evaluation requires matching perception and dynamics checkpoints. Checkpoint downloads are being prepared for GitHub Releases; see [checkpoint placement](checkpoints/README.md) for the expected directory structure.

Balls and Blocktower have each been evaluated on all 188 supplied test videos. See [evaluation results and validation scope](docs/RESULTS.md) for the measured metrics and current verification status.

After placing the checkpoints, run:

```bash
python tools/run.py --dataset ballsCF --mode eval --gpu 0
```

For the `main.py` evaluation path, results are saved as `metric_table.csv` and `log.json` beside the dynamics checkpoint. A dry run only prints commands; it does not validate model outputs or reproduce numerical results.

OBJ3D uses a separate fixed-ensemble inference command; see [OBJ3D inference and evaluation](docs/OBJ3D.md).

<a id="layout"></a>

## 🗂️ Repository layout

```text
configs/                 Dataset-specific training configurations
tools/                   Dataset registration and workflow entry points
src/                     Perception and dynamics modules
Dataloader/              Dataset loaders
checkpoints/             Checkpoint placement instructions
docs/                    Dataset and experiment documentation
```

<a id="acknowledgements"></a>

## 🤝 Acknowledgements

This work builds on Filtered-CoPhy and Interaction Networks and uses benchmarks introduced by CoPhy, G-SWM, and RPIN. Please retain the corresponding third-party acknowledgements and license notices.

<a id="citation"></a>

## 📄 Citation and license

If you use this code in your research, please cite the associated paper using [CITATION.cff](CITATION.cff).

The original code is released under the [MIT License](LICENSE). See [NOTICE](NOTICE) for third-party acknowledgements and terms.
