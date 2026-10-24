# Learning Visual Dynamics via Unsupervised Object-centric Prediction

**Huilin Xu and Feng Xu**

PyTorch code for learning object-centric representations and predicting visual dynamics through contextual and relational modeling.

The model first learns object states and image reconstruction. A dynamics module combines object states, spatial context, and motion history to predict future states, which are decoded into video frames or evaluated as trajectories.

## Setup

```bash
conda env create -f environment.yml
conda activate ocvd
```

The supplied environment specifies Python 3.7, PyTorch 1.11, torchvision 0.12, and CUDA 11.3. Python dependencies are listed in `requirements.txt`.

## Dataset preparation

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

## Training

Preview the two-stage workflow:

```bash
python tools/run.py --dataset ballsCF --mode reproduce --dry-run
```

Train perception and dynamics, then evaluate the selected model:

```bash
python tools/run.py --dataset ballsCF --mode reproduce --gpu 0
```

Dataset-specific training arguments are stored in `configs/`. The default seed is 1024; training outputs are written under `runs/<dataset>/`.

## Evaluation

Evaluation requires a matching perception checkpoint and dynamics checkpoint. Prepared checkpoint pairs cover Balls, Blocktower, and SimB. Public download links will be added when the release is uploaded. Balls and Blocktower have been evaluated on all 188 supplied test videos; see [evaluation results](docs/RESULTS.md). Once matching files have been placed at the paths specified in [checkpoints/README.md](checkpoints/README.md), run:

```bash
python tools/run.py --dataset ballsCF --mode eval --gpu 0
```

For the `main.py` evaluation path, results are saved as `metric_table.csv` and `log.json` beside the dynamics checkpoint. A dry run only prints commands; it does not validate model outputs or reproduce numerical results.


OBJ3D uses a separate fixed-ensemble inference command; see [OBJ3D inference and evaluation](docs/OBJ3D.md).

## Repository layout

```text
configs/                 Dataset-specific training configurations
tools/                   Dataset registration and workflow entry points
src/                     Perception and dynamics modules
Dataloader/              Dataset loaders
checkpoints/             Checkpoint placement instructions
docs/                    Dataset and experiment documentation
```

## Acknowledgements

This work builds on Filtered-CoPhy and Interaction Networks and uses benchmarks introduced by CoPhy, G-SWM, and RPIN. Please retain the corresponding third-party acknowledgements and license notices.

## Citation and license

Paper: **Learning Visual Dynamics via Unsupervised Object-centric Prediction**, Huilin Xu and Feng Xu.

See [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), and [NOTICE](NOTICE) for citation metadata and license information.
