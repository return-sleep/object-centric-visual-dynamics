# Contributing

## Before opening a change

Run:

```bash
python -m unittest discover -s tests
python -m py_compile \
  tools/run.py tools/prepare_dataset.py \
  main.py train_derendering.py train_simb_dynamics.py
python tools/run.py --dataset ballsCF --mode reproduce --dry-run
python tools/run.py --dataset obj3d --mode reproduce --dry-run
```

Check that the change does not add:

- raw datasets, archives, checkpoints, predictions, or event logs;
- absolute paths from a personal workstation or cluster;
- credentials, access tokens, email addresses, or review correspondence;
- code or artifacts from the excluded Traffic extension;
- third-party code without its license and attribution.

Do not commit a secret and then delete it in a later commit. If a secret ever
enters git history, rotate it and remove it from the entire history before
publishing.
