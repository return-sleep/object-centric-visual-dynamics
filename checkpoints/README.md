# Model checkpoints

## Download and evaluate Balls

Download [ocvd-balls-checkpoints-v1.zip](https://github.com/return-sleep/object-centric-visual-dynamics/releases/download/v1.0.0-balls/ocvd-balls-checkpoints-v1.zip) (27.4 MB). It includes the matching perception and dynamics weights, file checksums, and reference metrics. Extract it at the repository root:

```sh
unzip ocvd-balls-checkpoints-v1.zip
shasum -a 256 -c SHA256SUMS
```

Archive SHA-256: `7634999fb4804bf8d3f4e1e0c0deb11d627d09dd805ccce3cf8b205d88081b9e`.

After [registering the dataset](../docs/DATASETS.md), run:

```sh
python tools/run.py --dataset ballsCF --mode eval
```

This evaluates all 188 supplied Balls test videos using 3 conditioning frames and 25 predicted frames. The reference L-PSNR is 34.115024 dB, rounding to 34.12. See [results and environment details](../docs/RESULTS.md).

## File layout

Weights are distributed as release assets rather than stored in Git. Balls weights are available above; the other datasets do not yet have public weight downloads. Expected paths:

```text
checkpoints/
  ballsCF/
    perception/best_val_ballsCF_4kpts.nn
    dynamics/best_val_model.pt
  blocktowerCF/
    perception/best_val_blocktowerCF_4kpts.nn
    dynamics/best_val_model.pt
  simb/
    perception/best_val_simb_4kpts.nn
    dynamics/best_val_model.pt
  obj3d/
    perception/best_val_obj3d_4kpts.nn
    dynamics/best_val_psnr_model.pt
```

The Balls archive has its own `SHA256SUMS`, covering only its two weights. The repository's `MANIFEST.sha256` also lists unreleased checkpoints and is not the verification manifest for this download.
