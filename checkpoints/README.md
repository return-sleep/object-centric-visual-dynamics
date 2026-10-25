# Model checkpoints

## Download and evaluate

Each archive contains matching perception and dynamics weights, checksums, instructions, and a reference result file. Extract the archives into the repository root.

| Dataset | Archive | Checksum file |
|---|---|---|
| Balls | [ocvd-balls-checkpoints-v1.zip](https://github.com/return-sleep/object-centric-visual-dynamics/releases/download/v1.0.0-balls/ocvd-balls-checkpoints-v1.zip) | `SHA256SUMS` |
| Blocktower | [ocvd-blocktower-checkpoints-v1.zip](https://github.com/return-sleep/object-centric-visual-dynamics/releases/download/v1.0.0-blocktower/ocvd-blocktower-checkpoints-v1.zip) | `BLOCKTOWER_SHA256SUMS` |

```sh
unzip ocvd-balls-checkpoints-v1.zip
unzip ocvd-blocktower-checkpoints-v1.zip
shasum -a 256 -c SHA256SUMS
shasum -a 256 -c BLOCKTOWER_SHA256SUMS
```

Archive SHA-256:

- Balls: `7634999fb4804bf8d3f4e1e0c0deb11d627d09dd805ccce3cf8b205d88081b9e`
- Blocktower: `07e549f32741cd0744268145fe51a64c2eb59b4665a0c78a52544db8bbe4ffbe`

After [registering the datasets](../docs/DATASETS.md), run:

```sh
python tools/run.py --dataset ballsCF --mode eval
python tools/run.py --dataset blocktowerCF --mode eval
```

See [evaluation results](../docs/RESULTS.md) and [validation details](../docs/validation.json) for the reference results, test protocol, and evaluated environment.

## File layout

Weights are distributed as release assets rather than stored in Git. Balls and Blocktower weights are available above; SimB and OBJ3D do not yet have public weight downloads. Expected paths:

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

Use the archive-specific checksum files above. The repository's `MANIFEST.sha256` also lists unreleased checkpoints and is not the verification manifest for these downloads.
