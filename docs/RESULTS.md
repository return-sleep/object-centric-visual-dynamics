# Checkpoint evaluation

The following results were obtained through `tools/run.py --mode eval` on the complete supplied test splits (188 videos per dataset). These runs evaluate existing checkpoints; they do not verify retraining from scratch.

| Dataset | MSE ×1000 | PSNR | L-PSNR | SSIM | LPIPS-Alex (raw) |
|---|---:|---:|---:|---:|---:|
| ballsCF | 5.973760 | 30.102906 | 34.115024 | 0.946580 | 0.019056 |
| blocktowerCF | 9.798649 | 28.796962 | 32.061722 | 0.943537 | 0.025083 |

Balls uses 3 conditioning frames; Blocktower uses 1. Both predict 25 future frames at 112×112. LPIPS is reported in raw units here; multiply by 100 for tables using ×10⁻² units.

See [validation.json](validation.json) for checkpoint hashes, test-split hashes, source hashes, commands, and environment versions.

## Training and validation

The packaged CoPhy dynamics trainer now selects models on the separate `val` split. The supplied split lists contain 1734/375/188 Balls videos and 1486/376/188 Blocktower videos (train/validation/test), with no overlapping IDs. Retraining from scratch has not been verified for this package.

## Other configurations

SimB checkpoint evaluation is being checked separately. OBJ3D configuration and source are available, but a matching checkpoint/metric protocol has not yet been verified for distribution. No OBJ3D reproduction score is assigned to this package.
