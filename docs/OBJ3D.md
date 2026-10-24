# OBJ3D inference

The paper's OBJ3D result uses a fixed ensemble of prediction streams and a mean of 16 stochastic rollouts. The inference configuration is stored in `configs/obj3d_inference.json`; it is applied unchanged at test time.

## Checkpoints and prediction

Place the perception checkpoint and its architecture configuration in `checkpoints/obj3d/perception/`. The five dynamics checkpoints are stored in the `champ`, `r9_full`, `r6`, `motion`, and `nv3` subdirectories. The residual-corrected and stochastic streams reuse the base (`champ`) checkpoint.

```bash
python tools/run_obj3d_inference.py \
  --data-root /path/to/OBJ3D \
  --output-root runs/obj3d-inference \
  --gpu 0
```

This command runs the fixed prediction streams and writes the combined arrays to `runs/obj3d-inference/predictions/`. The historical single-model entry point in `tools/run.py --dataset obj3d` does not apply this ensemble.

The stochastic stream averages 16 rollouts with noise standard deviation 0.005 and seeds 1 through 16. The static-residual stream uses the six conditioning frames only, with strength 0.5, motion threshold 0.02, and dilation 7. The three temporal intervals use the frozen validation-set coefficients in the configuration file. No fitting is performed by the inference command.

## Evaluation

```bash
python tools/evaluate_obj3d_predictions.py \
  --pred-dir runs/obj3d-inference/predictions \
  --data-root /path/to/OBJ3D \
  --output runs/obj3d-inference/metrics.json
```

Each clip provides 6 conditioning frames followed by 44 predicted frames at 128×128. Metrics use PIQA PSNR, SSIM, and LPIPS-VGG on RGB values in `[0,1]`, averaged over the predicted frames and test clips. Prediction IDs must cover the complete requested test split. The supplied dataset contains 200 validation and 200 test clips.

Re-evaluation of the saved paper predictions over all 200 test clips gives PSNR 30.751619, SSIM 0.922713, and LPIPS-VGG 0.281065. These round to the paper's 30.75 / 0.923 / 0.281. This is a saved-prediction check; complete regeneration from the packaged checkpoints is a separate verification step.

The complete 22-run checkpoint-to-prediction workflow and this metric entry point have also been exercised on test episodes 0 and 1. Mean absolute RGB differences from their saved paper predictions were 0.0000350 and 0.0000495. Full-test-set regeneration from the packaged checkpoints remains pending.
