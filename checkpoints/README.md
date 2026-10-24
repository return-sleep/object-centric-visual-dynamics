# Model checkpoints

Model weights are not stored in git. Place matching files at the exact paths
below:

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

Before publishing weights:

1. upload only the selected files to a versioned release or model host;
2. record the public URLs here;
3. generate `MANIFEST.sha256`;
4. run every public `--mode eval` command in a clean environment;
5. archive the resulting metric files and exact evaluation command with the release.
   The main.py path writes `metric_table.csv` and `log.json`.

Example hash command:

```bash
python tools/checkpoint_manifest.py --root checkpoints > checkpoints/MANIFEST.sha256
python tools/checkpoint_manifest.py --root checkpoints --verify checkpoints/MANIFEST.sha256
```

Until download URLs are populated, `tools/run.py --mode eval` exits with a
clear missing-checkpoint message. Full training does not require public
weights.
