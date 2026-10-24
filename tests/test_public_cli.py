import importlib.util
import os
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "public_run", os.path.join(ROOT, "tools", "run.py"))
PUBLIC_RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLIC_RUN)


def options(dataset):
    return SimpleNamespace(
        dataset=dataset,
        mode="reproduce",
        stage="all",
        data_root=os.path.join(ROOT, "data", dataset),
        output_root=os.path.join(ROOT, "runs"),
        checkpoint_root=os.path.join(ROOT, "checkpoints"),
        gpu=0,
        seed=1024,
        epochs=1,
        max_videos=2,
        dry_run=True,
    )


class PublicCliTest(unittest.TestCase):
    def test_all_configs_load(self):
        for dataset in PUBLIC_RUN.SUPPORTED:
            config = PUBLIC_RUN.load_config(dataset)
            self.assertEqual(config["dataset"], dataset)
            self.assertIn("perception", config)
            self.assertIn("dynamics", config)

    def test_commands_use_deterministic_outputs(self):
        for dataset in PUBLIC_RUN.SUPPORTED:
            config = PUBLIC_RUN.load_config(dataset)
            paths = PUBLIC_RUN.checkpoint_paths(
                config, os.path.join(ROOT, "runs"))
            command = PUBLIC_RUN.stage_command(
                config, "dynamics", paths, options(dataset))
            self.assertIn("--output_dir", command)
            self.assertIn("--Perception_model_weight_path", command)
            self.assertNotIn("cuda:7", command)

    def test_obj3d_eval_uses_long_horizon(self):
        config = PUBLIC_RUN.load_config("obj3d")
        paths = PUBLIC_RUN.checkpoint_paths(
            config, os.path.join(ROOT, "checkpoints"))
        command = PUBLIC_RUN.stage_command(
            config, "dynamics", paths, options("obj3d"), evaluate=True)
        horizon = command[command.index("--horizon") + 1]
        self.assertEqual(horizon, "44")
        self.assertNotIn("--random_window", command)

    def test_checkpoint_layout(self):
        config = PUBLIC_RUN.load_config("ballsCF")
        perception, dynamics, dynamics_dir = PUBLIC_RUN.checkpoint_paths(
            config, "/tmp/checkpoints")
        self.assertTrue(perception.endswith(
            "ballsCF/perception/best_val_ballsCF_4kpts.nn"))
        self.assertTrue(dynamics.endswith(
            "ballsCF/dynamics/best_val_model.pt"))
        self.assertEqual(os.path.dirname(dynamics), dynamics_dir)


if __name__ == "__main__":
    unittest.main()
