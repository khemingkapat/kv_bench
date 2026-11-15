"""Tests for declarative Slurm benchmark suite definitions in src/config/suites.json."""

import json
import pathlib
import unittest

import src.techniques  # noqa: F401 -- ensure all techniques register
from src.core.registry import list_techniques


class SuitesConfigTests(unittest.TestCase):
    def test_suites_json_valid_and_techniques_registered(self) -> None:
        suites_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "config" / "suites.json"
        self.assertTrue(suites_path.exists(), f"Missing {suites_path}")
        with suites_path.open("r", encoding="utf-8") as f:
            suites = json.load(f)

        self.assertIsInstance(suites, dict)
        for required in ("baseline", "wp3_fixed", "settled_eval", "all"):
            self.assertIn(required, suites)

        registered = set(list_techniques())
        for suite_name, cfg in suites.items():
            self.assertIsInstance(cfg, dict, f"Suite {suite_name} must be a dict")
            techs = cfg.get("techniques")
            self.assertIsInstance(techs, list, f"Suite {suite_name} missing techniques list")
            self.assertGreater(len(techs), 0, f"Suite {suite_name} has empty techniques")
            for tech in techs:
                self.assertIn(tech, registered, f"Suite {suite_name} references unregistered technique {tech!r}")


if __name__ == "__main__":
    unittest.main()
