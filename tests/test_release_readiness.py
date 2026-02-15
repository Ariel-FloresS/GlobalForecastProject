"""Pruebas mínimas de readiness para release v1.0.0."""

from pathlib import Path
import unittest


class TestReleaseReadiness(unittest.TestCase):
    def test_readme_exists_and_has_title(self) -> None:
        readme_path = Path("README.md")
        self.assertTrue(readme_path.exists(), "README.md no existe")

        content = readme_path.read_text(encoding="utf-16")
        self.assertIn("Global Forecast Project", content)

    def test_release_checklist_exists(self) -> None:
        checklist_path = Path("docs/release_v1_0_0_checklist.md")
        self.assertTrue(checklist_path.exists(), "Falta checklist de release")

    def test_release_checklist_mentions_tests(self) -> None:
        checklist_path = Path("docs/release_v1_0_0_checklist.md")
        content = checklist_path.read_text(encoding="utf-8")
        self.assertIn("pruebas", content.lower())
        self.assertIn("v1.0.0", content)


if __name__ == "__main__":
    unittest.main()
