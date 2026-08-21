import tempfile
import unittest
from pathlib import Path

from tests.benchmark_laptop_matrix import CASES, CONFIGS, validate_cases


class LaptopBenchmarkTests(unittest.TestCase):
    def test_matrix_contains_requested_six_configs(self):
        self.assertEqual(
            CONFIGS,
            (("A", 200, 1), ("B", 250, 1), ("C", 300, 1),
             ("D", 200, 2), ("E", 250, 2), ("F", 300, 2)),
        )

    def test_requires_all_three_named_pdf_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in CASES:
                (root / name).write_bytes(b"%PDF-test")
            self.assertEqual([path.name for path in validate_cases(root)], list(CASES))
            (root / "0002.pdf").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "0002.pdf"):
                validate_cases(root)


if __name__ == "__main__":
    unittest.main()
