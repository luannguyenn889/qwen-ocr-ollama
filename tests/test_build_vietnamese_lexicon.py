import tempfile
import unittest
from pathlib import Path

from scripts.build_vietnamese_lexicon import build, read_entries


class BuildVietnameseLexiconTests(unittest.TestCase):
    def test_filters_and_builds_reproducible_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Viet74K.txt"
            source.write_text(
                "nghiên cứu\nCỪU\ncứu\nthree token entry\n121/2026\n",
                encoding="utf-8",
            )
            entries = read_entries(source)
        payload = build(entries)
        self.assertEqual(entries, ["cứu", "cừu", "nghiên cứu"])
        self.assertEqual(payload["accent_candidates"]["cuu"], ["cứu", "cừu"])

        self.assertEqual(payload["bigrams"], ["nghiên cứu"])
        self.assertEqual(payload["source"]["license"], "GPL-2.0")


if __name__ == "__main__":
    unittest.main()
