import unittest

from app.core.layout_detector import detect_columns


class LayoutColumnsTests(unittest.TestCase):
    def test_two_clearly_separated_groups_make_two_columns(self):
        regions = [
            (30, y, 300, y + 20) for y in (20, 60, 100)
        ] + [
            (700, y, 980, y + 20) for y in (20, 60, 100)
        ]
        self.assertEqual(detect_columns(regions, 1000), [(0, 502), (502, 1000)])

    def test_single_column_is_not_split(self):
        regions = [(50, y, 900, y + 20) for y in (20, 60, 100, 140)]
        self.assertEqual(detect_columns(regions, 1000), [(0, 1000)])


if __name__ == "__main__":
    unittest.main()
