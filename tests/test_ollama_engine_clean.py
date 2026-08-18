import unittest

from app.core.ollama_engine import OllamaQwenEngine


class OllamaEngineCleanTests(unittest.TestCase):
    def test_answer_labels_are_not_wrapped_as_math(self):
        cleaned = OllamaQwenEngine.clean("$A. x=1 B. x=2 C. x=3 D. x=4$")
        for label in "ABCD":
            self.assertNotIn(f"${label}.$", cleaned)
            self.assertIn(f"{label}. ", cleaned)
        self.assertEqual(cleaned, "A. $x=1$\nB. $x=2$\nC. $x=3$\nD. $x=4$")

    def test_label_after_expression_stays_outside_math(self):
        cleaned = OllamaQwenEngine.clean("$x=1 B. x=2$")
        self.assertEqual(cleaned, "$x=1$\nB. $x=2$")

    def test_regular_formula_is_unchanged(self):
        self.assertEqual(OllamaQwenEngine.clean("Giá trị $x=1$."), "Giá trị $x=1$.")


if __name__ == "__main__":
    unittest.main()
