import sys
import re
from pathlib import Path

class OCRValidator:
    def __init__(self):
        pass

    def validate_file(self, file_path: Path) -> dict:
        content = file_path.read_text(encoding="utf-8").strip()
        results = {
            "file": file_path.name,
            "is_empty": False,
            "markdown_errors": [],
            "formula_errors": [],
            "table_errors": [],
            "hallucinations": [],
        }

        # 1. Check for empty files
        if not content:
            results["is_empty"] = True
            return results

        # 2. Check for Markdown formatting issues (e.g. unclosed code blocks, unmatched asterisks/backticks)
        # Unclosed code blocks
        code_blocks = re.findall(r"```", content)
        if len(code_blocks) % 2 != 0:
            results["markdown_errors"].append("Unclosed code block (odd number of triple backticks)")

        # Unclosed inline code backticks
        # We search lines for single backticks
        for line_num, line in enumerate(content.splitlines(), 1):
            inline_backticks = re.findall(r"`", line)
            if len(inline_backticks) % 2 != 0 and "```" not in line:
                results["markdown_errors"].append(f"Line {line_num}: Unclosed inline code backtick")

        # 3. Check for formula errors (unmatched $ or $$, or basic LaTeX brace mismatches)
        # Count total $ signs. An odd number of single $ or $$ might indicate a formula issue.
        # But wait, $$ contains two $ signs, so let's check $$ first, then single $.
        double_dollars = content.count("$$")
        # Subtract double dollars to get remaining single dollars
        single_dollars = content.count("$") - (double_dollars * 2)
        if single_dollars % 2 != 0:
            results["formula_errors"].append("Unmatched single dollar ($) formula marker")
        if double_dollars % 2 != 0:
            results["formula_errors"].append("Unmatched double dollar ($$) formula marker")

        # Basic LaTeX brace matching inside formulas
        # Find all formulas
        formulas = re.findall(r"\$(.*?)\$", content)
        for idx, formula in enumerate(formulas):
            # Check matching of { and }
            open_braces = formula.count("{")
            close_braces = formula.count("}")
            if open_braces != close_braces:
                results["formula_errors"].append(f"Brace mismatch in formula: {formula[:40]}... ({open_braces} open, {close_braces} close)")

        # 4. Check for table errors
        # Markdown table column count matching
        lines = content.splitlines()
        in_table = False
        table_cols = 0
        table_start_line = 0
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            # A markdown table line starts and ends with | (usually) or contains multiple |
            is_table_row = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") > 1
            if is_table_row:
                cols = stripped.count("|") - 1
                if not in_table:
                    in_table = True
                    table_cols = cols
                    table_start_line = line_num
                else:
                    if cols != table_cols:
                        # Sometimes separator lines look like |---|---| which is fine.
                        # Let's skip checking columns if it's a separator line.
                        if not re.match(r"^\|[\s\-\:\+\|]*\|$", stripped):
                            results["table_errors"].append(
                                f"Table starting at line {table_start_line}: Row at line {line_num} has {cols} columns, expected {table_cols}"
                            )
            else:
                in_table = False

        # HTML Table tags matching
        html_tags = ["table", "tr", "td", "th"]
        for tag in html_tags:
            open_count = len(re.findall(rf"<{tag}\b", content, re.IGNORECASE))
            close_count = len(re.findall(rf"</{tag}>", content, re.IGNORECASE))
            if open_count != close_count:
                results["table_errors"].append(f"HTML Table mismatch: <{tag}> tag opened {open_count} times but closed {close_count} times")

        # 5. Check for Hallucinations (e.g. repeated lines/paragraphs, or huge length, repeated characters)
        # Check for consecutive duplicate lines
        dup_count = 0
        last_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == last_line:
                dup_count += 1
            last_line = stripped
        if dup_count > 3:
            results["hallucinations"].append(f"High line repetition detected ({dup_count} duplicate lines)")

        # Check for long repeated substrings (like word babbling)
        # E.g. "word word word word..."
        for line_num, line in enumerate(lines, 1):
            words = line.split()
            if len(words) > 10:
                # Find if any word is repeated more than 5 times consecutively
                consec = 0
                prev_word = ""
                for w in words:
                    if w.lower() == prev_word.lower():
                        consec += 1
                    else:
                        consec = 0
                    if consec > 4:
                        results["hallucinations"].append(f"Line {line_num}: Word repetition ('{w}')")
                        break

        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate OCR markdown results")
    parser.add_argument("path", help="Path to a markdown file or directory containing markdown files")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"Error: Path {path} does not exist.", file=sys.stderr)
        sys.exit(1)

    files_to_check = []
    if path.is_file():
        files_to_check.append(path)
    else:
        files_to_check.extend(path.glob("**/*.md"))

    if not files_to_check:
        print("No markdown files found to validate.")
        sys.exit(0)

    validator = OCRValidator()
    print(f"Validating {len(files_to_check)} markdown files...\n")
    
    any_errors = False
    for file in files_to_check:
        res = validator.validate_file(file)
        print(f"=== File: {res['file']} ===")
        if res["is_empty"]:
            print("  [WARNING] File is empty.")
            any_errors = True
            continue

        errors_found = False
        for category in ["markdown_errors", "formula_errors", "table_errors", "hallucinations"]:
            if res[category]:
                print(f"  [ERROR] {category.replace('_', ' ').capitalize()}:")
                for err in res[category]:
                    print(f"    - {err}")
                errors_found = True
                any_errors = True
        
        if not errors_found:
            print("  [OK] No errors detected.")
            
    if any_errors:
        sys.exit(1)
    else:
        print("\nAll validations passed successfully!")

if __name__ == "__main__":
    main()
