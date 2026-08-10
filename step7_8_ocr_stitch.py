"""
Module: step7_8_ocr_stitch.py
Nhiệm vụ: OCR từng trang ảnh tạm thời từ PDF bằng Ollama Qwen và ghép các trang thành một file Markdown duy nhất.
Quy trình:
  1. Quét toàn bộ ảnh dạng page_*.png trong thư mục hiện tại.
  2. OCR từng ảnh bằng mô hình Qwen.
  3. Ghép nội dung các trang và phân tách bằng bình luận <!-- Page X --> ghi vào file test.md.
"""

import sys
from pathlib import Path
from time import perf_counter
# pyrefly: ignore [missing-import]
from ollama import Client

MODEL = "qwen3.5:4b"

PROMPT = """
Convert this scanned document to Markdown format according to the following requirements:

1. Remove unnecessary elements:
   - Automatically detect and remove headers, footers, footnotes, and page numbers to make the content cleaner and clearer.

2. Preserve the original document structure and layout:
   - Maintain heading hierarchies (Heading #, ##, ###, ####), paragraphs, and lists (bulleted/numbered lists).
   - CRITICAL: All question headers (e.g., "Câu 1.", "Câu 2.", "Câu 10.") must be strictly formatted as Heading Level 4: `#### Câu X.`. Do not use bold tags (`**Câu X.**`) or normal text for question headers.
   - Ensure the correct natural reading order of the document. For multi-column layouts or multiple-choice questions side-by-side, read column-by-column and keep the options (A, B, C, D) associated with their correct question. Do not merge text across columns.

3. Recognize special components:
   - Mathematical expressions: Automatically convert all formulas, variables, subscripts (e.g., S_1 to $S_1$), primes (e.g., A' to $A'$), fractions (always use \\frac{num}{den} for vertical fractions), and math symbols into LaTeX format. Use $...$ for inline formulas and $$...$$ for block formulas. Do not repeat characters or duplicate terms.
      * CRITICAL: Never use HTML space entities (such as &nbsp; or &amp;nbsp;) anywhere in the document. Use standard markdown spaces or newlines to separate options. Each mathematical expression must have its own closed pair of dollar signs ($). Never group non-mathematical text, punctuation, labels (like 'B.', 'C.', 'D.') inside a dollar sign pair.
   - Tables: Extract tables accurately. For complex tables (with merged cells or multi-tiered headers), export them as HTML tables (<table>). For simple tables, use the standard Markdown table format.
   - Images and captions: If images or diagrams are detected, represent them as a Markdown image tag with the description inside the square brackets and a placeholder path inside the parentheses, for example: `![Description of the image/diagram](image_placeholder.png)`. Never put descriptive text inside the parentheses.

4. General requirements:
   - Do not summarize the content, keep both Vietnamese and English text exactly as written.
   - Return only the raw Markdown content, do not wrap it inside ```markdown code blocks.
""".strip()

def clean_markdown(text: str) -> str:
    import re
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Clean up HTML space entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;nbsp;", " ")
    
    # Fix math block spanning multiple options (e.g. A. $... B. ...$)
    def repl(match):
        content = match.group(1)
        parts = re.split(r"(\s+[B-D]\.\s+)", content)
        if len(parts) > 1:
            new_parts = []
            for part in parts:
                if re.match(r"^\s+[B-D]\.\s+$", part):
                    new_parts.append(part)
                else:
                    stripped = part.strip()
                    if stripped:
                        new_parts.append(f"${stripped}$")
                    else:
                        new_parts.append(part)
            return "".join(new_parts)
        return match.group(0)

    text = re.sub(r"\$([^$\n]+)\$", repl, text)
    return text

def main():
    root_dir = Path(".").resolve()
    # Find page_X.png files, sort them by page number
    image_paths = sorted(
        list(root_dir.glob("page_*.png")),
        key=lambda p: int(p.stem.split("_")[1])
    )
    
    if not image_paths:
        print("Error: No page_*.png files found in the current directory.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(image_paths)} pages to OCR.")
    
    client = Client(host="http://localhost:11434")
    output_path = Path("test.md").resolve()
    
    with output_path.open("w", encoding="utf-8") as f:
        for idx, img_path in enumerate(image_paths):
            page_num = idx + 1
            print(f"OCR'ing page {page_num}/{len(image_paths)}: {img_path.name}...", flush=True)
            started_at = perf_counter()
            
            response = client.generate(
                model=MODEL,
                prompt=PROMPT,
                images=[str(img_path)],
                think=False,
                stream=False,
                options={
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": 4096,
                },
                keep_alive="10m",
            )
            
            page_markdown = clean_markdown(response.response)
            elapsed = perf_counter() - started_at
            print(f"Page {page_num} completed in {elapsed:.1f} seconds.", flush=True)
            
            # Stitch with separators
            f.write(f"<!-- Page {page_num} -->\n\n")
            f.write(page_markdown)
            f.write("\n\n")
            
    print(f"Stitched PDF OCR output written to {output_path}", flush=True)

if __name__ == "__main__":
    main()
