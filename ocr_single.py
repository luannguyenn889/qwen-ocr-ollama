import sys
from pathlib import Path
from time import perf_counter
from ollama import Client

MODEL = "qwen3.5:4b"

PROMPT = """
Convert scanned document to Markdown.
Preserve all text.
Use LaTeX for formulas.
Do not summarize.
Return Markdown only.
""".strip()

def main():
    image_path = Path("test.png").resolve()
    output_path = Path("test.md").resolve()

    if not image_path.is_file():
        print(f"Error: test.png not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"OCR'ing {image_path} with {MODEL}...", flush=True)
    started_at = perf_counter()

    client = Client(host="http://localhost:11434")
    
    response = client.generate(
        model=MODEL,
        prompt=PROMPT,
        images=[str(image_path)],
        think=False,
        stream=False,
        options={
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
        keep_alive="10m",
    )

    raw_text = response.response.strip()

    # Clean potential markdown block wrapper
    clean_text = raw_text
    if clean_text.startswith("```markdown"):
        clean_text = clean_text[len("```markdown"):]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    output_path.write_text(clean_text, encoding="utf-8")
    elapsed = perf_counter() - started_at
    print(f"Completed in {elapsed:.1f} seconds. Output written to {output_path}", flush=True)

if __name__ == "__main__":
    main()
