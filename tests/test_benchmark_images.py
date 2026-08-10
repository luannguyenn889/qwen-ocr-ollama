"""Repeatable OCR-to-Markdown benchmark.

Run ``python tests\\test_benchmark_images.py``.  Every image is run three times:
one cold run after unloading the model, then warm runs.  Raw data is written to
``output/image_benchmark/results.csv`` and individual Markdown outputs are kept
per image/run for review.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ocr_metrics import score_markdown
from app.core.ollama_engine import OllamaQwenEngine


IMAGE_NAMES = (
    "01_scan_ro.png", "02_scan_mo.png", "03_nghieng.png", "04_bang.png",
    "05_hai_cot.png", "06_cong_thuc.png", "07_viet_anh.png", "08_photo_cu.png",
    "09_chu_nho.png", "10_layout_phuc_tap.png",
)
IMAGES_DIR = PROJECT_ROOT / "samples" / "images"
GROUND_TRUTH_DIR = PROJECT_ROOT / "samples" / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "output" / "image_benchmark"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"


def average(rows: list[dict[str, object]], name: str) -> float:
    values = [float(row[name]) for row in rows if row.get(name) not in (None, "")]
    return statistics.mean(values) if values else 0.0


def run_benchmark(runs: int) -> list[dict[str, object]]:
    missing = [name for name in IMAGE_NAMES if not (IMAGES_DIR / name).is_file()]
    missing += [name for name in IMAGE_NAMES if not (GROUND_TRUTH_DIR / f"{Path(name).stem}.md").is_file()]
    if missing:
        raise FileNotFoundError("Thiếu ảnh hoặc ground truth: " + ", ".join(missing))

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    engine = OllamaQwenEngine()
    engine.check_connection()
    rows: list[dict[str, object]] = []
    for image_name in IMAGE_NAMES:
        image_path = IMAGES_DIR / image_name
        expected = (GROUND_TRUTH_DIR / f"{image_path.stem}.md").read_text(encoding="utf-8")
        print(f"\nOCR {image_name} ({runs} lượt)", flush=True)
        for run_number in range(1, runs + 1):
            run_kind = "cold" if run_number == 1 else "warm"
            if run_kind == "cold":
                engine.unload()
            try:
                actual, timing = engine.ocr_image_with_metrics(image_path)
                metrics = score_markdown(expected, actual)
                status, error = "OK", ""
            except Exception as exc:
                actual, timing, metrics = "", {}, {}
                status, error = "ERROR", str(exc)
            output_path = MARKDOWN_DIR / f"{image_path.stem}.run_{run_number}.md"
            output_path.write_text(actual + "\n", encoding="utf-8")
            row: dict[str, object] = {
                "image": image_name, "run": run_number, "run_kind": run_kind,
                "status": status, "error": error,
                "markdown": str(output_path.relative_to(PROJECT_ROOT)),
                **timing, **metrics,
            }
            rows.append(row)
            print(
                f"  {run_kind}: {status}; total={float(timing.get('wall_seconds', 0)):.2f}s; "
                f"CER={float(metrics.get('cer', 0)):.2%}; WER={float(metrics.get('wer', 0)):.2%}",
                flush=True,
            )
    return rows


def write_reports(rows: list[dict[str, object]], runs: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Benchmark OCR ảnh", "",
        f"- Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Model: qwen3.5:4b", f"- Lượt mỗi ảnh: {runs} (1 cold, {runs - 1} warm).",
        "- `host_overhead_seconds` = encode ảnh + HTTP cục bộ + phần server chưa được Ollama phân loại.",
        "- CER/WER càng thấp càng tốt; F1 và tỷ lệ Markdown hợp lệ càng cao càng tốt.", "",
        "| Ảnh | Cold (s) | Warm TB (s) | CER | WER | Số F1 | CT F1 | Bảng F1 | DS F1 | Markdown hợp lệ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for image_name in IMAGE_NAMES:
        group = [row for row in rows if row["image"] == image_name and row["status"] == "OK"]
        cold = [row for row in group if row["run_kind"] == "cold"]
        warm = [row for row in group if row["run_kind"] == "warm"]
        report.append(
            f"| {image_name} | {average(cold, 'wall_seconds'):.2f} | {average(warm, 'wall_seconds'):.2f} "
            f"| {average(group, 'cer'):.2%} | {average(group, 'wer'):.2%} "
            f"| {average(group, 'numbers_f1'):.2%} | {average(group, 'formulas_f1'):.2%} "
            f"| {average(group, 'table_f1'):.2%} | {average(group, 'list_f1'):.2%} "
            f"| {average(group, 'markdown_valid'):.2%} |"
        )
    report.extend(["", "## Dữ liệu thô", "", "`results.csv` lưu từng lượt: wall/load/prompt-eval/eval/host-overhead, prompt và generated token, tất cả chỉ số chất lượng, lỗi và đường dẫn Markdown."])
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR Markdown có cold/warm và quality metrics.")
    parser.add_argument("--runs", type=int, default=3, help="Số lượt mỗi ảnh, tối thiểu 3.")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs phải từ 3 trở lên")
    rows = run_benchmark(args.runs)
    write_reports(rows, args.runs)
    print(f"\nĐã lưu: {OUTPUT_DIR / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
