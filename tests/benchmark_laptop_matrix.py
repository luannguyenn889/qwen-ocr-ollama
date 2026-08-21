"""Benchmark DPI/worker matrix on 0001.pdf, 0002.pdf and 0003.pdf.

Dry run (validates inputs without OCR):
    python tests/benchmark_laptop_matrix.py --input res --dry-run

Full matrix:
    python tests/benchmark_laptop_matrix.py --input <pdf-folder> --runs 1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from ollama import Client

from app.core.batch_ocr import OLLAMA_REQUEST_TIMEOUT_SECONDS, pdf_page_count, process_single_pdf
from app.core.ocr_metrics import score_markdown
from app.core.quality_gate import evaluate_page


CASES = ("0001.pdf", "0002.pdf", "0003.pdf")
CONFIGS = (
    ("A", 200, 1), ("B", 250, 1), ("C", 300, 1),
    ("D", 200, 2), ("E", 250, 2), ("F", 300, 2),
)


class Tee(io.TextIOBase):
    def __init__(self, *targets):
        self.targets = targets

    def write(self, value):
        for target in self.targets:
            target.write(value)
            target.flush()
        return len(value)

    def flush(self):
        for target in self.targets:
            target.flush()


class ResourceMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples: list[tuple[float, float, float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _vram_mb() -> float:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return sum(float(value.strip()) for value in result.stdout.splitlines() if value.strip())
        except Exception:
            return 0.0

    def _run(self):
        process = psutil.Process()
        while not self.stop.wait(0.5):
            ollama_mb = 0.0
            for item in psutil.process_iter(["name", "memory_info"]):
                try:
                    if "ollama" in (item.info["name"] or "").casefold():
                        ollama_mb += item.info["memory_info"].rss / 1024**2
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples.append((process.memory_info().rss / 1024**2, ollama_mb, self._vram_mb()))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=3)

    def peaks(self):
        return {
            "python_ram_mb_peak": max((row[0] for row in self.samples), default=0.0),
            "ollama_ram_mb_peak": max((row[1] for row in self.samples), default=0.0),
            "vram_mb_peak": max((row[2] for row in self.samples), default=0.0),
        }


def validate_cases(input_dir: Path) -> list[Path]:
    missing = [name for name in CASES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu {', '.join(missing)} trong {input_dir}. "
            "Hãy chép đúng ba PDF scan vào thư mục này."
        )
    return [input_dir / name for name in CASES]


def run_matrix(input_dir: Path, output_dir: Path, model: str, runs: int) -> list[dict]:
    pdfs = validate_cases(input_dir)
    client = Client(host="http://localhost:11434", timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS)
    client.list()
    rows = []
    for code, dpi, workers in CONFIGS:
        for run_number in range(1, runs + 1):
            for pdf in pdfs:
                destination = output_dir / code / f"run_{run_number}"
                destination.mkdir(parents=True, exist_ok=True)
                log_buffer = io.StringIO()
                started = time.perf_counter()
                status, error, result = "OK", "", None
                with ResourceMonitor() as monitor:
                    try:
                        with contextlib.redirect_stdout(Tee(sys.__stdout__, log_buffer)):
                            result = process_single_pdf(
                                pdf, destination, client, model,
                                workers=workers, render_dpi=dpi,
                            )
                    except Exception as exception:
                        status, error = "ERROR", str(exception)
                elapsed = time.perf_counter() - started
                log = log_buffer.getvalue()
                (destination / f"{pdf.stem}.log.txt").write_text(log, encoding="utf-8")
                pages = pdf_page_count(pdf)
                markdown = result.read_text(encoding="utf-8") if result and result.is_file() else ""
                report = evaluate_page(markdown, destination) if markdown else None
                metrics = {}
                reference = input_dir / "expected" / f"{pdf.stem}.md"
                if markdown and reference.is_file():
                    metrics = score_markdown(reference.read_text(encoding="utf-8"), markdown)
                lower = (log + "\n" + error).casefold()
                row = {
                    "config": code, "dpi": dpi, "workers": workers, "run": run_number,
                    "pdf": pdf.name, "pages": pages, "status": status, "error": error,
                    "elapsed_seconds": elapsed,
                    "seconds_per_page": elapsed / pages if pages else 0,
                    "retry_count": lower.count("quality retry"),
                    "timeout_count": lower.count("timed out") + lower.count("timeout"),
                    "cuda_oom_count": lower.count("cuda error: out of memory"),
                    "quality_score": report.score if report else 0,
                    "quality_errors": ";".join(report.errors) if report else "no_output",
                    "output": str(result) if result else "",
                    **monitor.peaks(), **metrics,
                }
                rows.append(row)
                print(
                    f"{code} | {pdf.name} | {elapsed:.1f}s | "
                    f"{row['seconds_per_page']:.1f}s/page | {status}", flush=True,
                )
    return rows


def write_reports(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for code, dpi, workers in CONFIGS:
        group = [row for row in rows if row["config"] == code]
        ok = [row for row in group if row["status"] == "OK"]
        disqualified = any(
            row["status"] != "OK" or row["cuda_oom_count"] or row["seconds_per_page"] > 120
            for row in group
        )
        summary.append({
            "config": code, "dpi": dpi, "workers": workers,
            "status": "LOẠI" if disqualified else "ĐẠT",
            "seconds_per_page_avg": sum(row["seconds_per_page"] for row in ok) / len(ok) if ok else 0,
            "quality_score_avg": sum(row["quality_score"] for row in ok) / len(ok) if ok else 0,
            "vram_mb_peak": max((row["vram_mb_peak"] for row in group), default=0),
            "retry_total": sum(row["retry_count"] for row in group),
            "timeout_total": sum(row["timeout_count"] for row in group),
            "cuda_oom_total": sum(row["cuda_oom_count"] for row in group),
        })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark laptop: DPI 200/250/300 × workers 1/2.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "test_input")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "laptop_matrix")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra đủ testcase và in ma trận.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs phải >= 1")
    pdfs = validate_cases(args.input.resolve())
    if args.dry_run:
        print("Testcase:", ", ".join(path.name for path in pdfs))
        for code, dpi, workers in CONFIGS:
            print(f"{code}: DPI={dpi}, workers={workers}")
        return
    rows = run_matrix(args.input.resolve(), args.output.resolve(), args.model, args.runs)
    write_reports(rows, args.output.resolve())
    print(f"Đã lưu kết quả tại: {args.output.resolve()}")


if __name__ == "__main__":
    main()
