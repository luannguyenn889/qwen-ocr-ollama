"""Benchmark Qwen-only versus Paddle-layout + Qwen on the same PDFs.

Run:
    python tests/benchmark_qwen_vs_hybrid.py --runs 1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ImportError as error:
    raise SystemExit("Thiếu psutil. Chạy: pip install psutil") from error

from ollama import Client
from app.core import batch_ocr

MODES = (
    ("qwen", "qwen3.5:4b", False),
    ("hybrid", "Hybrid (Paddle layout + qwen3.5:4b)", True),
)


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread: threading.Thread | None = None

    @staticmethod
    def _gpu() -> tuple[float, float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            rows = [row.split(",") for row in result.stdout.strip().splitlines() if row.strip()]
            return max(float(row[0]) for row in rows), sum(float(row[1]) for row in rows)
        except Exception:
            return 0.0, 0.0

    def _sample(self) -> None:
        python = psutil.Process()
        while not self.stop_event.wait(self.interval):
            ollama_rss = ollama_cpu = 0.0
            for process in psutil.process_iter(["name", "memory_info", "cpu_percent"]):
                try:
                    if "ollama" in (process.info["name"] or "").casefold():
                        ollama_rss += process.info["memory_info"].rss
                        ollama_cpu += process.info["cpu_percent"]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            gpu, vram = self._gpu()
            self.samples.append({
                "python_ram_mb": python.memory_info().rss / 1024**2,
                "python_cpu_percent": python.cpu_percent(),
                "ollama_ram_mb": ollama_rss / 1024**2,
                "ollama_cpu_percent": ollama_cpu,
                "system_ram_percent": psutil.virtual_memory().percent,
                "system_cpu_percent": psutil.cpu_percent(),
                "gpu_percent": gpu,
                "vram_mb": vram,
            })

    def start(self) -> None:
        psutil.cpu_percent()
        psutil.Process().cpu_percent()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, float]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        metrics: dict[str, float] = {"sample_count": float(len(self.samples))}
        for name in (
            "python_ram_mb", "python_cpu_percent", "ollama_ram_mb", "ollama_cpu_percent",
            "system_ram_percent", "system_cpu_percent", "gpu_percent", "vram_mb",
        ):
            values = [sample[name] for sample in self.samples]
            metrics[f"{name}_avg"] = sum(values) / len(values) if values else 0.0
            metrics[f"{name}_peak"] = max(values, default=0.0)
        return metrics


def pdf_inputs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.casefold() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.pdf"))
    return []


def run(input_path: Path, output_root: Path, runs: int) -> list[dict[str, object]]:
    pdfs = pdf_inputs(input_path)
    if not pdfs:
        raise FileNotFoundError(f"Không tìm thấy PDF tại: {input_path}")
    client = Client(host="http://localhost:11434", timeout=600)
    client.list()
    rows: list[dict[str, object]] = []
    original_layout = batch_ocr.ENABLE_LAYOUT_DETECTION
    try:
        for mode, model, enable_layout in MODES:
            batch_ocr.ENABLE_LAYOUT_DETECTION = enable_layout
            for run_number in range(1, runs + 1):
                for pdf in pdfs:
                    destination = output_root / mode / f"run_{run_number}"
                    monitor = ResourceMonitor()
                    monitor.start()
                    started = time.perf_counter()
                    status, error = "OK", ""
                    try:
                        result = batch_ocr.process_single_pdf(pdf, destination, client, model, workers=1)
                    except Exception as exception:
                        result, status, error = None, "ERROR", str(exception)
                    elapsed = time.perf_counter() - started
                    row: dict[str, object] = {
                        "mode": mode, "model": model, "layout_enabled": enable_layout,
                        "run": run_number, "pdf": pdf.name, "status": status, "error": error,
                        "elapsed_seconds": elapsed,
                        "output": str(result) if result else "",
                        **monitor.finish(),
                    }
                    rows.append(row)
                    print(f"{mode} | {pdf.name} | {status} | {elapsed:.2f}s", flush=True)
    finally:
        batch_ocr.ENABLE_LAYOUT_DETECTION = original_layout
    return rows


def write_reports(rows: list[dict[str, object]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for mode, _, _ in MODES:
        group = [row for row in rows if row["mode"] == mode and row["status"] == "OK"]
        summary[mode] = {
            "successful_runs": len(group),
            "elapsed_seconds_avg": sum(float(row["elapsed_seconds"]) for row in group) / len(group) if group else 0,
            "python_ram_mb_peak": max((float(row["python_ram_mb_peak"]) for row in group), default=0),
            "ollama_ram_mb_peak": max((float(row["ollama_ram_mb_peak"]) for row in group), default=0),
            "system_cpu_percent_peak": max((float(row["system_cpu_percent_peak"]) for row in group), default=0),
            "gpu_percent_peak": max((float(row["gpu_percent_peak"]) for row in group), default=0),
            "vram_mb_peak": max((float(row["vram_mb_peak"]) for row in group), default=0),
        }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh tài nguyên Qwen thuần và Hybrid trên cùng PDF.")
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "test_input",
        help="File PDF hoặc thư mục PDF (mặc định: test_input).",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "qwen_vs_hybrid")
    parser.add_argument("--runs", type=int, default=1, help="Số lượt cho mỗi mode/PDF.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs phải >= 1")
    rows = run(args.input.resolve(), args.output.resolve(), args.runs)
    write_reports(rows, args.output.resolve())
    print(f"Đã lưu benchmark tại: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
