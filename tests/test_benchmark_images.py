# Nhiệm vụ: Đánh giá hiệu năng (Benchmark) OCR trên bộ 10 ảnh mẫu: đo thời gian, dung lượng RAM sử dụng, VRAM GPU đỉnh (qua nvidia-smi) và tính toán độ chính xác (so với Ground Truth).

"""Benchmark OCR cho 10 ảnh mẫu.

Chạy trực tiếp:
    python tests\test_benchmark_images.py

Để tính độ chính xác, đặt Markdown đáp án chuẩn tại:
    samples\ground_truth\<tên_ảnh>.md
Ví dụ: samples\ground_truth\01_scan_ro.md
"""

import csv
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ollama_engine import OllamaQwenEngine


IMAGE_NAMES = (
    "01_scan_ro.png",
    "02_scan_mo.png",
    "03_nghieng.png",
    "04_bang.png",
    "05_hai_cot.png",
    "06_cong_thuc.png",
    "07_viet_anh.png",
    "08_photo_cu.png",
    "09_chu_nho.png",
    "10_layout_phuc_tap.png",
)
IMAGES_DIR = PROJECT_ROOT / "samples" / "images"
GROUND_TRUTH_DIR = PROJECT_ROOT / "samples" / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "output" / "image_benchmark"
MARKDOWN_DIR = OUTPUT_DIR / "markdown"


@dataclass
class VramMonitor:
    samples_mib: list[int]

    def __post_init__(self):
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        """Ghi VRAM mỗi giây bằng chính lệnh `nvidia-smi -l 1`."""
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-l",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError:
            print("[Cảnh báo] Không tìm thấy nvidia-smi; VRAM sẽ là N/A.", flush=True)
            return

        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def _collect(self):
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            if self._stop.is_set():
                return
            match = re.search(r"(\d+)\s*$", line)
            if match:
                self.samples_mib.append(int(match.group(1)))

    def stop(self):
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)


def ollama_ram_mib() -> float | None:
    """Lấy tổng Working Set của các tiến trình Ollama trên Windows."""
    command = (
        "(Get-Process -Name 'ollama*' -ErrorAction SilentlyContinue | "
        "Measure-Object -Property WorkingSet64 -Sum).Sum"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip()) / (1024 * 1024)
    except ValueError:
        return None


def normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def accuracy_percent(markdown: str, ground_truth_path: Path) -> float | None:
    """So sánh ký tự sau khi chuẩn hoá khoảng trắng với Markdown đáp án."""
    if not ground_truth_path.is_file():
        return None
    expected = normalized_text(ground_truth_path.read_text(encoding="utf-8"))
    actual = normalized_text(markdown)
    return 100 * SequenceMatcher(None, expected, actual).ratio()


def format_value(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.1f}{suffix}"


def main():
    missing = [name for name in IMAGE_NAMES if not (IMAGES_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Thiếu ảnh benchmark: " + ", ".join(missing))

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    engine = OllamaQwenEngine()
    print("=== Benchmark OCR ảnh (không xử lý PDF) ===", flush=True)
    print(f"Model: {engine.model}", flush=True)
    print("Đang kiểm tra kết nối Ollama...", flush=True)
    engine.check_connection()

    monitor = VramMonitor(samples_mib=[])
    monitor.start()
    rows: list[dict[str, str]] = []
    benchmark_started = perf_counter()

    try:
        for index, image_name in enumerate(IMAGE_NAMES, start=1):
            image_path = IMAGES_DIR / image_name
            markdown_path = MARKDOWN_DIR / f"{image_path.stem}.md"
            ground_truth_path = GROUND_TRUTH_DIR / f"{image_path.stem}.md"
            sample_start = len(monitor.samples_mib)
            ram_before = ollama_ram_mib()

            print(f"\n[{index}/10] Đang OCR: {image_name}", flush=True)
            started_at = perf_counter()
            try:
                markdown = engine.ocr_image(image_path)
                markdown_path.write_text(markdown + "\n", encoding="utf-8")
                status = "OK"
                error = ""
            except Exception as exc:
                markdown = ""
                status = "ERROR"
                error = str(exc)

            elapsed = perf_counter() - started_at
            ram_after = ollama_ram_mib()
            vram_samples = monitor.samples_mib[sample_start:]
            accuracy = accuracy_percent(markdown, ground_truth_path) if status == "OK" else None

            row = {
                "image": image_name,
                "status": status,
                "time_seconds": f"{elapsed:.1f}",
                "ollama_ram_before_mib": format_value(ram_before),
                "ollama_ram_after_mib": format_value(ram_after),
                "vram_peak_mib": str(max(vram_samples)) if vram_samples else "N/A",
                "accuracy_percent": format_value(accuracy),
                "markdown": str(markdown_path.relative_to(PROJECT_ROOT)) if status == "OK" else "",
                "error": error,
            }
            rows.append(row)
            print(
                f"    {status} | {elapsed:.1f}s | RAM: "
                f"{format_value(ram_after, ' MiB')} | VRAM đỉnh: {row['vram_peak_mib']} MiB | "
                f"Độ chính xác: {row['accuracy_percent']}",
                flush=True,
            )
    finally:
        monitor.stop()

    total_seconds = perf_counter() - benchmark_started
    write_reports(rows, total_seconds)
    print(f"\nHoàn tất trong {total_seconds:.1f} giây.", flush=True)
    print(f"Báo cáo: {OUTPUT_DIR / 'report.md'}", flush=True)


def write_reports(rows: list[dict[str, str]], total_seconds: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Benchmark OCR ảnh",
        "",
        f"- Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Model: qwen3.5:4b",
        f"- Tổng thời gian: {total_seconds:.1f} giây",
        "- VRAM: lấy mẫu mỗi giây bằng `nvidia-smi -l 1`.",
        "- Độ chính xác: độ tương đồng ký tự sau chuẩn hoá khoảng trắng so với đáp án Markdown.",
        "  `N/A` nghĩa là chưa có `samples/ground_truth/<tên_ảnh>.md`.",
        "",
        "| Ảnh | Trạng thái | Thời gian (s) | RAM Ollama sau (MiB) | VRAM đỉnh (MiB) | Độ chính xác (%) | Markdown |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            "| {image} | {status} | {time_seconds} | {ollama_ram_after_mib} | "
            "{vram_peak_mib} | {accuracy_percent} | `{markdown}` |".format(**row)
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
