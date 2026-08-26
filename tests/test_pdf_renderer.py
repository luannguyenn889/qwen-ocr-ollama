# Nhiệm vụ: Kiểm tra tính năng chuyển đổi PDF sang ảnh PNG của module pdf_renderer.

import sys
import shutil
from pathlib import Path

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import pymupdf
from app.core.pdf_renderer import render_pdf_to_images

def prepare_test_pdf():
    source_pdf = PROJECT_ROOT / "samples" / "pdfs" / "test.pdf"
    target_pdf = PROJECT_ROOT / "samples" / "pdfs" / "test_5_pages.pdf"

    if target_pdf.is_file():
        return target_pdf

    print(f"[*] Không tìm thấy {target_pdf.name}, đang tạo từ 5 trang đầu của {source_pdf.name}...")
    if not source_pdf.is_file():
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {source_pdf}")

    src_doc = pymupdf.open(source_pdf)
    dest_doc = pymupdf.open()

    # Copy 5 trang đầu
    pages_to_copy = min(5, len(src_doc))
    dest_doc.insert_pdf(src_doc, from_page=0, to_page=pages_to_copy - 1)

    target_pdf.parent.mkdir(parents=True, exist_ok=True)
    dest_doc.save(target_pdf)
    dest_doc.close()
    src_doc.close()
    print(f"[+] Đã tạo file: {target_pdf}")
    return target_pdf

def test_renderer():
    pdf_path = prepare_test_pdf()
    output_dir = PROJECT_ROOT / "output" / "test_pdf_renderer_output"

    # Xoá thư mục output cũ nếu có để đảm bảo test sạch
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"\n[*] Bắt đầu render {pdf_path.name} thành ảnh với DPI=200...")
    images = render_pdf_to_images(pdf_path, output_dir, dpi=200)

    print("\n=== KẾT QUẢ KIỂM TRA ===")

    # 1. Kiểm tra số lượng ảnh
    num_images = len(images)
    is_5_pages = num_images == 5
    status_5_pages = "[x] đúng 5 ảnh" if is_5_pages else f"[ ] đúng 5 ảnh (Hiện tại có: {num_images} ảnh)"
    print(status_5_pages)

    # 2. Kiểm tra đúng thứ tự đặt tên
    expected_names = [f"page_{i:05d}.png" for i in range(1, 6)]
    actual_names = [img.name for img in sorted(images)]
    is_correct_order = actual_names == expected_names
    status_order = "[x] đúng thứ tự" if is_correct_order else f"[ ] đúng thứ tự (Thực tế: {actual_names})"
    print(status_order)

    # 3. Kiểm tra không lỗi màu và chữ đọc rõ (kiểm tra thuộc tính ảnh thông qua pymupdf/PIL)
    # pyrefly: ignore [missing-import]
    import PIL.Image
    color_ok = True
    readable_ok = True

    for img_path in images:
        if not img_path.is_file():
            color_ok = False
            readable_ok = False
            continue

        try:
            with PIL.Image.open(img_path) as img:
                # Kiểm tra hệ màu RGB
                if img.mode != "RGB":
                    color_ok = False
                # Kiểm tra độ phân giải có đủ lớn cho DPI=200 (thường chiều rộng/cao của A4 > 1000px)
                width, height = img.size
                if width < 1000 or height < 1000:
                    readable_ok = False
        except Exception as e:
            print(f"[!] Lỗi khi đọc ảnh {img_path.name}: {e}")
            color_ok = False
            readable_ok = False

    status_color = "[x] không lỗi màu" if color_ok else "[ ] không lỗi màu (có ảnh không ở hệ màu RGB hoặc lỗi định dạng)"
    status_readable = "[x] chữ đọc rõ" if readable_ok else "[ ] chữ đọc rõ (độ phân giải quá thấp hoặc lỗi không thể đọc được file)"

    print(status_color)
    print(status_readable)

    print(f"\nCác file ảnh đã được lưu tại: {output_dir.as_posix()}")

if __name__ == "__main__":
    test_renderer()
