import sys
from pathlib import Path
import fitz  # PyMuPDF

def render_pdf_to_simple_names(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_images = []
    
    doc = fitz.open(pdf_path)
    try:
        for index in range(len(doc)):
            page = doc.load_page(index)
            # Render page to a pixmap
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            
            # Simple name: page_1.png, page_2.png, ...
            image_name = f"page_{index + 1}.png"
            image_path = output_dir / image_name
            
            pix.save(str(image_path))
            saved_images.append(image_path)
            print(f"Rendered: {image_path.name}")
    finally:
        doc.close()
        
    return saved_images

def main():
    pdf_path = Path("test.pdf").resolve()
    output_dir = Path(".").resolve()  # output to current dir or a subfolder?
    # The prompt says "test.pdf -> page_1.png page_2.png page_3.png" in the same level or output.
    # Let's save in the current root folder to match "test.pdf -> page_1.png".
    
    if not pdf_path.is_file():
        print(f"Error: test.pdf not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Rendering {pdf_path} using PyMuPDF...")
    images = render_pdf_to_simple_names(pdf_path, output_dir, dpi=300)
    print(f"Successfully rendered {len(images)} pages.")

if __name__ == "__main__":
    main()
