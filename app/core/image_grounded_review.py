"""Conservative, image-grounded review of suspicious OCR lines."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from PIL import Image


@dataclass(frozen=True)
class SuspiciousLine:
    index: int
    text: str
    reason: str


def find_suspicious_lines(
    markdown: str, *, review_document_footer: bool = False,
) -> list[SuspiciousLine]:
    """Find high-value review candidates without changing their text."""
    from app.core.vietnamese_spell_corrector import suggest_vietnamese_spelling

    lines = markdown.splitlines()
    reasons: dict[int, set[str]] = {}

    def flag(index: int, reason: str) -> None:
        if 0 <= index < len(lines) and lines[index].strip():
            reasons.setdefault(index, set()).add(reason)

    for index, line in enumerate(lines):
        plain = line.strip()
        date_like = bool(re.search(r"\bngày\s*.{0,12}\d", plain, re.I))
        valid_date = bool(
            re.search(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", plain, re.I)
            or re.search(r"ngày\s+\d{1,2}/\d{1,2}/\d{4}", plain, re.I)
        )
        if date_like and (re.search(r"\d\s*[.]{2,}|\d[.]\d[.]", plain) or not valid_date):
            flag(index, "malformed_date")
        if re.search(r"\bSố\s*:\s*[^\s]*[.]{2,}|\bSố\s*:\s*\d+[.]\/", plain, re.I):
            flag(index, "malformed_document_number")
        hyphenated_words = re.findall(r"(?<!\d)([A-Za-zÀ-ỹĐđ]+)-([A-Za-zÀ-ỹĐđ]+)", plain)
        if any(not left.isupper() and not right.isupper() for left, right in hyphenated_words):
            flag(index, "possible_pen_stroke_or_glued_punctuation")
        if re.search(r"[^\W\d_]{45,}", plain, re.UNICODE):
            flag(index, "glued_characters")
    for warning in suggest_vietnamese_spelling(markdown):
        flag(int(warning["line"]) - 1, "unusual_word")

    if review_document_footer:
        # Every document type may contain printed text obscured by graphics at
        # the end.  Review the final page structurally, without requiring an
        # administrative profile, recipient heading, or known signing title.
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip() and not lines[index].lstrip().startswith("<!--"):
                flag(index, "possible_missing_printed_footer")
                break

    return [
        SuspiciousLine(index, lines[index], ",".join(sorted(values)))
        for index, values in sorted(reasons.items())
    ]


def _crop_for_line(
    page_image: Path, markdown: str, candidate: SuspiciousLine, output_dir: Path,
    regions: list[tuple[float, float, float, float]] | None = None,
) -> Path:
    """Create a temporary contextual crop using reading-order line position."""
    lines = markdown.splitlines()
    content_indices = [i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("<!--")]
    rank = content_indices.index(candidate.index) if candidate.index in content_indices else 0
    fraction = rank / max(len(content_indices) - 1, 1)
    with Image.open(page_image) as source:
        image = source.convert("RGB")
        if "possible_missing_signer" in candidate.reason or "missing_printed_footer" in candidate.reason:
            top, bottom = int(image.height * 0.62), image.height
        elif regions:
            ordered = sorted(regions, key=lambda box: (box[1], box[0]))
            region_index = round(fraction * max(len(ordered) - 1, 0))
            first = max(0, region_index - 1)
            last = min(len(ordered) - 1, region_index + 1)
            top = max(0, int(ordered[first][1]) - 30)
            bottom = min(image.height, int(ordered[last][3]) + 30)
        else:
            center = int(image.height * fraction)
            context = max(int(image.height * 0.12), 180)
            top, bottom = max(0, center - context), min(image.height, center + context)
        crop = image.crop((0, top, image.width, bottom))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"review_line_{candidate.index + 1}.png"
    crop.save(path)
    return path


def _response_text(response) -> str:
    value = getattr(response, "response", None)
    if value is not None:
        return str(value)
    if isinstance(response, dict):
        return str(response.get("response", ""))
    return str(response)


def review_suspicious_lines(
    client, model: str, page_image: Path, markdown: str, temp_dir: Path,
    *, log_func=print, before_request=None, max_reviews: int = 4,
    regions: list[tuple[float, float, float, float]] | None = None,
    review_document_footer: bool = False,
) -> str:
    """Reread suspicious lines from temporary crops and apply only near-certain edits."""
    from app.core.batch_ocr import generate_with_retry

    candidates = find_suspicious_lines(
        markdown, review_document_footer=review_document_footer,
    )[:max_reviews]
    if not candidates:
        return markdown
    lines = markdown.splitlines()
    for candidate in candidates:
        if before_request is not None:
            before_request()
        crop = _crop_for_line(page_image, markdown, candidate, temp_dir, regions)
        if "possible_missing_printed_footer" in candidate.reason:
            prompt = f"""Đọc riêng vùng ký/xác nhận ở cuối ảnh.
Chép mọi dòng CHỮ IN nhìn rõ theo đúng thứ tự, bất kể đó là cơ quan, vai trò, chức danh,
họ tên, nhãn xác nhận hay loại văn bản nào. Không giới hạn vào một danh sách chức danh có sẵn.
Không đọc nét ký tay thành chữ. Chữ trong con dấu chỉ giữ khi đọc chắc chắn và có ý nghĩa.
Chữ in có thể mang màu đỏ, xanh hoặc bị con dấu/nét ký chồng lên; vẫn giữ nếu hình dạng chữ rõ.
Đặc biệt kiểm tra chữ in nằm BÊN DƯỚI nét ký/con dấu; đó thường là họ tên và không được bỏ sót.
Không lặp các dòng đã có trong OCR. Dòng neo cuối hiện tại là:
{candidate.text}

Chỉ trả JSON hợp lệ:
{{"original": {json.dumps(candidate.text, ensure_ascii=False)}, "printed_lines": ["..."], "confidence": 0.0}}
Nếu không có dòng in mới đọc chắc chắn, trả printed_lines rỗng."""
        else:
            signing_instruction = ""
            response_schema = f'''Chỉ trả JSON hợp lệ:
{{"original": {json.dumps(candidate.text, ensure_ascii=False)}, "corrected": "...", "confidence": 0.0}}
Nếu không chắc chắn hoặc ảnh đúng như OCR, đặt corrected giống original.'''
            prompt = f"""Đối chiếu dòng OCR mục tiêu với ảnh crop (có thêm ngữ cảnh trước/sau).
Chỉ sửa khi ký tự trên ảnh chứng minh rõ OCR sai. Giữ nguyên lỗi chính tả vốn được in trên tài liệu.
Không sửa văn phong, không viết lại câu, không suy đoán phần mờ, không đọc nét ký tay thành chữ.
Nếu phát hiện họ tên được IN rõ bên dưới chức danh thì có thể bổ sung; không chép nét chữ ký.
{signing_instruction}

Dòng OCR mục tiêu:
{candidate.text}

Lý do kiểm tra: {candidate.reason}

{response_schema}"""
        response = generate_with_retry(
            client,
            dict(
                model=model, prompt=prompt, images=[str(crop)], think=False, stream=False,
                format="json",
                options={"temperature": 0, "num_ctx": 4096, "num_predict": 512}, keep_alive="10m",
            ),
            log_func=log_func,
        )
        raw = _response_text(response).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        try:
            data = json.loads(raw)
            original = str(data["original"])
            confidence = float(data["confidence"])
            if "possible_missing_printed_footer" in candidate.reason:
                values = data.get("printed_lines", [])
                if not isinstance(values, list):
                    raise TypeError("printed_lines must be a list")
                existing = markdown.casefold()
                printed_lines = [str(value).strip() for value in values if str(value).strip()]
                printed_lines = [value for value in printed_lines if value.casefold() not in existing]
                corrected = original + ("\n\n" + "\n".join(printed_lines) if printed_lines else "")
            else:
                corrected = str(data["corrected"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            log_func(f"Image review line {candidate.index + 1}: invalid response; kept original.")
            continue
        minimum_confidence = 0.95 if "possible_missing_printed_footer" in candidate.reason else 0.98
        if original != candidate.text or confidence < minimum_confidence or not corrected.strip() or corrected == original:
            continue
        similarity = SequenceMatcher(None, original, corrected).ratio()
        may_add_signer = (
            "possible_missing_signer" in candidate.reason
            or "possible_missing_printed_footer" in candidate.reason
        )
        if similarity < (0.20 if may_add_signer else 0.72):
            log_func(f"Image review line {candidate.index + 1}: change too broad; kept original.")
            continue
        lines[candidate.index] = corrected
        log_func(f"Image review line {candidate.index + 1}: applied image-confirmed correction.")
    return "\n".join(lines)
