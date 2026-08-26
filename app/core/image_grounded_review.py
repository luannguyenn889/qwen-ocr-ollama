"""Conservative, image-grounded review of suspicious OCR lines."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from time import perf_counter

# pyrefly: ignore [missing-import]
from PIL import Image


@dataclass(frozen=True)
class SuspiciousLine:
    index: int
    text: str
    reason: str
    severity: str = "medium"
    bbox: tuple[float, float, float, float] | None = None


SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
REASON_SEVERITY = {
    "mass_diacritic_loss": "critical",
    "invalid_table_grid": "critical",
    "ocr_disagreement": "high",
    "graphic_overlap": "high",
    "truncated_line": "high",
    "glued_characters": "high",
    "malformed_date": "high",
    "malformed_document_number": "high",
    "possible_pen_stroke_or_glued_punctuation": "medium",
    "possible_missing_printed_footer": "high",
    "unusual_word": "low",
}


def find_suspicious_lines(
    markdown: str, *, review_document_footer: bool = False,
    regions: list[tuple[float, float, float, float]] | None = None,
    graphic_regions: list[tuple[float, float, float, float]] | None = None,
    regional_text_by_index: dict[int, str] | None = None,
    block_spans: list | None = None,
) -> list[SuspiciousLine]:
    """Find high-value review candidates without changing their text."""
    from app.core.vietnamese_spell_corrector import suggest_vietnamese_spelling

    lines = markdown.splitlines()
    reasons: dict[int, set[str]] = {}
    content_indices = [
        index for index, line in enumerate(lines)
        if line.strip() and not line.lstrip().startswith("<!--")
    ]
    line_boxes: dict[int, tuple[float, float, float, float]] = {}
    if block_spans is not None:
        cursor = 0
        for index, line in enumerate(lines):
            start = markdown.find(line, cursor)
            end = start + len(line)
            cursor = end
            span = next((item for item in block_spans if item.markdown_start <= start < item.markdown_end), None)
            if span is not None and span.mapping_confidence == "exact" and span.kind != "image":
                line_boxes[index] = span.bbox
    else:
        ordered_regions = sorted(regions or [], key=lambda box: (box[1], box[0]))
    if block_spans is None and ordered_regions and content_indices:
        for rank, line_index in enumerate(content_indices):
            region_index = round(rank / max(len(content_indices) - 1, 1) * (len(ordered_regions) - 1))
            line_boxes[line_index] = ordered_regions[region_index]

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
        if re.search(r"(?:\ufffd|\w-$|\.{3,}$)", plain, re.UNICODE):
            flag(index, "truncated_line")

    from app.core.quality_gate import evaluate_page
    report = evaluate_page(markdown, ".", check_tables=True)
    if "missing_vietnamese_diacritics" in report.errors:
        for index in content_indices:
            words = re.findall(r"[A-Za-zÀ-ỹĐđ]+", lines[index])
            ascii_words = [word.casefold() for word in words if word.isascii()]
            signals = {"cua", "va", "la", "trong", "duoc", "nhung", "mot", "cho", "voi", "khong"}
            if len(words) >= 7 and sum(word in signals for word in ascii_words) >= 2:
                flag(index, "mass_diacritic_loss")
    if any(error.startswith("malformed_html_table:") for error in report.errors):
        for index, line in enumerate(lines):
            if re.search(r"<table\b", line, re.I):
                flag(index, "invalid_table_grid")
                break

    for index, reread in (regional_text_by_index or {}).items():
        if 0 <= index < len(lines) and reread.strip():
            similarity = SequenceMatcher(None, lines[index].strip(), reread.strip()).ratio()
            if similarity < 0.72:
                flag(index, "ocr_disagreement")

    def intersects(first, second) -> bool:
        return not (
            first[2] <= second[0] or second[2] <= first[0]
            or first[3] <= second[1] or second[3] <= first[1]
        )

    # Never escalate a line from an approximate reading-order box. On complex
    # pages that approximation can map one graphic onto many unrelated lines.
    if block_spans is not None:
        for index, bbox in line_boxes.items():
            if any(intersects(bbox, graphic) for graphic in graphic_regions or []):
                flag(index, "graphic_overlap")
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

    candidates = []
    for index, values in reasons.items():
        severity = min((REASON_SEVERITY.get(value, "medium") for value in values), key=SEVERITY_RANK.get)
        candidates.append(SuspiciousLine(
            index, lines[index], ",".join(sorted(values)), severity, line_boxes.get(index),
        ))
    return sorted(candidates, key=lambda item: (SEVERITY_RANK[item.severity], item.index))


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
            left, top, right, bottom = 0, int(image.height * 0.62), image.width, image.height
        elif candidate.bbox is not None and regions:
            ordered = sorted(regions, key=lambda box: (box[1], box[0]))
            region_index = min(
                range(len(ordered)),
                key=lambda index: sum(abs(ordered[index][axis] - candidate.bbox[axis]) for axis in range(4)),
            )
            first = max(0, region_index - 1)
            last = min(len(ordered) - 1, region_index + 1)
            left = max(0, int(min(box[0] for box in ordered[first:last + 1])) - 30)
            top = max(0, int(ordered[first][1]) - 30)
            right = min(image.width, int(max(box[2] for box in ordered[first:last + 1])) + 30)
            bottom = min(image.height, int(ordered[last][3]) + 30)
        else:
            center = int(image.height * fraction)
            context = max(int(image.height * 0.12), 180)
            left, right = 0, image.width
            top, bottom = max(0, center - context), min(image.height, center + context)
        crop = image.crop((left, top, right, bottom))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"review_line_{candidate.index + 1}.png"
    crop.save(path)
    return path


def _intersects(first, second) -> bool:
    return not (
        first[2] <= second[0] or second[2] <= first[0]
        or first[3] <= second[1] or second[3] <= first[1]
    )


def _footer_has_graphic_text_overlap(
    page_image: Path,
    text_regions: list[tuple[float, float, float, float]] | None,
    graphic_regions: list[tuple[float, float, float, float]] | None,
) -> bool:
    """Require actual lower-page graphic/text overlap before footer review."""
    if not text_regions or not graphic_regions:
        return False
    with Image.open(page_image) as source:
        footer_top = source.height * 0.55
    footer_graphics = [box for box in graphic_regions if box[3] >= footer_top]
    return any(_intersects(text, graphic) for text in text_regions for graphic in footer_graphics)


def _group_nearby_candidates(
    candidates: list[SuspiciousLine], page_image: Path,
) -> list[list[SuspiciousLine]]:
    """Merge two nearby suspicious lines so Qwen sees one contextual crop."""
    if len(candidates) < 2:
        return [[candidate] for candidate in candidates]
    first, second = candidates[:2]
    nearby = abs(first.index - second.index) <= 2
    if first.bbox is not None and second.bbox is not None:
        with Image.open(page_image) as source:
            max_gap = source.height * 0.08
        vertical_gap = max(0.0, second.bbox[1] - first.bbox[3], first.bbox[1] - second.bbox[3])
        horizontal_overlap = min(first.bbox[2], second.bbox[2]) > max(first.bbox[0], second.bbox[0])
        nearby = vertical_gap <= max_gap and horizontal_overlap
    return [[first, second]] if nearby else [[first], [second]]


def _crop_for_group(
    page_image: Path, markdown: str, group: list[SuspiciousLine], output_dir: Path,
    regions: list[tuple[float, float, float, float]] | None = None,
) -> Path:
    if len(group) == 1:
        return _crop_for_line(page_image, markdown, group[0], output_dir, regions)
    with Image.open(page_image) as source:
        image = source.convert("RGB")
        boxes = [candidate.bbox for candidate in group if candidate.bbox is not None]
        if len(boxes) == len(group):
            left = max(0, int(min(box[0] for box in boxes)) - 40)
            top = max(0, int(min(box[1] for box in boxes)) - 80)
            right = min(image.width, int(max(box[2] for box in boxes)) + 40)
            bottom = min(image.height, int(max(box[3] for box in boxes)) + 80)
        else:
            lines = markdown.splitlines()
            content = [i for i, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("<!--")]
            fractions = [
                content.index(candidate.index) / max(len(content) - 1, 1)
                if candidate.index in content else 0.0
                for candidate in group
            ]
            context = max(int(image.height * 0.10), 160)
            left, right = 0, image.width
            top = max(0, int(image.height * min(fractions)) - context)
            bottom = min(image.height, int(image.height * max(fractions)) + context)
        crop = image.crop((left, top, right, bottom))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(str(candidate.index + 1) for candidate in group)
    path = output_dir / f"review_lines_{suffix}.png"
    crop.save(path)
    return path


def _review_client(client, timeout_seconds: float = 45.0):
    """Clone a real local Ollama client with a short optional-review timeout."""
    transport = getattr(client, "_client", None)
    base_url = getattr(transport, "base_url", None)
    base_url_text = str(base_url or "")
    if not base_url_text.startswith(("http://", "https://")):
        return client, None
    # pyrefly: ignore [missing-import]
    from ollama import Client
    bounded = Client(host=base_url_text, timeout=timeout_seconds)
    return bounded, getattr(getattr(bounded, "_client", None), "close", None)


def _response_text(response) -> str:
    value = getattr(response, "response", None)
    if value is not None:
        return str(value)
    if isinstance(response, dict):
        return str(response.get("response", ""))
    return str(response)


def select_review_candidates(candidates: list[SuspiciousLine]) -> list[SuspiciousLine]:
    """Select only image-actionable candidates within one compact page batch."""
    # Spelling is a discovery signal, not enough evidence to spend another
    # vision request. It must never directly change the OCR result.
    candidates = [candidate for candidate in candidates if candidate.severity in {"critical", "high"}]
    budgets = {"critical": 2, "high": 2}
    selected: list[SuspiciousLine] = []
    for severity in ("critical", "high"):
        group = [candidate for candidate in candidates if candidate.severity == severity]
        structural = [candidate for candidate in group if "missing_printed_footer" in candidate.reason]
        others = [candidate for candidate in group if candidate not in structural]
        budget = budgets[severity]
        selected.extend(structural)
        selected.extend(others[:max(0, budget - len(structural))])
    # Too many images can overflow the vision context on a laptop. Severity
    # ordering is deterministic, so the highest-value regions win.
    return selected[:2]


def should_reread_full_page(candidates: list[SuspiciousLine], markdown: str) -> bool:
    """Escalate widespread independent text failures to a full-page reread."""
    text_candidates = [
        candidate for candidate in candidates if "invalid_table_grid" not in candidate.reason
        and "missing_printed_footer" not in candidate.reason
    ]
    affected = {candidate.index for candidate in text_candidates}
    content_line_count = sum(
        bool(line.strip() and not line.lstrip().startswith("<!--"))
        for line in markdown.splitlines()
    )
    critical = {candidate.index for candidate in text_candidates if candidate.severity == "critical"}
    high_or_worse = {
        candidate.index for candidate in text_candidates
        if SEVERITY_RANK[candidate.severity] <= SEVERITY_RANK["high"]
    }
    widespread_diacritics = sum("mass_diacritic_loss" in candidate.reason for candidate in text_candidates) >= 2
    return (
        len(critical) >= 3
        or len(high_or_worse) >= 3
        # A single bad line is always a regional reread, even on a short page.
        or (len(affected) >= 2 and len(affected) / max(content_line_count, 1) >= 0.20)
        or widespread_diacritics
    )


def review_suspicious_lines(
    client, model: str, page_image: Path, markdown: str, temp_dir: Path,
    *, log_func=print, before_request=None, max_reviews: int | None = None,
    regions: list[tuple[float, float, float, float]] | None = None,
    graphic_regions: list[tuple[float, float, float, float]] | None = None,
    regional_text_by_index: dict[int, str] | None = None,
    block_spans: list | None = None,
    review_document_footer: bool = False,
    quality_score: float | None = None,
) -> str:
    """Reread suspicious regions with at most one extra Qwen request per page."""
    from app.core.batch_ocr import ocr_qwen_images

    started_at = perf_counter()
    qwen_requests = 0
    applied = 0
    kept = 0
    skipped = 0

    footer_requested = review_document_footer
    review_document_footer = review_document_footer and _footer_has_graphic_text_overlap(
        page_image, regions, graphic_regions,
    )
    if footer_requested and not review_document_footer:
        log_func("[Hậu kiểm] Bỏ qua footer: không có đồ họa giao vùng chữ in ở cuối trang.")
    candidates = find_suspicious_lines(
        markdown, review_document_footer=review_document_footer, regions=regions,
        graphic_regions=graphic_regions, regional_text_by_index=regional_text_by_index,
        block_spans=block_spans,
    )
    if block_spans is not None:
        unmapped = [candidate for candidate in candidates if candidate.bbox is None]
        if unmapped:
            skipped += len(unmapped)
            log_func(
                f"[Hậu kiểm] Bỏ qua {len(unmapped)} dòng nghi ngờ không có ánh xạ OCR-block chính xác."
            )
        candidates = [candidate for candidate in candidates if candidate.bbox is not None]
    if quality_score is not None and quality_score >= 100.0:
        skipped_lines = [
            candidate for candidate in candidates
            if "possible_missing_printed_footer" not in candidate.reason
        ]
        skipped += len(skipped_lines)
        candidates = [
            candidate for candidate in candidates
            if "possible_missing_printed_footer" in candidate.reason
        ]
        if skipped_lines:
            log_func(
                f"[Hậu kiểm] Quality Gate đạt 100; bỏ qua {len(skipped_lines)} ứng viên dòng."
            )
    if not candidates:
        log_func(
            f"[Hậu kiểm] Không có ứng viên cần đọc lại; hoàn tất trong "
            f"{perf_counter() - started_at:.2f}s."
        )
        return markdown
    reason_names = sorted({reason for item in candidates for reason in item.reason.split(",")})
    log_func(
        f"[Hậu kiểm] Phát hiện {len(candidates)} ứng viên: {', '.join(reason_names)}."
    )
    table_candidates = [candidate for candidate in candidates if "invalid_table_grid" in candidate.reason]
    if table_candidates:
        log_func(f"[Hậu kiểm bảng] Chuyển {len(table_candidates)} lỗi lưới sang bộ kiểm tra bảng.")
    non_actionable_count = sum(candidate.severity in {"medium", "low"} for candidate in candidates)
    if non_actionable_count:
        skipped += non_actionable_count
        log_func(
            f"[Hậu kiểm] Bỏ qua {non_actionable_count} cảnh báo medium/low; không gọi Qwen."
        )
    candidates = [candidate for candidate in candidates if candidate.severity in {"critical", "high"}]
    if not candidates:
        log_func(
            f"[Hậu kiểm] Hoàn tất: không có vùng nghiêm trọng cần đọc lại, "
            f"bỏ qua={skipped}, Qwen thêm=0, thời gian={perf_counter() - started_at:.2f}s."
        )
        return markdown
    if should_reread_full_page(candidates, markdown):
        reasons = sorted({reason for candidate in candidates for reason in candidate.reason.split(",")})
        log_func(f"[Hậu kiểm] Lỗi diện rộng ({', '.join(reasons)}); đọc lại toàn trang.")
        request_client, close_client = _review_client(client)
        try:
            qwen_requests += 1
            reread = ocr_qwen_images(
                request_client, model, [page_image],
                extra_instruction=(
                    "\n\nĐọc lại toàn trang vì phát hiện lỗi diện rộng. Chép đúng nội dung nhìn thấy; "
                    "không sửa chính tả bản gốc, không viết lại câu và không suy đoán phần mờ."
                ),
                log_func=log_func, before_request=before_request, max_retries=1,
            )
            from app.core.quality_gate import choose_best_page, evaluate_page
            initial_report = evaluate_page(markdown, ".")
            reread_report = evaluate_page(reread, ".")
            selected, _report = choose_best_page(markdown, initial_report, reread, reread_report)
            if selected != markdown:
                markdown = selected
                applied += 1
                log_func("[Hậu kiểm] Đã áp dụng bản đọc lại toàn trang có chất lượng cao hơn.")
            else:
                kept += 1
                log_func("[Hậu kiểm] Bản đọc lại không an toàn hơn; giữ OCR ban đầu.")
        except Exception as error:
            kept += 1
            log_func(f"[Hậu kiểm] Đọc lại toàn trang thất bại; giữ OCR ban đầu: {error}")
        finally:
            if callable(close_client):
                close_client()
        # The full-page reread has consumed this page's single review request.
        # Regional rereads after it were the main source of apparent GUI stalls.
        log_func(
            f"[Hậu kiểm] Hoàn tất sau đọc lại toàn trang: đã sửa={applied}, "
            f"giữ nguyên={kept}, bỏ qua={skipped}, Qwen thêm={qwen_requests}, "
            f"thời gian={perf_counter() - started_at:.2f}s."
        )
        return markdown
    candidates = select_review_candidates(candidates)
    if max_reviews is not None:
        candidates = candidates[:max_reviews]
    candidates = [
        candidate for candidate in candidates if "invalid_table_grid" not in candidate.reason
    ]
    if not candidates:
        log_func(
            f"[Hậu kiểm] Hoàn tất: không có vùng nghiêm trọng cần đọc lại, "
            f"bỏ qua={skipped}, Qwen thêm=0, thời gian={perf_counter() - started_at:.2f}s."
        )
        return markdown

    groups = _group_nearby_candidates(candidates, page_image)
    crops = [_crop_for_group(page_image, markdown, group, temp_dir, regions) for group in groups]
    image_index_by_line = {
        candidate.index: image_index
        for image_index, group in enumerate(groups, 1)
        for candidate in group
    }
    request_items = [
        {
            "id": f"line_{candidate.index + 1}",
            "original": candidate.text,
            "reason": candidate.reason,
            "footer": "possible_missing_printed_footer" in candidate.reason,
            "image_index": image_index_by_line[candidate.index],
        }
        for candidate in candidates
    ]
    log_func(
        f"[Hậu kiểm] Đọc lại {len(candidates)} ứng viên bằng {len(crops)} crop "
        f"trong một yêu cầu Qwen "
        f"(dòng {', '.join(str(item.index + 1) for item in candidates)})."
    )
    prompt = f"""Đối chiếu {len(crops)} ảnh crop với các mục OCR dưới đây theo image_index.
Chỉ sửa khi ký tự trên ảnh chứng minh rõ OCR sai. Chép đúng nội dung nhìn thấy; giữ nguyên lỗi
chính tả vốn được in trên tài liệu. Không sửa văn phong, không viết lại câu, không suy đoán phần mờ.
Không đọc nét ký tay thành chữ. Chữ trong con dấu chỉ giữ khi đọc chắc chắn và có ý nghĩa.
Với mục footer=true, chép vào printed_lines mọi dòng CHỮ IN mới nhìn rõ ở vùng ký/xác nhận,
kể cả cơ quan, vai trò, chức danh và họ tên; không lặp dòng đã có. Với mục khác, dùng corrected.

Các mục theo đúng thứ tự ảnh:
{json.dumps(request_items, ensure_ascii=False)}

Chỉ trả JSON hợp lệ dạng:
{{"items":[{{"id":"line_1","original":"...","corrected":"...","printed_lines":[],"confidence":0.0}}]}}
Phải trả đúng một kết quả cho mỗi id. Nếu không chắc chắn, corrected phải giống original và
printed_lines phải rỗng."""
    if before_request is not None:
        before_request()
    qwen_requests += 1
    request_client, close_client = _review_client(client)
    try:
        response = request_client.generate(
            model=model, prompt=prompt, images=[str(crop) for crop in crops],
            think=False, stream=False, format="json",
            options={
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": min(1024, 256 + 256 * len(candidates)),
            },
            keep_alive="30m",
        )
        raw = _response_text(response).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        payload = json.loads(raw)
        response_items = payload.get("items") if isinstance(payload, dict) else None
        if response_items is None and isinstance(payload, dict) and len(candidates) == 1:
            response_items = [payload]
        if not isinstance(response_items, list):
            raise TypeError("items must be a list")
        by_id = {
            str(item.get("id", request_items[index]["id"])): item
            for index, item in enumerate(response_items)
            if isinstance(item, dict) and index < len(request_items)
        }
    except Exception as error:
        kept += len(candidates)
        log_func(f"[Hậu kiểm] Phản hồi batch không hợp lệ; giữ OCR ban đầu: {error}")
        log_func(
            f"[Hậu kiểm] Hoàn tất: ứng viên={len(candidates)}, đã sửa=0, "
            f"giữ nguyên={kept}, bỏ qua={skipped}, Qwen thêm={qwen_requests}, "
            f"thời gian={perf_counter() - started_at:.2f}s."
        )
        return markdown
    finally:
        if callable(close_client):
            close_client()

    lines = markdown.splitlines()
    existing = markdown.casefold()
    for candidate in candidates:
        data = by_id.get(f"line_{candidate.index + 1}")
        try:
            if not isinstance(data, dict):
                raise TypeError("missing result")
            original = str(data["original"])
            confidence = float(data["confidence"])
            if "possible_missing_printed_footer" in candidate.reason:
                values = data.get("printed_lines", [])
                if not isinstance(values, list):
                    raise TypeError("printed_lines must be a list")
                printed_lines = [str(value).strip() for value in values if str(value).strip()]
                printed_lines = [value for value in printed_lines if value.casefold() not in existing]
                corrected = original + ("\n\n" + "\n".join(printed_lines) if printed_lines else "")
            else:
                corrected = str(data["corrected"])
        except (KeyError, TypeError, ValueError):
            kept += 1
            log_func(f"[Hậu kiểm] Dòng {candidate.index + 1}: phản hồi không hợp lệ; giữ nguyên.")
            continue
        minimum_confidence = 0.95 if "possible_missing_printed_footer" in candidate.reason else 0.98
        if original != candidate.text or confidence < minimum_confidence or not corrected.strip() or corrected == original:
            kept += 1
            log_func(
                f"[Hậu kiểm] Dòng {candidate.index + 1}: giữ nguyên "
                f"(độ tin cậy {confidence:.2f}, yêu cầu {minimum_confidence:.2f})."
            )
            continue
        similarity = SequenceMatcher(None, original, corrected).ratio()
        may_add_signer = (
            "possible_missing_signer" in candidate.reason
            or "possible_missing_printed_footer" in candidate.reason
        )
        if similarity < (0.20 if may_add_signer else 0.72):
            kept += 1
            log_func(f"[Hậu kiểm] Dòng {candidate.index + 1}: thay đổi quá rộng; giữ nguyên.")
            continue
        lines[candidate.index] = corrected
        applied += 1
        log_func(
            f"[Hậu kiểm] Dòng {candidate.index + 1}: đã áp dụng "
            f"(độ tin cậy {confidence:.2f})."
        )
    log_func(
        f"[Hậu kiểm] Hoàn tất: ứng viên={len(candidates)}, đã sửa={applied}, "
        f"giữ nguyên={kept}, bỏ qua={skipped}, Qwen thêm={qwen_requests}, "
        f"thời gian={perf_counter() - started_at:.2f}s."
    )
    return "\n".join(lines)
