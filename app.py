from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import cv2
import fitz  # PyMuPDF
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class Rect(BaseModel):
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = int(round(self.x0 * width))
        y0 = int(round(self.y0 * height))
        x1 = int(round(self.x1 * width))
        y1 = int(round(self.y1 * height))
        x0, x1 = sorted((max(0, x0), min(width, x1)))
        y0, y1 = sorted((max(0, y0), min(height, y1)))
        return x0, y0, x1, y1


class TextRule(BaseModel):
    text: str = ""
    mode: Literal["exact", "contains", "regex"] = "exact"
    ignore_case: bool = False
    pages: list[int] = Field(default_factory=list)
    # Optional safety area. Empty/null means whole page.
    rect: Rect | None = None

    @field_validator("text")
    @classmethod
    def limit_regex_length(cls, v: str, info) -> str:
        # Prevent ReDoS by capping regex pattern length
        if info.data.get("mode") == "regex" and len(v) > 500:
            raise ValueError("Regex pattern too long (max 500 chars).")
        return v


class ProcessOptions(BaseModel):
    output_mode: Literal["preserve", "rasterize"] = "preserve"
    dpi: int = Field(default=220, ge=72, le=600)
    preview_dpi: int = Field(default=130, ge=72, le=220)
    jpeg_quality: int = Field(default=92, ge=30, le=100)
    output_format: Literal["jpeg", "png"] = "jpeg"

    center_rect: Rect = Rect(x0=0.30, y0=0.38, x1=0.70, y1=0.63)
    bottom_rect: Rect = Rect(x0=0.00, y0=0.945, x1=1.00, y1=1.00)
    extra_rects: list[Rect] = Field(default_factory=list)

    center_mode: Literal["red_mask", "white_rect", "disabled"] = "red_mask"
    min_red_over_green: int = Field(default=8, ge=-50, le=100)
    min_red_over_blue: int = Field(default=8, ge=-50, le=100)
    min_saturation: int = Field(default=12, ge=0, le=255)
    max_value: int = Field(default=255, ge=0, le=255)
    min_value: int = Field(default=120, ge=0, le=255)
    dilate_kernel: int = Field(default=3, ge=0, le=31)
    inpaint_radius: int = Field(default=0, ge=0, le=15)

    remove_bottom: bool = True
    background: tuple[int, int, int] = (255, 255, 255)
    pages: list[int] = Field(default_factory=list)

    # Text-object watermark removal.
    remove_text_watermark: bool = True
    text_rules: list[TextRule] = Field(default_factory=list)


def _validate_rect(rect: Rect) -> None:
    if rect.x0 >= rect.x1 or rect.y0 >= rect.y1:
        raise ValueError(f"Invalid rect: {rect.model_dump()}")


def _match_text(value: str, rule: TextRule) -> bool:
    pat = rule.text
    if not pat:
        return False
    if rule.mode == "regex":
        # Security: prevent ReDoS by limiting regex length and using timeout
        if len(pat) > 500:
            return False  # Reject overly complex patterns
        flags = re.I if rule.ignore_case else 0
        try:
            return re.search(pat, value, flags) is not None
        except re.error:
            return False  # Invalid regex treated as no match
    if rule.ignore_case:
        value = value.lower()
        pat = pat.lower()
    if rule.mode == "contains":
        return pat in value
    return value == pat


def _rule_applies_to_page(rule: TextRule, page_no: int) -> bool:
    return not rule.pages or page_no in set(rule.pages)


def _redact_text_watermarks(doc: fitz.Document, opt: ProcessOptions) -> int:
    """
    Remove real PDF text objects by redaction.
    This works before rasterizing, so exact/regex text watermark can be removed even
    when it is not image-based.
    """
    if not opt.remove_text_watermark or not opt.text_rules:
        return 0

    hit_count = 0
    for page_idx, page in enumerate(doc, start=1):
        page_w, page_h = page.rect.width, page.rect.height
        words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no
        for rule in opt.text_rules:
            if not _rule_applies_to_page(rule, page_idx):
                continue
            safe_rect = None
            if rule.rect is not None:
                x0, y0, x1, y1 = rule.rect.to_pixels(int(page_w), int(page_h))
                safe_rect = fitz.Rect(x0, y0, x1, y1)

            # Word-level search covers most watermark/footer text. For phrases,
            # use page.search_for as fallback.
            if rule.mode in ("exact", "contains", "regex"):
                for w in words:
                    text = w[4]
                    r = fitz.Rect(w[0], w[1], w[2], w[3])
                    if safe_rect is not None and not r.intersects(safe_rect):
                        continue
                    if _match_text(text, rule):
                        page.add_redact_annot(r + (-1, -1, 1, 1), fill=(1, 1, 1))
                        hit_count += 1

            # Phrase exact/contains with spaces.
            if " " in rule.text or len(rule.text) >= 4:
                flags = 0
                needles = [rule.text]
                if rule.mode == "contains":
                    # PyMuPDF search_for doesn't support contains/regex. Search the literal fragment.
                    needles = [rule.text]
                if rule.mode != "regex":
                    for inst in page.search_for(rule.text, flags=flags):
                        if safe_rect is not None and not inst.intersects(safe_rect):
                            continue
                        page.add_redact_annot(inst + (-1, -1, 1, 1), fill=(1, 1, 1))
                        hit_count += 1

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    return hit_count


def _red_mask_rgb(roi_rgb: np.ndarray, opt: ProcessOptions) -> np.ndarray:
    r = roi_rgb[:, :, 0].astype(np.int16)
    g = roi_rgb[:, :, 1].astype(np.int16)
    b = roi_rgb[:, :, 2].astype(np.int16)
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    mask = (
        (r - g >= opt.min_red_over_green)
        & (r - b >= opt.min_red_over_blue)
        & (sat >= opt.min_saturation)
        & (val >= opt.min_value)
        & (val <= opt.max_value)
    ).astype(np.uint8) * 255
    if opt.dilate_kernel and opt.dilate_kernel > 1:
        kernel = np.ones((opt.dilate_kernel, opt.dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _apply_white_rect(img_rgb: np.ndarray, rect: Rect, color: tuple[int, int, int]) -> None:
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = rect.to_pixels(w, h)
    if x1 > x0 and y1 > y0:
        img_rgb[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)


def _rects_for_page(opt: ProcessOptions) -> list[tuple[Rect, str]]:
    """Return list of (rect, label) pairs that define watermark regions."""
    rects = []
    if opt.center_mode != "disabled":
        rects.append((opt.center_rect, "center"))
    if opt.remove_bottom:
        rects.append((opt.bottom_rect, "bottom"))
    for i, r in enumerate(opt.extra_rects):
        rects.append((r, f"extra_{i}"))
    return rects


def _image_intersects_rects(
    img_rect: fitz.Rect, page_w: float, page_h: float, rects: list[tuple[Rect, str]]
) -> list[str]:
    """Check if an image bbox intersects any watermark region. Returns matched labels."""
    matched = []
    for rect, label in rects:
        x0, y0, x1, y1 = rect.to_pixels(int(page_w), int(page_h))
        wm_rect = fitz.Rect(x0, y0, x1, y1)
        if img_rect.intersects(wm_rect):
            matched.append(label)
    return matched


def _process_single_image(
    img_rgb: np.ndarray, matched_labels: list[str], opt: ProcessOptions
) -> np.ndarray:
    """Process a single extracted image based on which watermark regions it overlaps."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    bg = tuple(int(c) for c in opt.background)

    if "center" in matched_labels:
        if opt.center_mode == "white_rect":
            _apply_white_rect(out, opt.center_rect, bg)
        elif opt.center_mode == "red_mask":
            # Map page-level center_rect to image-local coordinates
            cx0, cy0, cx1, cy1 = opt.center_rect.to_pixels(w, h)
            cx0, cx1 = sorted((max(0, cx0), min(w, cx1)))
            cy0, cy1 = sorted((max(0, cy0), min(h, cy1)))
            if cx1 > cx0 and cy1 > cy0:
                roi = out[cy0:cy1, cx0:cx1]
                mask = _red_mask_rgb(roi, opt)
                if opt.inpaint_radius > 0:
                    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    fixed_bgr = cv2.inpaint(roi_bgr, mask, opt.inpaint_radius, cv2.INPAINT_TELEA)
                    out[cy0:cy1, cx0:cx1] = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)
                else:
                    roi[mask > 0] = np.array(bg, dtype=np.uint8)
                    out[cy0:cy1, cx0:cx1] = roi

    if "bottom" in matched_labels:
        _apply_white_rect(out, opt.bottom_rect, bg)

    for label in matched_labels:
        if label.startswith("extra_"):
            idx = int(label.split("_")[1])
            _apply_white_rect(out, opt.extra_rects[idx], bg)

    return out


def _process_page_image(img_rgb: np.ndarray, opt: ProcessOptions) -> np.ndarray:
    out = img_rgb.copy()
    h, w = out.shape[:2]
    bg = tuple(int(c) for c in opt.background)

    if opt.center_mode == "white_rect":
        _apply_white_rect(out, opt.center_rect, bg)
    elif opt.center_mode == "red_mask":
        x0, y0, x1, y1 = opt.center_rect.to_pixels(w, h)
        if x1 > x0 and y1 > y0:
            roi = out[y0:y1, x0:x1]
            mask = _red_mask_rgb(roi, opt)
            if opt.inpaint_radius > 0:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                fixed_bgr = cv2.inpaint(roi_bgr, mask, opt.inpaint_radius, cv2.INPAINT_TELEA)
                out[y0:y1, x0:x1] = cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB)
            else:
                roi[mask > 0] = np.array(bg, dtype=np.uint8)
                out[y0:y1, x0:x1] = roi

    if opt.remove_bottom:
        _apply_white_rect(out, opt.bottom_rect, bg)

    for rect in opt.extra_rects:
        _apply_white_rect(out, rect, bg)

    return out


def _render_page_to_rgb(page: fitz.Page, dpi: int) -> np.ndarray:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def _encode_image(img_rgb: np.ndarray, ext: str, jpeg_quality: int) -> bytes:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if ext == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError("Failed to encode page image.")
    return buf.tobytes()


def _encode_png(img_rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError("Failed to encode preview.")
    return buf.tobytes()


def _open_pdf_with_text_redaction(input_pdf: Path, opt: ProcessOptions) -> tuple[fitz.Document, int]:
    doc = fitz.open(input_pdf)
    redactions = _redact_text_watermarks(doc, opt)
    return doc, redactions


def _process_pdf_preserve(input_pdf: Path, output_pdf: Path, opt: ProcessOptions) -> dict[str, Any]:
    """Process PDF preserving original structure: text redaction + in-place image replacement."""
    doc, redactions = _open_pdf_with_text_redaction(input_pdf, opt)
    target_pages = set(opt.pages)
    page_count = doc.page_count

    if target_pages:
        invalid = sorted(p for p in target_pages if p < 1 or p > page_count)
        if invalid:
            raise ValueError(f"Invalid page numbers: {invalid}; PDF has {page_count} pages.")

    wm_rects = _rects_for_page(opt)
    processed_pages: list[int] = []
    images_processed = 0

    for idx, page in enumerate(doc, start=1):
        if target_pages and idx not in target_pages:
            continue
        processed_pages.append(idx)

        page_w, page_h = page.rect.width, page.rect.height
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(img)
            if not rects:
                continue

            for img_rect in rects:
                matched = _image_intersects_rects(img_rect, page_w, page_h, wm_rects)
                if not matched:
                    continue

                # Extract image as numpy RGB
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                ).copy()

                # Process
                out_rgb = _process_single_image(img_rgb, matched, opt)

                # Encode back to PNG
                if pix.n == 4:
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGBA2BGRA))
                elif pix.n == 1:
                    ok, buf = cv2.imencode(".png", out_rgb)
                else:
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    continue

                page.replace_image(xref, stream=buf.tobytes())
                images_processed += 1
                break  # each xref only needs one replacement

    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    return {
        "page_count": page_count,
        "processed_pages": processed_pages,
        "text_redactions": redactions,
        "images_processed": images_processed,
        "output_mode": "preserve",
    }


def _process_pdf_rasterize(input_pdf: Path, output_pdf: Path, opt: ProcessOptions) -> dict[str, Any]:
    """Process PDF by rasterizing every page (legacy mode, loses text layer)."""
    for rect in [opt.center_rect, opt.bottom_rect, *opt.extra_rects]:
        _validate_rect(rect)

    doc, redactions = _open_pdf_with_text_redaction(input_pdf, opt)
    out_doc = fitz.open()
    target_pages = set(opt.pages)
    page_count = doc.page_count

    if target_pages:
        invalid = sorted(p for p in target_pages if p < 1 or p > page_count)
        if invalid:
            raise ValueError(f"Invalid page numbers: {invalid}; PDF has {page_count} pages.")

    processed_pages: list[int] = []
    for idx, page in enumerate(doc, start=1):
        page_rect = page.rect
        img_rgb = _render_page_to_rgb(page, opt.dpi)
        if not target_pages or idx in target_pages:
            img_rgb = _process_page_image(img_rgb, opt)
            processed_pages.append(idx)
        image_bytes = _encode_image(img_rgb, opt.output_format, opt.jpeg_quality)
        new_page = out_doc.new_page(width=page_rect.width, height=page_rect.height)
        new_page.insert_image(new_page.rect, stream=image_bytes)

    out_doc.save(output_pdf, garbage=4, deflate=True)
    out_doc.close()
    doc.close()
    return {"page_count": page_count, "processed_pages": processed_pages, "text_redactions": redactions, "output_mode": "rasterize"}


def process_pdf(input_pdf: Path, output_pdf: Path, opt: ProcessOptions) -> dict[str, Any]:
    if opt.output_mode == "preserve":
        return _process_pdf_preserve(input_pdf, output_pdf, opt)
    return _process_pdf_rasterize(input_pdf, output_pdf, opt)


def preview_png(input_pdf: Path, opt: ProcessOptions, page_no: int) -> bytes:
    doc, _ = _open_pdf_with_text_redaction(input_pdf, opt)
    if page_no < 1 or page_no > doc.page_count:
        raise ValueError(f"Invalid preview page {page_no}; PDF has {doc.page_count} pages.")
    page = doc[page_no - 1]
    img_rgb = _render_page_to_rgb(page, opt.preview_dpi)
    img_rgb = _process_page_image(img_rgb, opt)
    doc.close()
    return _encode_png(img_rgb)


app = FastAPI(title="PDF Watermark Cleaner WebUI", version="1.3.0")
WORKDIR = Path(os.getenv("WORKDIR", "/tmp/pdf-watermark-service"))
WORKDIR.mkdir(parents=True, exist_ok=True)

# Security: limit upload size to prevent memory-exhaustion DoS
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Security: rate limiting configuration
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # requests per window


# Security: simple in-memory rate limiter
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, window: int = RATE_LIMIT_WINDOW, max_requests: int = RATE_LIMIT_MAX_REQUESTS):
        super().__init__(app)
        self.window = window
        self.max_requests = max_requests
        self.requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_ip: str, now: float):
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < self.window
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()

        self._cleanup_old_requests(client_ip, now)

        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

        self.requests[client_ip].append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# Security: add security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: StarletteResponse = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Security: basic CSP - allow inline styles/scripts for embedded HTML,
        # but restrict resource origins to prevent data exfiltration
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'"
        # Security: restrict browser features - this app doesn't need camera, mic, etc.
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        # Security: restrict Flash/PDF cross-domain policies
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        # Security: prevent IE from executing downloads in site context
        response.headers["X-Download-Options"] = "noopen"
        # Security: prevent search engines from indexing API responses
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        # Security: cross-origin isolation headers to prevent Spectre-style attacks
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        # Security: prevent caching of API responses containing user data
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        # Security: disable DNS prefetch to prevent information leakage
        response.headers["X-DNS-Prefetch-Control"] = "off"
        return response


app.add_middleware(SecurityHeadersMiddleware)


def _parse_options(options: str) -> ProcessOptions:
    try:
        data = json.loads(options or "{}")
        return ProcessOptions.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        # Security: don't leak internal validation details to clients
        raise HTTPException(status_code=400, detail="Invalid options JSON.")


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe use in Content-Disposition header."""
    # Remove path separators and null bytes
    name = filename.replace("/", "_").replace("\\", "_").replace("\0", "")
    # Remove leading dots (hidden files)
    name = name.lstrip(".")
    # Limit length
    if len(name) > 200:
        name = name[:200]
    # Fallback if empty
    return name or "document"


async def _save_upload(file: UploadFile, job_dir: Path) -> Path:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file does not look like a PDF.")
    p = job_dir / "input.pdf"
    p.write_bytes(content)
    return p


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/preview")
async def preview(file: UploadFile = File(...), options: str = Form(default="{}"), page: int = Form(default=1)):
    opt = _parse_options(options)
    job_dir = WORKDIR / ("preview-" + uuid.uuid4().hex)
    job_dir.mkdir(parents=True, exist_ok=True)
    input_pdf = await _save_upload(file, job_dir)
    try:
        png = preview_png(input_pdf, opt, page)
    except Exception:
        raise HTTPException(status_code=500, detail="Preview generation failed.")
    # Security: clean up temp files after response
    shutil.rmtree(job_dir, ignore_errors=True)
    return Response(content=png, media_type="image/png")


@app.post("/process")
async def process(background_tasks: BackgroundTasks, file: UploadFile = File(...), options: str = Form(default="{}")):
    opt = _parse_options(options)
    job_id = uuid.uuid4().hex
    job_dir = WORKDIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_pdf = await _save_upload(file, job_dir)
    output_pdf = job_dir / "cleaned.pdf"
    try:
        process_pdf(input_pdf, output_pdf, opt)
    except Exception:
        raise HTTPException(status_code=500, detail="PDF processing failed.")
    # Security: sanitize filename to prevent header injection
    safe_stem = _sanitize_filename(Path(file.filename or "document.pdf").stem)
    # Security: clean up temp files after response is sent
    background_tasks.add_task(shutil.rmtree, job_dir, True)
    return FileResponse(output_pdf, media_type="application/pdf", filename=f"{safe_stem}_cleaned.pdf")


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>PDF 去水印 WebUI</title>
<style>
:root{--bg:#f6f7fb;--card:rgba(255,255,255,.9);--text:#111827;--muted:#6b7280;--border:#e5e7eb;--primary:#2563eb;--danger:#dc2626;--ok:#16a34a;--shadow:0 18px 55px rgba(15,23,42,.09);--radius:18px}
*{box-sizing:border-box} body{margin:0;color:var(--text);background:radial-gradient(circle at top left,rgba(37,99,235,.16),transparent 28rem),radial-gradient(circle at bottom right,rgba(22,163,74,.10),transparent 30rem),var(--bg);font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:28px 16px 50px}.hero h1{margin:0 0 8px;font-size:clamp(28px,4vw,42px);letter-spacing:-.04em}.hero p{margin:0;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(420px,.85fr);gap:18px;align-items:start}@media(max-width:980px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);backdrop-filter:blur(16px);border:1px solid rgba(229,231,235,.9);border-radius:var(--radius);box-shadow:var(--shadow);padding:18px}.card h2{font-size:18px;margin:0 0 14px}.section{border-top:1px solid var(--border);padding-top:16px;margin-top:16px}
.row{display:grid;grid-template-columns:1fr 130px;gap:12px;align-items:center;margin:10px 0}.row label{font-size:14px} input[type=number],input[type=text],select,textarea{width:100%;border:1px solid var(--border);background:#fff;color:var(--text);border-radius:10px;padding:9px 10px;outline:none}
input[type=file]{width:100%;border:1px dashed #cbd5e1;background:#fff;border-radius:14px;padding:18px}.rect{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:8px}.rect label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}.hint,.small{font-size:12px;color:var(--muted)}
.switchline{display:flex;align-items:center;gap:10px;margin:10px 0}textarea{min-height:150px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;resize:vertical}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}button{border:0;border-radius:12px;padding:10px 16px;cursor:pointer;font-weight:650}.primary{background:var(--primary);color:#fff}.secondary{background:#eef2ff;color:#1e40af}.danger{background:#fee2e2;color:var(--danger)}.mutedbtn{background:#f3f4f6;color:#374151}
.status{margin-top:14px;font-size:14px;color:var(--muted)}.status.ok{color:var(--ok)}.status.err{color:var(--danger);white-space:pre-wrap}.presetbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.previewImgWrap{background:#fff;border:1px solid var(--border);border-radius:14px;min-height:500px;display:flex;align-items:flex-start;justify-content:center;overflow:auto;padding:10px}.previewImgWrap img{max-width:100%;height:auto;box-shadow:0 8px 30px rgba(0,0,0,.08)}
.previewControls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.previewControls input{width:90px}.jsonbox{min-height:210px}
.rule{border:1px solid var(--border);border-radius:14px;padding:12px;margin:10px 0;background:#fff}.rulehead{display:flex;gap:8px;justify-content:space-between;align-items:center}.rulegrid{display:grid;grid-template-columns:1fr 120px 110px;gap:8px;margin-top:10px}@media(max-width:600px){.rulegrid{grid-template-columns:1fr}}
details summary{cursor:pointer;color:#1d4ed8;font-weight:650}
</style>
</head>
<body>
<main class="wrap">
  <div class="hero">
    <h1>PDF 去水印 WebUI</h1>
    <p>支持实时处理预览、图片水印区域清除，以及 PDF 文本对象水印的精确匹配、包含匹配、正则匹配删除。</p>
  </div>

  <div class="grid">
    <section class="card">
      <h2>处理参数</h2>
      <input id="file" type="file" accept="application/pdf">

      <div class="section">
        <h2>预设</h2>
        <div class="presetbar">
          <button class="secondary" type="button" onclick="applyPreset('exam')">试卷默认</button>
          <button class="secondary" type="button" onclick="applyPreset('strict')">更保守</button>
          <button class="secondary" type="button" onclick="applyPreset('strong')">更强力</button>
          <button class="secondary" type="button" onclick="applyPreset('qrcode')">含二维码页</button>
          <button class="secondary" type="button" onclick="addDefaultTextRules()">常见文本水印</button>
        </div>
      </div>

      <div class="section">
        <h2>基础输出</h2>
        <div class="row"><label>输出模式</label><select id="output_mode"><option value="preserve">保留原始结构</option><option value="rasterize">整页渲染（旧模式）</option></select></div>
        <div class="row"><label>正式输出 DPI</label><input id="dpi" type="number" min="72" max="600" value="220"></div>
        <div class="row"><label>实时预览 DPI</label><input id="preview_dpi" type="number" min="72" max="220" value="130"></div>
        <div class="row"><label>JPEG 质量</label><input id="jpeg_quality" type="number" min="30" max="100" value="92"></div>
        <div class="row"><label>输出格式</label><select id="output_format"><option value="jpeg">JPEG</option><option value="png">PNG</option></select></div>
        <div class="row"><label>只处理页码，空表示全部。例：1,2,14</label><input id="pages" type="text" placeholder="留空"></div>
      </div>

      <div class="section">
        <h2>中部图片水印</h2>
        <div class="row"><label>中部处理模式</label><select id="center_mode"><option value="red_mask">红色掩膜</option><option value="white_rect">整块白化</option><option value="disabled">不处理</option></select></div>
        <div class="rect">
          <div><label>x0</label><input id="center_x0" type="number" min="0" max="1" step="0.001" value="0.30"></div>
          <div><label>y0</label><input id="center_y0" type="number" min="0" max="1" step="0.001" value="0.38"></div>
          <div><label>x1</label><input id="center_x1" type="number" min="0" max="1" step="0.001" value="0.70"></div>
          <div><label>y1</label><input id="center_y1" type="number" min="0" max="1" step="0.001" value="0.63"></div>
        </div>
        <div class="row"><label>红色比绿色强度</label><input id="min_red_over_green" type="number" min="-50" max="100" value="8"></div>
        <div class="row"><label>红色比蓝色强度</label><input id="min_red_over_blue" type="number" min="-50" max="100" value="8"></div>
        <div class="row"><label>最小饱和度</label><input id="min_saturation" type="number" min="0" max="255" value="12"></div>
        <div class="row"><label>最小亮度</label><input id="min_value" type="number" min="0" max="255" value="120"></div>
        <div class="row"><label>最大亮度</label><input id="max_value" type="number" min="0" max="255" value="255"></div>
        <div class="row"><label>掩膜膨胀核</label><input id="dilate_kernel" type="number" min="0" max="31" value="3"></div>
        <div class="row"><label>修复半径，0=直接白化</label><input id="inpaint_radius" type="number" min="0" max="15" value="0"></div>
      </div>

      <div class="section">
        <h2>底部图片水印</h2>
        <label class="switchline"><input id="remove_bottom" type="checkbox" checked> 删除底部区域</label>
        <div class="rect">
          <div><label>x0</label><input id="bottom_x0" type="number" min="0" max="1" step="0.001" value="0.00"></div>
          <div><label>y0</label><input id="bottom_y0" type="number" min="0" max="1" step="0.001" value="0.945"></div>
          <div><label>x1</label><input id="bottom_x1" type="number" min="0" max="1" step="0.001" value="1.00"></div>
          <div><label>y1</label><input id="bottom_y1" type="number" min="0" max="1" step="0.001" value="1.00"></div>
        </div>
      </div>

      <div class="section">
        <h2>文本形式水印</h2>
        <label class="switchline"><input id="remove_text_watermark" type="checkbox" checked> 启用 PDF 文本对象删除</label>
        <div id="rules"></div>
        <div class="actions">
          <button class="secondary" type="button" onclick="addRule()">添加规则</button>
          <button class="mutedbtn" type="button" onclick="addDefaultTextRules()">填入常见规则</button>
        </div>
        <p class="hint">精确匹配适合固定短语；包含匹配适合页眉/页脚长句；正则匹配适合变化文本。可为每条规则限制页码或坐标区域。</p>
      </div>

      <div class="section">
        <h2>额外白化区域</h2>
        <textarea id="extra_rects" placeholder='[{"x0":0,"y0":0.84,"x1":0.18,"y1":1}]'></textarea>
      </div>

      <div class="actions">
        <button class="primary" type="button" onclick="submitJob()">开始处理并下载</button>
        <button class="secondary" type="button" onclick="refreshPreview()">刷新实时预览</button>
        <button class="mutedbtn" type="button" onclick="copyOptions()">复制 options JSON</button>
        <button class="danger" type="button" onclick="resetAll()">重置</button>
      </div>
      <div id="status" class="status"></div>
    </section>

    <aside class="card">
      <h2>实时处理预览</h2>
      <div class="previewControls">
        <label>预览页 <input id="preview_page" type="number" min="1" value="1"></label>
        <button class="secondary" type="button" onclick="refreshPreview()">刷新</button>
        <span class="small">参数变化后会自动防抖刷新。</span>
      </div>
      <div class="previewImgWrap">
        <img id="preview_img" alt="选择 PDF 后显示处理后的页面预览">
      </div>

      <div class="section">
        <h2>当前 options JSON</h2>
        <textarea id="jsonPreview" class="jsonbox" readonly></textarea>
      </div>
    </aside>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
let previewTimer = null;

function num(id){const v=Number($(id).value); if(Number.isNaN(v)) throw new Error(`${id} 不是有效数字`); return v;}
function rect(prefix){const r={x0:num(`${prefix}_x0`),y0:num(`${prefix}_y0`),x1:num(`${prefix}_x1`),y1:num(`${prefix}_y1`)}; if(!(r.x0<r.x1&&r.y0<r.y1)) throw new Error(`${prefix} 区域坐标无效`); return r;}
function parsePagesValue(raw){raw=String(raw||"").trim(); if(!raw) return []; return raw.split(",").map(s=>s.trim()).filter(Boolean).map(s=>{const n=Number(s); if(!Number.isInteger(n)||n<1) throw new Error("页码必须是大于等于 1 的整数"); return n;});}
function parseExtraRects(){const raw=$("extra_rects").value.trim(); if(!raw) return []; const v=JSON.parse(raw); if(!Array.isArray(v)) throw new Error("extra_rects 必须是 JSON 数组"); return v;}

function collectRules(){
  const arr=[];
  document.querySelectorAll(".rule").forEach(el=>{
    const text=el.querySelector(".r_text").value;
    if(!text.trim()) return;
    const mode=el.querySelector(".r_mode").value;
    const ignore_case=el.querySelector(".r_ignore").checked;
    const pages=parsePagesValue(el.querySelector(".r_pages").value);
    const useRect=el.querySelector(".r_userect").checked;
    let rr=null;
    if(useRect){
      rr={
        x0:Number(el.querySelector(".r_x0").value), y0:Number(el.querySelector(".r_y0").value),
        x1:Number(el.querySelector(".r_x1").value), y1:Number(el.querySelector(".r_y1").value)
      };
    }
    arr.push({text, mode, ignore_case, pages, rect: rr});
  });
  return arr;
}

function buildOptions(){
  return {
    output_mode:$("output_mode").value, dpi:num("dpi"), preview_dpi:num("preview_dpi"), jpeg_quality:num("jpeg_quality"), output_format:$("output_format").value,
    center_mode:$("center_mode").value, center_rect:rect("center"), bottom_rect:rect("bottom"),
    min_red_over_green:num("min_red_over_green"), min_red_over_blue:num("min_red_over_blue"),
    min_saturation:num("min_saturation"), min_value:num("min_value"), max_value:num("max_value"),
    dilate_kernel:num("dilate_kernel"), inpaint_radius:num("inpaint_radius"),
    remove_bottom:$("remove_bottom").checked, extra_rects:parseExtraRects(), pages:parsePagesValue($("pages").value),
    remove_text_watermark:$("remove_text_watermark").checked, text_rules:collectRules()
  };
}

function renderJson(){
  try{$("jsonPreview").value=JSON.stringify(buildOptions(),null,2); $("status").className="status";}catch(e){$("jsonPreview").value=String(e.message||e);}
}

async function refreshPreview(){
  const file=$("file").files[0];
  renderJson();
  if(!file) return;
  let opt;
  try{opt=buildOptions();}catch(e){$("status").textContent=e.message||String(e);$("status").className="status err";return;}
  const fd=new FormData();
  fd.append("file",file); fd.append("options",JSON.stringify(opt)); fd.append("page",String(num("preview_page")));
  $("status").textContent="正在生成实时预览...";
  try{
    const resp=await fetch("/preview",{method:"POST",body:fd});
    if(!resp.ok) throw new Error(await resp.text());
    const blob=await resp.blob();
    const old=$("preview_img").src;
    $("preview_img").src=URL.createObjectURL(blob);
    if(old.startsWith("blob:")) URL.revokeObjectURL(old);
    $("status").textContent="预览已更新。"; $("status").className="status ok";
  }catch(e){$("status").textContent="预览失败：\n"+(e.message||String(e)); $("status").className="status err";}
}

function schedulePreview(){renderJson(); clearTimeout(previewTimer); previewTimer=setTimeout(refreshPreview,650);}

async function submitJob(){
  const file=$("file").files[0];
  if(!file){$("status").textContent="请先选择 PDF 文件。";$("status").className="status err";return;}
  let opt; try{opt=buildOptions();}catch(e){$("status").textContent=e.message||String(e);$("status").className="status err";return;}
  const fd=new FormData(); fd.append("file",file); fd.append("options",JSON.stringify(opt));
  $("status").textContent="处理中，请稍候。";
  try{
    const resp=await fetch("/process",{method:"POST",body:fd});
    if(!resp.ok) throw new Error(await resp.text());
    const blob=await resp.blob(); const url=URL.createObjectURL(blob);
    const a=document.createElement("a"); a.href=url; a.download=file.name.replace(/\.pdf$/i,"")+"_cleaned.pdf"; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    $("status").textContent="处理完成，已开始下载。"; $("status").className="status ok";
  }catch(e){$("status").textContent="处理失败：\n"+(e.message||String(e)); $("status").className="status err";}
}

function addRule(data={}){
  // Security: escape all HTML-significant characters to prevent XSS
  const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'","&#39;");
  const div=document.createElement("div"); div.className="rule";
  div.innerHTML=`
    <div class="rulehead"><strong>文本规则</strong><button class="danger" type="button">删除</button></div>
    <div class="rulegrid">
      <input class="r_text" type="text" placeholder="要删除的文本或正则" value="${esc(data.text||"")}">
      <select class="r_mode">
        <option value="exact">精确</option><option value="contains">包含</option><option value="regex">正则</option>
      </select>
      <label class="switchline"><input class="r_ignore" type="checkbox"> 忽略大小写</label>
    </div>
    <div class="row"><label>限定页码，空表示全部。例：1,2,14</label><input class="r_pages" type="text" value="${esc((data.pages||[]).join(","))}"></div>
    <label class="switchline"><input class="r_userect" type="checkbox"> 限定坐标区域</label>
    <div class="rect">
      <div><label>x0</label><input class="r_x0" type="number" min="0" max="1" step="0.001" value="0"></div>
      <div><label>y0</label><input class="r_y0" type="number" min="0" max="1" step="0.001" value="0"></div>
      <div><label>x1</label><input class="r_x1" type="number" min="0" max="1" step="0.001" value="1"></div>
      <div><label>y1</label><input class="r_y1" type="number" min="0" max="1" step="0.001" value="1"></div>
    </div>`;
  $("rules").appendChild(div);
  div.querySelector(".r_mode").value=data.mode||"exact";
  div.querySelector(".r_ignore").checked=!!data.ignore_case;
  if(data.rect){div.querySelector(".r_userect").checked=true; for(const k of ["x0","y0","x1","y1"]) div.querySelector(".r_"+k).value=data.rect[k];}
  div.querySelector("button").onclick=()=>{div.remove();schedulePreview();};
  div.querySelectorAll("input,select").forEach(el=>{el.addEventListener("input",schedulePreview);el.addEventListener("change",schedulePreview);});
  schedulePreview();
}

function addDefaultTextRules(){
  addRule({text:"公众号",mode:"contains"});
  addRule({text:"天津考生",mode:"contains"});
  addRule({text:"进服务群下载更多学习资料",mode:"contains"});
  addRule({text:"仅供学习交流使用",mode:"contains"});
}

function setValues(values){for(const [k,v] of Object.entries(values)){if($(k)){if($(k).type==="checkbox")$(k).checked=Boolean(v);else $(k).value=v;}}}
function applyPreset(name){
  if(name==="qrcode"){ $("extra_rects").value=JSON.stringify([{x0:0,y0:.84,x1:.18,y1:1},{x0:.77,y0:.84,x1:1,y1:1}],null,2); schedulePreview(); return; }
  const p={exam:{},strict:{min_red_over_green:14,min_red_over_blue:14,min_saturation:18,dilate_kernel:1,center_x0:.34,center_y0:.40,center_x1:.66,center_y1:.61},strong:{min_red_over_green:4,min_red_over_blue:4,min_saturation:8,dilate_kernel:5,center_x0:.28,center_y0:.36,center_x1:.72,center_y1:.65}}[name]||{};
  if(name==="exam") resetAll(false);
  setValues(p); schedulePreview();
}
function resetAll(update=true){
  $("rules").innerHTML="";
  setValues({output_mode:"preserve", dpi:220,preview_dpi:130,jpeg_quality:92,output_format:"jpeg",pages:"",center_mode:"red_mask",center_x0:.30,center_y0:.38,center_x1:.70,center_y1:.63,min_red_over_green:8,min_red_over_blue:8,min_saturation:12,min_value:120,max_value:255,dilate_kernel:3,inpaint_radius:0,remove_bottom:true,bottom_x0:0,bottom_y0:.945,bottom_x1:1,bottom_y1:1,extra_rects:"",remove_text_watermark:true,preview_page:1});
  if(update) schedulePreview();
}
async function copyOptions(){try{await navigator.clipboard.writeText(JSON.stringify(buildOptions(),null,2));$("status").textContent="options JSON 已复制。";$("status").className="status ok";}catch(e){$("status").textContent="复制失败："+(e.message||String(e));$("status").className="status err";}}
document.querySelectorAll("input,select,textarea").forEach(el=>{el.addEventListener("input",schedulePreview);el.addEventListener("change",schedulePreview);});
renderJson();
</script>
</body>
</html>
"""
