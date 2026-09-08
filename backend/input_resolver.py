

import io
import os
import re
import logging
from typing import Optional, Union

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("input_resolver")

ImageInput = Union[str, bytes, io.IOBase, Image.Image]


_BARCODE_RE = re.compile(r"^\d{8,14}$")


class InputResolutionError(Exception):
    pass


def is_barcode(text: str) -> bool:
    return bool(_BARCODE_RE.match((text or "").strip()))


def load_image(image_input: ImageInput) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input))
    if isinstance(image_input, str) and os.path.exists(image_input):
        return Image.open(image_input)
    if hasattr(image_input, "read"): 
        return Image.open(image_input)
    raise InputResolutionError("Unrecognized image input type — expected path, bytes, file-like, or PIL.Image.")


def extract_barcode(image: Image.Image) -> Optional[str]:
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        logger.warning("pyzbar not installed — skipping barcode detection. `pip install pyzbar` + libzbar0.")
        return None

    try:
        results = decode(image)
    except Exception as e:
        logger.warning("Barcode decoding failed: %s", e)
        return None

    for result in results:
        data = result.data.decode("utf-8", errors="ignore").strip()
        if is_barcode(data):
            return data
    return None


def extract_text(image: Image.Image) -> str:
    try:
        import pytesseract
    except ImportError:
        raise InputResolutionError(
            "pytesseract not installed and no barcode was found in the image. "
            "`pip install pytesseract` + install the tesseract-ocr system package."
        )
    try:
        raw = pytesseract.image_to_string(image)
    except Exception as e:
        raise InputResolutionError(f"OCR failed: {e}")

    # Keep the longest alphabetic line — usually the brand/product name on packaging
    lines = [l.strip() for l in raw.splitlines() if re.search(r"[A-Za-z]{3,}", l)]
    if not lines:
        raise InputResolutionError("Could not read any product name text from the image.")
    return max(lines, key=len)


def resolve_image_input(image_input: ImageInput) -> dict:
    image = load_image(image_input)
    barcode = extract_barcode(image)
    if barcode:
        return {"type": "barcode", "value": barcode}

    text = extract_text(image)
    return {"type": "name", "value": text}


def resolve_input(user_input: Union[str, ImageInput]) -> dict:

    if isinstance(user_input, str) and not os.path.exists(user_input):
        text = user_input.strip()
        if is_barcode(text):
            return {"type": "barcode", "value": text}
  
        if not re.search(r"\.(jpg|jpeg|png|webp|bmp)$", text, re.IGNORECASE):
            return {"type": "name", "value": text}

    return resolve_image_input(user_input)
