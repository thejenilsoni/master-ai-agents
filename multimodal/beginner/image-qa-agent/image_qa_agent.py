"""
Image Q&A Agent (Multimodal - Beginner)

The fundamentals of giving a chat model eyes. This agent takes one or more local
image files plus a question and returns an answer grounded in what is actually
visible in the pixels.

It covers the four things that trip people up first:

1. **Encoding** — a local file has to become a `data:` URL
   (`data:image/png;base64,<...>`) before it can ride along in a chat message.
2. **Multiple images in one request** — the `content` list can hold several
   image parts; the model sees them in order, so label them in the prompt.
3. **The `detail` setting** — `"low"` is a flat, cheap 85 tokens; `"high"` tiles
   the image and costs far more. `estimate_image_tokens()` reproduces the tiling
   arithmetic so you can price a request *before* you send it.
4. **Downscaling** — high-detail images are normalised to a fixed tile grid
   anyway, so a 4000x3000 photo costs the same as a 1024x768 one. Shrinking is
   still worth it (upload bytes, latency), but only dropping the short side
   below 768 px actually reduces the token bill — and that is exactly when small
   text becomes unreadable.

Every piece of that arithmetic is a plain function, so it can be checked without
an API key (see `--selftest`).

Run:
    python make_samples.py
    export OPENAI_API_KEY="sk-..."
    python image_qa_agent.py --image samples/whiteboard.png "What is on the whiteboard?"
"""

from __future__ import annotations

import argparse
import base64
import binascii
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Both of these accept image input. Nothing else in this project is model-specific.
DEFAULT_MODEL = "gpt-4o-mini"
VISION_MODELS = ("gpt-4o", "gpt-4o-mini")

# The API rejects individual images above ~20 MB, and a base64 payload is ~33%
# bigger than the file on disk. Fail early with a useful message instead of
# waiting for a 400 from the server.
MAX_ENCODED_BYTES = 20 * 1024 * 1024

# Bound the request: more images means more tokens and worse per-image accuracy.
MAX_IMAGES_PER_REQUEST = 4

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# --------------------------------------------------------------------------- #
# 1. Encoding a local file as a data URL
# --------------------------------------------------------------------------- #
def guess_mime_type(path: str | Path) -> str:
    """Map a file suffix to an image MIME type the API accepts."""
    suffix = Path(path).suffix.lower()
    if suffix not in _MIME_BY_SUFFIX:
        raise ValueError(
            f"Unsupported image type {suffix!r}. Supported: "
            + ", ".join(sorted(_MIME_BY_SUFFIX))
        )
    return _MIME_BY_SUFFIX[suffix]


def to_data_url(data: bytes, mime_type: str) -> str:
    """Wrap raw image bytes in a base64 `data:` URL."""
    encoded = base64.b64encode(data).decode("ascii")
    url = f"data:{mime_type};base64,{encoded}"
    if len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError(
            f"Encoded image is {len(encoded) / 1e6:.1f} MB, over the ~20 MB limit. "
            "Downscale it first (see --max-side)."
        )
    return url


def encode_image_data_url(path: str | Path) -> str:
    """Read an image file from disk and return it as a `data:` URL."""
    path = Path(path)
    return to_data_url(path.read_bytes(), guess_mime_type(path))


def decode_data_url(url: str) -> tuple[str, bytes]:
    """Inverse of `to_data_url`: return (mime_type, raw_bytes).

    Only used for tests and debugging, but having the round-trip available is
    what makes the encoding path verifiable without a network call.
    """
    if not url.startswith("data:"):
        raise ValueError("not a data URL")
    header, _, payload = url.partition(",")
    if not payload or not header.endswith(";base64"):
        raise ValueError("expected a base64-encoded data URL")
    mime_type = header[len("data:") : -len(";base64")]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"corrupt base64 payload: {exc}") from exc
    return mime_type, raw


# --------------------------------------------------------------------------- #
# 2. Downscaling and what it actually costs
# --------------------------------------------------------------------------- #
def fit_within(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Aspect-preserving shrink so the longest side is at most `max_side`.

    Never upscales — enlarging a small image adds bytes without adding detail.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


# Published per-image token costs. `base` is charged once, `tile` once per
# 512x512 tile. These change over time — re-check current pricing before you
# build a budget on them; the *shape* of the formula is the durable lesson.
_TOKEN_COSTS = {
    "gpt-4o": {"base": 85, "tile": 170},
    "gpt-4o-mini": {"base": 2833, "tile": 5667},
}


def count_tiles(width: int, height: int) -> int:
    """Number of 512x512 tiles a high-detail image is split into.

    The image is first shrunk to fit inside 2048x2048, then shrunk again so its
    *shortest* side is 768 px, then cut into 512x512 tiles. Neither step ever
    upscales, which is why a small image can end up as a single tile.
    """
    width, height = fit_within(width, height, 2048)
    shortest = min(width, height)
    if shortest > 768:
        scale = 768 / shortest
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    return math.ceil(width / 512) * math.ceil(height / 512)


def estimate_image_tokens(
    width: int, height: int, detail: str = "high", model: str = DEFAULT_MODEL
) -> int:
    """Estimate the token cost of one image. `detail="low"` is a flat base cost."""
    costs = _TOKEN_COSTS.get(model)
    if costs is None:
        raise ValueError(f"No published token costs for {model!r}. Known: {list(_TOKEN_COSTS)}")
    if detail not in ("low", "high", "auto"):
        raise ValueError("detail must be 'low', 'high', or 'auto'")
    if detail == "low":
        return costs["base"]
    # "auto" lets the server decide; assume the expensive path when budgeting.
    return costs["base"] + costs["tile"] * count_tiles(width, height)


@dataclass
class PreparedImage:
    """An image ready to send, plus everything needed to explain its cost."""

    source: Path
    data_url: str
    width: int
    height: int
    byte_size: int
    detail: str

    def estimated_tokens(self, model: str = DEFAULT_MODEL) -> int:
        return estimate_image_tokens(self.width, self.height, self.detail, model)


def prepare_image(path: str | Path, max_side: int = 1024, detail: str = "auto") -> PreparedImage:
    """Load an image, downscale it if needed, and encode it as a data URL.

    Pillow is imported here rather than at module scope so `--selftest` runs with
    nothing installed. If Pillow is missing we still send the original file —
    correct, just larger and slower.
    """
    path = Path(path)
    raw = path.read_bytes()
    mime_type = guess_mime_type(path)
    try:
        import io

        from PIL import Image
    except ImportError:
        print("[warn] Pillow not installed — sending the original file un-resized.")
        return PreparedImage(path, to_data_url(raw, mime_type), 0, 0, len(raw), detail)

    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        target = fit_within(width, height, max_side)
        if target == (width, height):
            return PreparedImage(path, to_data_url(raw, mime_type), width, height, len(raw), detail)
        resized = img.convert("RGB").resize(target, Image.LANCZOS)

    buffer = io.BytesIO()
    # JPEG at quality 85 is a good default for photos; the visible difference is
    # tiny and the payload is often 5-10x smaller than a re-encoded PNG.
    resized.save(buffer, format="JPEG", quality=85)
    data = buffer.getvalue()
    return PreparedImage(
        path, to_data_url(data, "image/jpeg"), target[0], target[1], len(data), detail
    )


# --------------------------------------------------------------------------- #
# 3. Building the request
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You answer questions about images. Describe only what is actually visible. "
    "When several images are provided they are numbered in the order given; refer "
    "to them by number. If text in the image is too small or blurry to read with "
    "confidence, say so explicitly instead of guessing."
)


def build_vision_messages(question: str, images: list[PreparedImage]) -> list[dict]:
    """Assemble the chat payload: one text part, then one part per image."""
    if not images:
        raise ValueError("at least one image is required")
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"{len(images)} images exceeds the cap of {MAX_IMAGES_PER_REQUEST}. "
            "Send fewer per request — accuracy drops as the count climbs."
        )
    labels = ", ".join(f"image {i} = {img.source.name}" for i, img in enumerate(images, start=1))
    content: list[dict] = [{"type": "text", "text": f"{question}\n\n({labels})"}]
    for img in images:
        content.append(
            {"type": "image_url", "image_url": {"url": img.data_url, "detail": img.detail}}
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def ask(question: str, images: list[PreparedImage], model: str = DEFAULT_MODEL) -> str:
    """Send the question plus images and return the model's answer."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=build_vision_messages(question, images),
        max_tokens=700,
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# 4. Entry points
# --------------------------------------------------------------------------- #
def _tiny_png_bytes(width: int = 2, height: int = 2, rgb: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    """Build a valid PNG using only the standard library (test fixture)."""
    import struct
    import zlib

    scanlines = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _selftest() -> None:
    """Verify encoding, downscale math, and token estimates — no API key needed."""
    import tempfile

    # --- data URL round-trip -------------------------------------------------
    png = _tiny_png_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pixel.png"
        path.write_bytes(png)
        url = encode_image_data_url(path)
        assert url.startswith("data:image/png;base64,"), url[:40]
        mime, raw = decode_data_url(url)
        assert mime == "image/png"
        assert raw == png, "round-trip must return the exact original bytes"

    try:
        decode_data_url("data:image/png;base64,not-base64!!")
        raise AssertionError("corrupt payload should have been rejected")
    except ValueError:
        pass
    try:
        guess_mime_type("scan.bmp")
        raise AssertionError("unsupported suffix should have been rejected")
    except ValueError:
        pass

    # --- downscale math ------------------------------------------------------
    assert fit_within(4000, 3000, 1024) == (1024, 768)
    assert fit_within(3000, 4000, 1024) == (768, 1024)
    assert fit_within(800, 600, 1024) == (800, 600), "must never upscale"
    assert fit_within(100, 4000, 1000) == (25, 1000)

    # --- token estimates -----------------------------------------------------
    assert estimate_image_tokens(4000, 3000, "low", "gpt-4o") == 85
    # High detail normalises to the same tile grid, so the huge photo and the
    # 1024px version cost exactly the same — downscaling here buys upload speed,
    # not tokens.
    big = estimate_image_tokens(4000, 3000, "high", "gpt-4o")
    small = estimate_image_tokens(1024, 768, "high", "gpt-4o")
    assert big == small == 85 + 170 * 4, (big, small)
    # Dropping the short side below 768 is what actually cuts the bill.
    assert estimate_image_tokens(512, 384, "high", "gpt-4o") == 85 + 170
    assert count_tiles(512, 384) == 1
    assert estimate_image_tokens(1024, 768, "high", "gpt-4o-mini") == 2833 + 5667 * 4

    # --- request shape -------------------------------------------------------
    images = [
        PreparedImage(Path("a.png"), "data:image/png;base64,AA==", 1024, 768, 10, "high"),
        PreparedImage(Path("b.png"), "data:image/png;base64,BB==", 640, 480, 8, "low"),
    ]
    messages = build_vision_messages("Compare these.", images)
    parts = messages[1]["content"]
    assert messages[0]["role"] == "system"
    assert parts[0]["type"] == "text" and "image 1 = a.png" in parts[0]["text"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[1]["image_url"]["url"].endswith("AA=="), "image order must be preserved"
    assert parts[2]["image_url"]["detail"] == "low", "per-image detail must survive"

    try:
        build_vision_messages("too many", images * 3)
        raise AssertionError("image cap should have been enforced")
    except ValueError:
        pass

    print("selftest passed: data-URL round-trip, downscale math, tile/token estimates,")
    print(f"  high-detail 4000x3000 == 1024x768 == {big} tokens, 512x384 == 255 tokens (gpt-4o)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about local images.")
    parser.add_argument("question", nargs="*", help="The question to ask about the image(s).")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help=f"Image file; repeat for up to {MAX_IMAGES_PER_REQUEST} images.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=VISION_MODELS)
    parser.add_argument(
        "--detail",
        default="auto",
        choices=("auto", "low", "high"),
        help="'low' is a flat cheap cost; 'high' tiles the image and reads small text better.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Downscale so the longest side is at most this many pixels (default: 1024).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the images and print the cost estimate without calling the API.",
    )
    parser.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.image:
        parser.error("at least one --image is required (run `python make_samples.py` first)")

    question = " ".join(args.question).strip() or "Describe this image in detail."
    images = [prepare_image(p, max_side=args.max_side, detail=args.detail) for p in args.image]

    total = 0
    print("Prepared images:")
    for index, img in enumerate(images, start=1):
        tokens = img.estimated_tokens(args.model)
        total += tokens
        size = f"{img.width}x{img.height}" if img.width else "unknown size"
        print(
            f"  {index}. {img.source.name:<24} {size:>11}  "
            f"{img.byte_size / 1024:7.1f} KB  ~{tokens} tokens ({img.detail})"
        )
    print(f"  estimated image tokens for {args.model}: ~{total}\n")

    if args.dry_run:
        print("--dry-run: nothing was sent.")
        return

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY (copy .env.example to .env) or run --selftest.")

    print(f"Q: {question}\n")
    print(ask(question, images, model=args.model))


if __name__ == "__main__":
    main()
