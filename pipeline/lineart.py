"""Stage 1: anime image to grayscale line-art extraction."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LINEART_ANIME = "lineart-anime"
ANIME2SKETCH = "anime2sketch"
SUPPORTED_EXTRACTORS = (LINEART_ANIME, ANIME2SKETCH)


class SketchExtractor(Protocol):
    name: str

    def extract(self, src: Path) -> Image.Image:
        """Extract an 8-bit grayscale sketch from *src*."""


def collect_images(
    input_root: Path,
    output_root: Path,
    max_images: int | None,
) -> list[tuple[Path, Path]]:
    """Collect source/destination image pairs for line-art extraction.

    The output path mirrors the source folder layout and always uses ``.png``.
    Existing outputs are skipped before the limit is applied, so the command can
    be resumed safely against a flat portraits folder.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    limit = None if max_images is None or max_images <= 0 else max_images

    pairs: list[tuple[Path, Path]] = []
    for src in sorted(input_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = src.relative_to(input_root).with_suffix(".png")
        dst = output_root / relative_path
        if dst.exists():
            continue

        pairs.append((src, dst))
        if limit is not None and len(pairs) >= limit:
            break

    return pairs


def extractor_output_name(extractor: str) -> str:
    return extractor.replace("-", "_")




def autocrop_black_borders(
    image: "Image.Image",
    threshold: float = 10.0,
    min_band_width: int = 4,
    min_retained: int = 64,
) -> "Image.Image":
    """Detect and crop solid-black letterbox/pillarbox padding from edges.

    Scans *contiguous* dark bands inward from each edge: a border is only
    detected when every pixel in every row/column of the band is below
    ``threshold``.  This avoids removing rows or columns that contain
    genuine dark foreground detail.

    Args:
        image: RGB or RGBA PIL image.
        threshold: Maximum pixel brightness (0-255) for a pixel to be
            considered part of a solid-black border.
        min_band_width: Minimum number of contiguous edge rows/columns
            required before a border is recognized.  Prevents micro-crops
            from single dark edge lines.
        min_retained: Minimum width and height the cropped image must
            retain.  If cropping would shrink either dimension below this,
            the image is returned unchanged.

    Returns:
        Cropped PIL image, or the original if no qualifying border was found.
    """
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # --- scan contiguous dark bands inward from each edge ----------------
    def _scan_top(arr: "np.ndarray") -> int:
        """Return the number of contiguous all-dark rows from the top."""
        for i in range(arr.shape[0]):
            if arr[i].max() > threshold:
                return i
        return arr.shape[0]

    top = _scan_top(gray)
    bottom = _scan_top(gray[::-1])
    left = _scan_top(gray.T)
    right = _scan_top(gray.T[::-1])

    # Only crop sides that form a genuine border band
    top = top if top >= min_band_width else 0
    bottom = bottom if bottom >= min_band_width else 0
    left = left if left >= min_band_width else 0
    right = right if right >= min_band_width else 0

    if top + bottom + left + right == 0:
        return image

    new_w = w - left - right
    new_h = h - top - bottom

    # Guard: do not shrink below minimum dimensions
    if new_w < min_retained or new_h < min_retained:
        return image

    cropped = image.crop((left, top, left + new_w, top + new_h))
    print(
        f"[autocrop] Removed border  T={top} B={bottom} L={left} R={right}  "
        f"({w}x{h} -> {new_w}x{new_h})"
    )
    return cropped




_PREPROCESSING_CONFIG_FILE = ".autocrop_config.json"


def write_preprocessing_config(output_dir: Path, *, autocrop: bool) -> None:
    """Write a preprocessing configuration marker to the output directory.

    On subsequent runs, :func:`check_preprocessing_config` compares the
    saved settings with the current ones to detect incompatible mixtures.

    The marker is written atomically via a temporary file to avoid leaving
    a truncated file if the process is interrupted.
    """
    config = {"autocrop": autocrop, "version": 1}
    config_path = output_dir / _PREPROCESSING_CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    tmp_path.replace(config_path)


def _dir_has_images(output_dir: Path) -> bool:
    """Return True if *output_dir* contains at least one image file."""
    return any(
        p.suffix.lower() in IMAGE_EXTENSIONS
        for p in output_dir.rglob("*")
        if p.is_file() and p.name != _PREPROCESSING_CONFIG_FILE
    )


def check_preprocessing_config(output_dir: Path, *, autocrop: bool) -> None:
    """Validate that a resumed run uses compatible preprocessing settings.

    Raises
    ------
    RuntimeError
        * The output directory contains images but no configuration marker
          and *autocrop* is enabled (legacy uncropped outputs must not be
          silently mixed with cropped ones).
        * The marker exists but is unreadable, has invalid JSON, or is
          missing required fields.
        * The marker records a different *autocrop* setting than requested.
    """
    config_path = output_dir / _PREPROCESSING_CONFIG_FILE

    if not config_path.exists():
        # No marker — only a problem when the directory already has images
        # and the caller wants autocrop enabled, because those images were
        # produced without cropping.
        if autocrop and _dir_has_images(output_dir):
            raise RuntimeError(
                f"Preprocessing mismatch in {output_dir}: "
                f"existing outputs have no preprocessing marker and were "
                f"likely produced without autocrop. Use --no-autocrop to "
                f"continue, or use a fresh output directory for cropped "
                f"extraction."
            )
        return

    # --- marker exists: it must be valid and compatible -----------------
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read preprocessing marker {config_path}: {exc}"
        ) from exc

    try:
        saved = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Corrupted preprocessing marker {config_path}: {exc}. "
            f"Delete the file and retry, or use a fresh output directory."
        ) from exc

    if not isinstance(saved, dict):
        raise RuntimeError(
            f"Invalid preprocessing marker {config_path}: expected a JSON "
            f"object, got {type(saved).__name__}."
        )

    if "autocrop" not in saved:
        raise RuntimeError(
            f"Invalid preprocessing marker {config_path}: missing required "
            f"'autocrop' field. Delete the file and retry."
        )

    saved_autocrop = saved["autocrop"]
    if not isinstance(saved_autocrop, bool):
        raise RuntimeError(
            f"Invalid preprocessing marker {config_path}: 'autocrop' must "
            f"be a boolean, got {type(saved_autocrop).__name__}."
        )

    if saved_autocrop != autocrop and _dir_has_images(output_dir):
        raise RuntimeError(
            f"Preprocessing mismatch in {output_dir}: "
            f"existing outputs used autocrop={saved_autocrop}, "
            f"but current run uses autocrop={autocrop}. "
            f"Use a fresh output directory or pass matching settings."
        )


def create_extractor(
    extractor: str,
    detect_resolution: int,
    image_resolution: int,
    autocrop: bool = True,
    anime2sketch_dir: str | Path | None = None,
    anime2sketch_python: str | Path | None = None,
    anime2sketch_model: str = "default",
    anime2sketch_gpu_ids: str = "",
    anime2sketch_clahe_clip: float = -1.0,
) -> SketchExtractor:
    """Load a sketch extractor by name."""
    if extractor == LINEART_ANIME:
        return ControlNetLineartAnimeExtractor(
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            autocrop=autocrop,
        )

    if extractor == ANIME2SKETCH:
        return Anime2SketchExtractor(
            repo_dir=anime2sketch_dir,
            python_executable=anime2sketch_python,
            load_size=image_resolution,
            model=anime2sketch_model,
            gpu_ids=anime2sketch_gpu_ids,
            clahe_clip=anime2sketch_clahe_clip,
        )

    supported = ", ".join(SUPPORTED_EXTRACTORS)
    raise ValueError(f"Unknown extractor '{extractor}'. Supported: {supported}")


class ControlNetLineartAnimeExtractor:
    name = LINEART_ANIME

    def __init__(
        self,
        detect_resolution: int,
        image_resolution: int,
        autocrop: bool = True,
    ) -> None:
        print("[init] Loading ControlNet LineartAnimeDetector ...")
        from controlnet_aux import LineartAnimeDetector  # lazy import

        self.detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        self.detect_resolution = detect_resolution
        self.image_resolution = image_resolution
        self.autocrop = autocrop

    def extract(self, src: Path) -> Image.Image:
        with Image.open(src) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if self.autocrop:
                image = autocrop_black_borders(image)
            detected = self.detector(
                image,
                detect_resolution=self.detect_resolution,
                image_resolution=self.image_resolution,
            )

        return _to_grayscale(detected)


class Anime2SketchExtractor:
    name = ANIME2SKETCH

    def __init__(
        self,
        repo_dir: str | Path | None,
        python_executable: str | Path | None,
        load_size: int,
        model: str,
        gpu_ids: str,
        clahe_clip: float,
    ) -> None:
        if not repo_dir:
            raise RuntimeError(
                "Anime2Sketch requires --anime2sketch-dir or ANIME2SKETCH_DIR "
                "pointing to a local Mukosame/Anime2Sketch checkout."
            )

        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.script_path = self.repo_dir / "test.py"
        if not self.script_path.exists():
            raise RuntimeError(f"Anime2Sketch test.py not found: {self.script_path}")

        self.python_executable = _resolve_python_executable(python_executable)
        self.load_size = load_size
        self.model = model
        self.gpu_ids = gpu_ids.strip()
        self.clahe_clip = clahe_clip

        print(f"[init] Using Anime2Sketch checkout: {self.repo_dir}")

    def extract(self, src: Path) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="anime2sketch_") as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                self.python_executable,
                str(self.script_path),
                "--dataroot",
                str(src),
                "--load_size",
                str(self.load_size),
                "--output_dir",
                str(output_dir),
                "--model",
                self.model,
            ]
            if self.gpu_ids:
                cmd.extend(["--gpu_ids", self.gpu_ids])
            if self.clahe_clip > 0:
                cmd.extend(["--clahe_clip", str(self.clahe_clip)])

            try:
                subprocess.run(
                    cmd,
                    cwd=self.repo_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                details = stderr or stdout or f"exit code {exc.returncode}"
                raise RuntimeError(f"Anime2Sketch failed for {src}: {details}") from exc

            output_path = output_dir / src.name
            if not output_path.exists():
                candidates = sorted(
                    path
                    for path in output_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not candidates:
                    raise RuntimeError(f"Anime2Sketch did not write an image for {src}")
                output_path = candidates[0]

            with Image.open(output_path) as image:
                return _to_grayscale(image)


def process_image(src: Path, dst: Path, extractor: SketchExtractor) -> bool:
    """Run a sketch extractor on one source image."""
    try:
        sketch = extractor.extract(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sketch.save(dst)
        return True
    except Exception as exc:
        print(f"[skip] {src}: {exc}")
        return False


def _resolve_python_executable(python_executable: str | Path | None) -> str:
    if not python_executable:
        return sys.executable

    python_text = str(python_executable)
    python_path = Path(python_text).expanduser()
    if python_path.is_absolute() or python_path.parent != Path("."):
        # Keep virtualenv executables as their venv path. Path.resolve() follows
        # the venv python symlink back to /usr/bin/python, losing site-packages.
        return str(python_path.absolute())

    return python_text


def _to_grayscale(image_like: Any) -> Image.Image:
    """Preserve line confidence for thresholding during vectorization."""

    image = _to_pil_image(image_like)
    return ImageOps.grayscale(image)


def _to_binary_grayscale(image_like: Any) -> Image.Image:
    """Compatibility helper for callers that still require legacy binary output."""

    gray = _to_grayscale(image_like)
    return gray.point(lambda pixel: 0 if pixel < 128 else 255, mode="L")


def _to_pil_image(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like

    try:
        return Image.fromarray(image_like)
    except Exception as exc:
        image_type = type(image_like)
        raise TypeError(f"Detector returned unsupported image type: {image_type!r}") from exc
