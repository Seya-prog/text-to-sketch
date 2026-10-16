"""Regression tests for pipeline.lineart.autocrop_black_borders."""

from __future__ import annotations

import unittest

import numpy as np
from pathlib import Path
from PIL import Image

from pipeline.lineart import autocrop_black_borders


def _make_image(arr: np.ndarray) -> Image.Image:
    """Convert an (H, W) uint8 array to a PIL RGB image."""
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


class TestAutocropBlackBorders(unittest.TestCase):
    """Cover dark backgrounds, edge-touching subjects, thin features,
    compressed borders, and true letterboxing."""

    # ── true letterboxing / pillarboxing ────────────────────────────
    def test_letterbox_top_bottom(self) -> None:
        """Horizontal black bars at top and bottom are cropped."""
        arr = np.zeros((200, 120), dtype=np.uint8)
        arr[30:170, :] = 200  # content in the middle
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertEqual(result.size, (120, 140))

    def test_pillarbox_left_right(self) -> None:
        """Vertical black bars at left and right are cropped."""
        arr = np.zeros((80, 100), dtype=np.uint8)
        arr[:, 15:85] = 200
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertEqual(result.size, (70, 80))

    def test_combined_letterbox_pillarbox(self) -> None:
        """All-four-sides padding is cropped."""
        arr = np.zeros((120, 120), dtype=np.uint8)
        arr[10:110, 20:100] = 180
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertEqual(result.size, (80, 100))

    # ── no borders ─────────────────────────────────────────────────
    def test_no_borders_unchanged(self) -> None:
        """An image with no black borders is returned unchanged."""
        arr = np.full((100, 100), 200, dtype=np.uint8)
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertIs(result, img)

    # ── dark backgrounds ───────────────────────────────────────────
    def test_dark_background_with_content_not_cropped(self) -> None:
        """A mostly dark image with scattered bright pixels must NOT be
        cropped aggressively.  Every row and column has at least one
        pixel above threshold, so no contiguous band exists."""
        arr = np.full((100, 100), 5, dtype=np.uint8)
        # Scatter bright pixels in every row and every column
        for i in range(100):
            arr[i, (i * 7) % 100] = 200
            arr[(i * 13) % 100, i] = 200
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertIs(result, img)

    # ── edge-touching subjects ─────────────────────────────────────
    def test_edge_touching_subject(self) -> None:
        """Content touching all 4 edges prevents any cropping."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[0, 50] = 255      # top edge
        arr[99, 50] = 255     # bottom edge
        arr[50, 0] = 255      # left edge
        arr[50, 99] = 255     # right edge
        arr[40:60, 40:60] = 200
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertIs(result, img)

    # ── thin features ──────────────────────────────────────────────
    def test_thin_horizontal_feature_preserved(self) -> None:
        """A thin white line spanning the full width prevents top/bottom crop
        because it breaks the contiguous dark band."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[20:80, 10:90] = 180    # central content
        arr[5, :] = 255            # thin line near top edge
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        # Top should NOT be cropped (thin line at row 5 breaks the band)
        self.assertGreaterEqual(result.height, 75)

    def test_thin_vertical_feature_preserved(self) -> None:
        """A thin white line spanning the full height prevents left/right crop."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[10:90, 20:80] = 180
        arr[:, 3] = 255           # thin line near left edge
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertGreaterEqual(result.width, 77)

    # ── compressed borders (slightly non-zero) ─────────────────────
    def test_compressed_border_with_artifacts(self) -> None:
        """Borders with slight JPEG compression noise (values 1-8) are
        still detected as borders since they are below default threshold=10."""
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 9, size=(100, 80), dtype=np.uint8)
        arr[10:90, :] = 200
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertEqual(result.size, (80, 80))

    # ── min_band_width guard ───────────────────────────────────────
    def test_narrow_band_ignored(self) -> None:
        """Bands narrower than min_band_width are NOT cropped."""
        arr = np.full((100, 100), 200, dtype=np.uint8)
        arr[:2, :] = 0   # only 2px top band (default min_band_width=4)
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        self.assertIs(result, img)

    # ── min_retained guard ─────────────────────────────────────────
    def test_min_retained_prevents_overcrop(self) -> None:
        """If cropping would make the image smaller than min_retained,
        the original is returned unchanged."""
        arr = np.zeros((80, 80), dtype=np.uint8)
        arr[30:50, 30:50] = 200  # only 20x20 content
        img = _make_image(arr)
        result = autocrop_black_borders(img, min_retained=64)
        self.assertIs(result, img)

    # ── the reviewer's specific reproduction case ──────────────────
    def test_reviewer_case_central_square_with_thin_feature(self) -> None:
        """100x100 image with a central white square and a thin white
        feature spanning its width should NOT be cropped to 20x20."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[40:60, 40:60] = 255           # central 20x20 white square
        arr[50, :] = 255                  # thin feature spanning full width
        img = _make_image(arr)
        result = autocrop_black_borders(img)
        # The thin feature at row 50 spanning full width breaks the
        # contiguous dark band from the top and bottom, so no vertical
        # crop should occur.  The full-width line also means columns
        # 0 and 99 have a bright pixel, so no horizontal crop either.
        self.assertEqual(result.size, (100, 100))


class TestPreprocessingConfig(unittest.TestCase):
    """Cover preprocessing marker safeguards."""

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.mkdtemp(prefix="test_preproc_")
        self.output_dir = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write_dummy_image(self, name: str = "test.png") -> None:
        """Create a small dummy image in the output directory."""
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        img.save(self.output_dir / name)

    # ── P1: unmarked directory with existing images ──────────
    def test_unmarked_dir_with_images_rejects_autocrop(self) -> None:
        """An unmarked directory containing images must reject autocrop=True."""
        from pipeline.lineart import check_preprocessing_config
        self._write_dummy_image()
        with self.assertRaises(RuntimeError):
            check_preprocessing_config(self.output_dir, autocrop=True)

    def test_unmarked_dir_with_images_allows_no_autocrop(self) -> None:
        """An unmarked directory containing images is accepted with autocrop=False."""
        from pipeline.lineart import check_preprocessing_config
        self._write_dummy_image()
        # Should not raise
        check_preprocessing_config(self.output_dir, autocrop=False)

    def test_unmarked_empty_dir_allows_autocrop(self) -> None:
        """An empty unmarked directory is accepted with autocrop=True."""
        from pipeline.lineart import check_preprocessing_config
        check_preprocessing_config(self.output_dir, autocrop=True)

    # ── P2: damaged / invalid markers ────────────────────────
    def test_truncated_json_raises(self) -> None:
        """A truncated/corrupt marker must raise, not silently pass."""
        from pipeline.lineart import check_preprocessing_config
        (self.output_dir / ".autocrop_config.json").write_text(
            "{broken", encoding="utf-8"
        )
        with self.assertRaises(RuntimeError):
            check_preprocessing_config(self.output_dir, autocrop=True)

    def test_empty_object_raises(self) -> None:
        """A marker with {} (missing 'autocrop' field) must raise."""
        from pipeline.lineart import check_preprocessing_config
        (self.output_dir / ".autocrop_config.json").write_text(
            "{}", encoding="utf-8"
        )
        with self.assertRaises(RuntimeError):
            check_preprocessing_config(self.output_dir, autocrop=True)

    def test_wrong_type_raises(self) -> None:
        """A marker where 'autocrop' is not a bool must raise."""
        from pipeline.lineart import check_preprocessing_config
        (self.output_dir / ".autocrop_config.json").write_text(
            '{"autocrop": "yes", "version": 1}', encoding="utf-8"
        )
        with self.assertRaises(RuntimeError):
            check_preprocessing_config(self.output_dir, autocrop=True)

    def test_valid_marker_matching_settings_accepted(self) -> None:
        """A valid marker with matching settings is accepted."""
        from pipeline.lineart import (
            check_preprocessing_config,
            write_preprocessing_config,
        )
        write_preprocessing_config(self.output_dir, autocrop=True)
        self._write_dummy_image()
        # Should not raise
        check_preprocessing_config(self.output_dir, autocrop=True)

    def test_valid_marker_mismatched_settings_rejected(self) -> None:
        """A valid marker with different settings is rejected when images exist."""
        from pipeline.lineart import (
            check_preprocessing_config,
            write_preprocessing_config,
        )
        write_preprocessing_config(self.output_dir, autocrop=True)
        self._write_dummy_image()
        with self.assertRaises(RuntimeError):
            check_preprocessing_config(self.output_dir, autocrop=False)

    def test_atomic_write_produces_valid_marker(self) -> None:
        """write_preprocessing_config must produce a readable, valid marker."""
        import json
        from pipeline.lineart import write_preprocessing_config
        write_preprocessing_config(self.output_dir, autocrop=False)
        marker = self.output_dir / ".autocrop_config.json"
        self.assertTrue(marker.exists())
        data = json.loads(marker.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        self.assertIn("autocrop", data)
        self.assertIs(data["autocrop"], False)
        self.assertEqual(data["version"], 1)
        # Temp file should not linger
        self.assertFalse(marker.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
