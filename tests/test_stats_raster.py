"""Tests for the pixel-level raster comparator (ADR-031)."""

from pathlib import Path

from PIL import Image

from mdi.stats.raster import RasterDiff, raster_diff


def _save(img: Image.Image, path: Path, **kwargs: int) -> str:
    """Write *img* as PNG and return the path as a string."""
    img.save(path, format="PNG", **kwargs)
    return str(path)


def test_raster_diff_sees_an_rgb_change_the_alpha_channel_hides(tmp_path: Path) -> None:
    """A colour-only change under uniform alpha must read as different.

    Pillow's ``Image.getbbox()`` defaults to ``alpha_only=True`` on RGBA, so a
    difference confined to the colour channels reads as "identical" through
    that API -- the false pass ADR-031's decision records (five changed
    figures were restored as noise on 8/25). The comparator must see it,
    down to a single channel step: the tolerance is zero.
    """
    # Given: two fully opaque images, one pixel's blue channel apart by 1/255
    base = Image.new("RGBA", (8, 6), (10, 20, 30, 255))
    changed = base.copy()
    changed.putpixel((3, 2), (10, 20, 31, 255))
    left = _save(base, tmp_path / "a.png")
    right = _save(changed, tmp_path / "b.png")
    # When: compared
    diff = raster_diff(left, right)
    # Then: one pixel, one channel step, not identical
    assert diff == RasterDiff(n_pixels_differing=1, max_channel_delta=1, size_mismatch=False)
    assert diff.identical is False


def test_raster_diff_reads_reencoded_identical_pixels_as_identical(tmp_path: Path) -> None:
    """Different PNG bytes carrying the same pixels are encoding noise, not change."""
    # Given: one structured image written at two compression levels -> different bytes
    raw = bytes((x * 7 + y * 13) % 256 for y in range(64) for x in range(64))
    img = Image.frombytes("L", (64, 64), raw).convert("RGBA")
    fast = _save(img, tmp_path / "fast.png", compress_level=0)
    small = _save(img, tmp_path / "small.png", compress_level=9)
    assert Path(fast).read_bytes() != Path(small).read_bytes()
    # When: compared as pixels (one side as raw bytes, covering both input forms)
    diff = raster_diff(fast, Path(small).read_bytes())
    # Then: identical at the pixel level
    assert diff.identical is True
    assert diff == RasterDiff(n_pixels_differing=0, max_channel_delta=0, size_mismatch=False)


def test_raster_diff_flags_a_size_mismatch_as_fully_different(tmp_path: Path) -> None:
    """Canvases that cannot be aligned are not identical, whatever their pixels."""
    # Given: a 4x4 and a 5x4 canvas of the same colour
    a = _save(Image.new("RGBA", (4, 4), (0, 0, 0, 255)), tmp_path / "a.png")
    b = _save(Image.new("RGBA", (5, 4), (0, 0, 0, 255)), tmp_path / "b.png")
    # When: compared
    diff = raster_diff(a, b)
    # Then: reported as every pixel of the larger canvas differing
    assert diff == RasterDiff(n_pixels_differing=20, max_channel_delta=255, size_mismatch=True)
    assert diff.identical is False
