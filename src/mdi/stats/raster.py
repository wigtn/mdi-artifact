"""Pixel-level comparison for raster figure assets (ADR-031).

The determinism invariant reads rasters pixel-wise, not byte-wise: the zlib
bundled per platform re-encodes identical pixels into different bytes. This
module is the comparator the ADR-031 procedure runs on. It never calls
``Image.getbbox()``: on RGBA images that method inspects only the alpha
channel by default (``alpha_only=True`` since Pillow 9.2), so two figures
with different titles compare as "identical" -- the false pass recorded in
ADR-031's decision. ``tests/test_stats_raster.py`` pins that scenario.
"""

from io import BytesIO
from typing import NamedTuple, Union

from PIL import Image, ImageChops


class RasterDiff(NamedTuple):
    """Outcome of a pixel-level comparison between two rasters."""

    n_pixels_differing: int
    max_channel_delta: int
    size_mismatch: bool

    @property
    def identical(self) -> bool:
        """True when both rasters decode to exactly the same pixels (tolerance 0)."""
        return not self.size_mismatch and self.n_pixels_differing == 0


def _load_rgba(source: Union[str, bytes]) -> Image.Image:
    """Decode *source* -- a file path or raw file bytes -- to an RGBA image."""
    if isinstance(source, bytes):
        return Image.open(BytesIO(source)).convert("RGBA")
    return Image.open(source).convert("RGBA")


def raster_diff(left: Union[str, bytes], right: Union[str, bytes]) -> RasterDiff:
    """Compare two rasters pixel-wise over all four RGBA channels.

    Returns the number of pixels whose channels differ at all and the largest
    per-channel difference -- the two numbers the ADR-031 procedure reports. A
    size mismatch cannot be aligned, so it is returned as every pixel of the
    larger canvas differing at full amplitude.
    """
    a = _load_rgba(left)
    b = _load_rgba(right)
    if a.size != b.size:
        larger = max(a.size[0] * a.size[1], b.size[0] * b.size[1])
        return RasterDiff(n_pixels_differing=larger, max_channel_delta=255, size_mismatch=True)
    # Reduce the RGBA difference to one band holding each pixel's largest
    # channel delta; its histogram then yields both numbers without touching
    # getbbox or any per-pixel Python loop.
    channels = ImageChops.difference(a, b).split()
    strongest = channels[0]
    for band in channels[1:]:
        strongest = ImageChops.lighter(strongest, band)
    histogram = strongest.histogram()
    max_delta = max((delta for delta, count in enumerate(histogram) if count), default=0)
    return RasterDiff(
        n_pixels_differing=sum(histogram[1:]), max_channel_delta=max_delta, size_mismatch=False
    )
