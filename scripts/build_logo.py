"""Bake the LEO logo SVG with text converted to <path> elements.

Pixel-identical rendering across machines requires the SVG to not
depend on system fonts. We extract glyph outlines from local font
files and embed them as <path> data scaled into a clean viewBox.

Sources:
- DejaVu Sans Bold for "LEO" (Latin)
- Noto Sans CJK SC Bold for "力行智能"
"""
from __future__ import annotations

from pathlib import Path
from fontTools.ttLib import TTFont, TTCollection
from fontTools.pens.svgPathPen import SVGPathPen


DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NOTO_CJK_COLLECTION = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def load_noto_sans_cjk_sc_bold() -> TTFont:
    """Find the 'Noto Sans CJK SC' face inside the .ttc collection."""
    coll = TTCollection(NOTO_CJK_COLLECTION)
    for f in coll.fonts:
        names = f["name"].names
        for n in names:
            try:
                s = n.toUnicode()
            except Exception:
                continue
            if "Noto Sans CJK SC" in s and "Mono" not in s:
                return f
    raise RuntimeError("Noto Sans CJK SC not found in collection")


def glyph_paths(font: TTFont, text: str, *, font_size: float, x_offset: float, baseline_y: float, fill: str) -> tuple[list[str], float]:
    """Convert `text` to a list of <path> elements positioned starting at
    (x_offset, baseline_y). Returns (path_strings, end_x)."""
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    units_per_em = font["head"].unitsPerEm
    scale = font_size / units_per_em
    hmtx = font["hmtx"]

    paths: list[str] = []
    pen_x = x_offset
    for ch in text:
        codepoint = ord(ch)
        glyph_name = cmap.get(codepoint)
        if glyph_name is None:
            raise RuntimeError(f"no glyph for U+{codepoint:04X} ({ch})")
        glyph = glyph_set[glyph_name]
        pen = SVGPathPen(glyph_set)
        glyph.draw(pen)
        d = pen.getCommands()
        if d:
            # Transform: translate to (pen_x, baseline_y), scale, flip Y
            # (font outlines are y-up; SVG is y-down).
            transform = f"translate({pen_x:.3f} {baseline_y:.3f}) scale({scale:.6f} {-scale:.6f})"
            paths.append(
                f'  <path d="{d}" transform="{transform}" fill="{fill}"/>'
            )
        advance_width, _lsb = hmtx[glyph_name]
        pen_x += advance_width * scale
    return paths, pen_x


def main() -> None:
    leo_font = TTFont(DEJAVU)
    cjk_font = load_noto_sans_cjk_sc_bold()

    LEO_SIZE = 86
    CJK_SIZE = 78
    BASELINE_Y = 80
    GAP = 24  # gap between "LEO" and "力行智能"

    leo_paths, leo_end_x = glyph_paths(
        leo_font, "LEO",
        font_size=LEO_SIZE, x_offset=0, baseline_y=BASELINE_Y,
        fill="#FFB158",
    )
    cjk_x = leo_end_x + GAP
    cjk_paths, cjk_end_x = glyph_paths(
        cjk_font, "力行智能",
        font_size=CJK_SIZE, x_offset=cjk_x, baseline_y=BASELINE_Y,
        fill="#FF5F05",
    )

    width = int(cjk_end_x + 4)
    height = 100

    parts = [
        "<!-- LEO brand mark — text baked to <path> elements for",
        "     pixel-identical rendering. Generated from",
        "     DejaVu Sans Bold (LEO) + Noto Sans CJK SC Bold (力行智能).",
        "     Source: scripts/build_logo.py — do not hand-edit. -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">',
        *leo_paths,
        *cjk_paths,
        "</svg>",
    ]
    out = "\n".join(parts) + "\n"

    target = Path("/home/yuan/git/my/leo/web/src/assets/leo-logo.svg")
    target.write_text(out)
    print(f"wrote {target} ({len(out):,} bytes, viewBox 0 0 {width} {height})")


if __name__ == "__main__":
    main()
