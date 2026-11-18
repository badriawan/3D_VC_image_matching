"""Small dependency-free SVG comparison plots for experiment outputs."""

from collections import defaultdict
from html import escape
from pathlib import Path
import statistics


def write_charts(rows, output):
    """Plot scene means, retaining failures and marking missing values as N/A."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["method"])].append(row)
    keys = sorted(groups)
    for field, title, subtitle in (
        ("inliers", "Geometrically consistent matches", "Mean inliers per pair; geometry failures count as zero"),
        ("inlier_ratio", "Geometric consistency", "Mean inlier ratio on estimated models only; not matching accuracy"),
        ("total_ms", "Processing time (ms)", "Mean of per-pair median times; loading, preprocessing and saving excluded"),
    ):
        width, left, plot_width = 1060, 340, 560
        height = 130 + len(keys) * 48
        values = []
        for key in keys:
            available = [r[field] for r in groups[key] if isinstance(r.get(field), (int, float))]
            values.append(statistics.mean(available) if available else None)
        maximum = 1.0 if field == "inlier_ratio" else max(
            [v for v in values if v is not None] + [1.0]) * 1.1
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}" role="img">',
                 f'<title>{escape(title)}</title>', '<rect width="100%" height="100%" fill="white"/>',
                 '<g font-family="sans-serif" fill="#243247">',
                 f'<text x="24" y="30" font-size="21" font-weight="bold">{escape(title)}</text>',
                 f'<text x="24" y="56" font-size="13">{escape(subtitle)}</text>']
        for tick in range(5):
            value = tick * maximum / 4
            x = left + plot_width * tick / 4
            parts.extend([f'<line x1="{x}" y1="84" x2="{x}" y2="{height-22}" stroke="#e5e9ef"/>',
                          f'<text x="{x}" y="79" text-anchor="middle" font-size="11">{value:.2f}</text>'])
        for i, ((scene, method), value) in enumerate(zip(keys, values)):
            y = 99 + 48 * i
            group = groups[(scene, method)]
            ok = sum(r.get("status") == "ok" for r in group)
            label = escape(f"{scene} / {method}")
            parts.append(f'<text x="24" y="{y+17}" font-size="13">{label}</text>')
            if value is not None:
                size = plot_width * value / maximum
                color = "#2874b5" if method == "SIFT" else "#d77b27"
                parts.append(f'<rect x="{left}" y="{y}" width="{size:.2f}" height="25" fill="{color}"/>')
                value_label = f"{value:.3f}" if field == "inlier_ratio" else f"{value:.1f}"
            else:
                value_label = "N/A"
            parts.append(f'<text x="{left+plot_width+12}" y="{y+17}" font-size="12">'
                         f'{value_label} ({ok}/{len(group)} models)</text>')
        parts.append('</g></svg>')
        (Path(output) / f"{field}.svg").write_text("\n".join(parts), encoding="utf-8")
