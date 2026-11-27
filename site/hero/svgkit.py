"""Small helpers for hand-built cartoon figures (flat pastel, thin dark outlines, Comic Neue)."""

INK = "#2B2B2B"
FONT = "Comic Neue"
C = dict(
    blue="#DCEBFA", blue_s="#8FBCE8", peach="#FDE6DA", peach_s="#F0B59A", mint="#DDF3E4", mint_s="#94D0A9",
    lav="#ECE6FA", lav_s="#B7A9E6", yellow="#FFF1B0", yellow_s="#E2C35A", pink="#FBE0E3", pink_s="#EBA3AC",
    screen="#23263A", eye="#CFE8FF", accent="#8EC5F0", ant="#F7C948", green="#2E9E4F", red="#D9453B",
    bar_mem="#7FB6EA", bar_ctrl="#8FD3A6", bar_none="#B9A8EE", wood="#D9A36A", wood_s="#9C6B3A",
)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="white"/>\n{body}\n</svg>\n')


def text(x, y, s, size=26, weight="bold", anchor="middle", fill=INK, style=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{fill}" {style}>{s}</text>')


def lines(x, y, rows, size=26, gap=None, **kw):
    gap = gap or size * 1.15
    return "".join(text(x, y + i * gap, r, size, **kw) for i, r in enumerate(rows))


def rrect(x, y, w, h, r=18, fill="white", stroke=INK, sw=3, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" {extra}/>')


def panel(x, y, w, h, fill, stroke):
    return rrect(x, y, w, h, 30, fill, stroke, 3)


def tag(cx, y, s, fill, stroke, size=34, pad=34):
    w = len(s) * size * 0.50 + 2 * pad
    return rrect(cx - w / 2, y, w, size * 1.6, size * 0.8, fill, INK, 3) + text(cx, y + size * 1.12, s, size)


def pill(cx, cy, s, fill, size=24, pad=18, stroke=INK, weight="bold"):
    w = len(s) * size * 0.52 + 2 * pad
    h = size * 1.55
    return rrect(cx - w / 2, cy - h / 2, w, h, h / 2, fill, stroke, 2.5) + text(cx, cy + size * 0.36, s, size, weight)


def arrow(x1, y1, x2, y2, color=INK, sw=3.5, dash="", head=13):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx1, hy1 = x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45)
    hx2, hy2 = x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45)
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" {d}/>'
            f'<path d="M{hx1:.1f},{hy1:.1f} L{x2},{y2} L{hx2:.1f},{hy2:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')


def curve_arrow(d, end, ang, color=INK, sw=3.5, dash="", head=13):
    """Path d, arrow head at end=(x,y) pointing along angle ang (radians)."""
    import math
    x2, y2 = end
    hx1, hy1 = x2 - head * math.cos(ang - 0.45), y2 - head * math.sin(ang - 0.45)
    hx2, hy2 = x2 - head * math.cos(ang + 0.45), y2 - head * math.sin(ang + 0.45)
    dd = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" {dd}/>'
            f'<path d="M{hx1:.1f},{hy1:.1f} L{x2},{y2} L{hx2:.1f},{hy2:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')


def check(cx, cy, s=1.0, color=None):
    color = color or C["green"]
    return (f'<path d="M{cx - 14 * s},{cy} L{cx - 4 * s},{cy + 11 * s} L{cx + 16 * s},{cy - 12 * s}" fill="none" '
            f'stroke="{color}" stroke-width="{6 * s}" stroke-linecap="round" stroke-linejoin="round"/>')


def cross(cx, cy, s=1.0, color=None):
    color = color or C["red"]
    k = 12 * s
    return (f'<path d="M{cx - k},{cy - k} L{cx + k},{cy + k} M{cx + k},{cy - k} L{cx - k},{cy + k}" '
            f'stroke="{color}" stroke-width="{6 * s}" stroke-linecap="round"/>')


def sparkle(x, y, s=1.0, color=None):
    color = color or C["ant"]
    return (f'<g stroke="{color}" stroke-width="{4 * s}" stroke-linecap="round">'
            f'<line x1="{x}" y1="{y}" x2="{x + 6 * s}" y2="{y - 14 * s}"/>'
            f'<line x1="{x + 12 * s}" y1="{y + 4 * s}" x2="{x + 24 * s}" y2="{y - 6 * s}"/>'
            f'<line x1="{x + 16 * s}" y1="{y + 16 * s}" x2="{x + 30 * s}" y2="{y + 12 * s}"/></g>')


def robot(x, y, s=1.0, badge=None, happy=True, sparkles=True, arms_up=False):
    """Robot ~120 x 150 units at scale 1, top-left (x, y)."""
    eyes = ('<path d="M42,57 q6,-9 12,0 M66,57 q6,-9 12,0" fill="none" stroke="#CFE8FF" stroke-width="4" '
            'stroke-linecap="round"/>') if happy else \
           '<circle cx="48" cy="54" r="5" fill="#CFE8FF"/><circle cx="72" cy="54" r="5" fill="#CFE8FF"/>'
    mouth = '<path d="M53,65 q7,6 14,0" fill="none" stroke="#CFE8FF" stroke-width="3.5" stroke-linecap="round"/>'
    b = badge or ""
    arms = ('<ellipse cx="26" cy="104" rx="9" ry="13" fill="white" stroke="#2B2B2B" stroke-width="3"/>'
            '<ellipse cx="94" cy="104" rx="9" ry="13" fill="white" stroke="#2B2B2B" stroke-width="3"/>')
    sp = sparkle(98, 14, 0.8) if sparkles else ""
    return (f'<g transform="translate({x},{y}) scale({s})">'
            '<line x1="60" y1="10" x2="60" y2="26" stroke="#2B2B2B" stroke-width="3"/>'
            '<circle cx="60" cy="9" r="7" fill="#F7C948" stroke="#2B2B2B" stroke-width="3"/>'
            '<circle cx="17" cy="55" r="11" fill="#8EC5F0" stroke="#2B2B2B" stroke-width="3"/>'
            '<circle cx="103" cy="55" r="11" fill="#8EC5F0" stroke="#2B2B2B" stroke-width="3"/>'
            '<rect x="20" y="24" width="80" height="62" rx="28" fill="white" stroke="#2B2B2B" stroke-width="3"/>'
            '<rect x="30" y="35" width="60" height="40" rx="16" fill="#23263A"/>'
            f'{eyes}{mouth}'
            '<rect x="40" y="128" width="16" height="16" rx="6" fill="white" stroke="#2B2B2B" stroke-width="3"/>'
            '<rect x="64" y="128" width="16" height="16" rx="6" fill="white" stroke="#2B2B2B" stroke-width="3"/>'
            f'{arms}'
            '<rect x="32" y="86" width="56" height="48" rx="18" fill="white" stroke="#2B2B2B" stroke-width="3"/>'
            '<rect x="49" y="101" width="22" height="11" rx="5" fill="#8EC5F0"/>'
            f'{b}{sp}</g>')


def notebook(x, y, w, h, label="", fill=None, size=26, rings=True, lined=False):
    fill = fill or C["yellow"]
    out = rrect(x, y, w, h, 8, fill, INK, 3)
    if rings:
        for i in range(4):
            yy = y + 14 + i * (h - 28) / 3
            out += f'<circle cx="{x}" cy="{yy:.1f}" r="5" fill="white" stroke="{INK}" stroke-width="2.5"/>'
    if lined:
        for i in range(3):
            yy = y + h * 0.35 + i * h * 0.18
            out += f'<line x1="{x + w * .25}" y1="{yy}" x2="{x + w * .85}" y2="{yy}" stroke="{INK}" stroke-width="2.5"/>'
    if label:
        out += text(x + w / 2 + 4, y + h / 2 + size * 0.35, label, size)
    return out


def bullets(x, y, n, w, gap=26, color=INK):
    out = ""
    for i in range(n):
        yy = y + i * gap
        out += f'<circle cx="{x}" cy="{yy}" r="3.5" fill="{color}"/>'
        out += f'<line x1="{x + 14}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
    return out


def brace(x, y1, y2, w=16, sw=3, color=INK):
    """Right-pointing curly brace from y1 (top) to y2 (bottom) at x."""
    m = (y1 + y2) / 2
    return (f'<path d="M{x},{y1} q{w},0 {w},{w} L{x + w},{m - w} q0,{w} {w},{w} q-{w},0 -{w},{w} '
            f'L{x + w},{y2 - w} q0,{w} -{w},{w}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round"/>')
