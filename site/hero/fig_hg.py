"""Hero illustration of the project page: a hand-drawn SVG redraw of an AI-generated concept sketch.
Build: python3 fig_hg.py && rsvg-convert -f svg fig_hg.svg -o ../../docs/hero.svg  (text becomes vector paths;
needs the Comic Neue font and Droid Sans Fallback installed for fontconfig).  scripts/build_site.py asserts 3,600 graphs."""
from svgkit import *
import random

W, H = 2150, 720
b = []

# ---------------- panel 1: plain loop ----------------
b.append(panel(12, 58, 676, 650, C["blue"], C["blue_s"]))
b.append(tag(350, 24, "Plain loop", "#CFE3F8", C["blue_s"], size=38))
b.append(robot(58, 214, 1.55))
# desk + laptop with a bug
b.append(f'<line x1="40" y1="520" x2="470" y2="520" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')
b.append(f'<path d="M96,388 L286,388 L306,512 L76,512 Z" fill="#AEB6C2" stroke="{INK}" stroke-width="3.5" stroke-linejoin="round"/>'
         f'<rect x="58" y="510" width="270" height="14" rx="6" fill="#8E97A4" stroke="{INK}" stroke-width="3"/>')
bug = (f'<g transform="translate(190,452)"><g stroke="{INK}" stroke-width="3" stroke-linecap="round">'
       f'<line x1="-26" y1="-10" x2="-12" y2="-4"/><line x1="-28" y1="6" x2="-14" y2="4"/><line x1="-24" y1="20" x2="-12" y2="10"/>'
       f'<line x1="26" y1="-10" x2="12" y2="-4"/><line x1="28" y1="6" x2="14" y2="4"/><line x1="24" y1="20" x2="12" y2="10"/>'
       f'<line x1="-6" y1="-24" x2="-12" y2="-34"/><line x1="6" y1="-24" x2="12" y2="-34"/></g>'
       f'<ellipse cx="0" cy="2" rx="16" ry="21" fill="#F28B82" stroke="{INK}" stroke-width="3"/>'
       f'<circle cx="0" cy="-20" r="9" fill="{INK}"/><line x1="0" y1="-12" x2="0" y2="22" stroke="{INK}" stroke-width="2.5"/></g>')
b.append(bug + sparkle(226, 430, 0.8, C["red"]))
# patch -> grader
b.append(f'<g transform="rotate(-6 352 330)"><path d="M300,270 h96 l20,20 v112 h-116 z" fill="white" stroke="{INK}" stroke-width="3" stroke-linejoin="round"/>'
         + text(356, 312, "patch", 26) + bullets(322, 340, 2, 70, gap=24) + '</g>')
b.append(arrow(424, 342, 470, 342, INK, 4))
b.append(rrect(480, 262, 178, 178, 18, "#EEF0F6", INK, 3) + text(569, 302, "Grader", 30) +
         rrect(512, 322, 114, 96, 12, "white", INK, 3) + text(569, 392, "?", 64, fill="#8A8F99"))
# sticky note
b.append(f'<g transform="rotate(-5 200 616)">' + rrect(60, 560, 300, 112, 6, C["yellow"], INK, 3) +
         lines(210, 602, ["Works on my machine"], 27) +
         f'<text x="210" y="648" font-size="30" font-weight="bold" text-anchor="middle" fill="{INK}" '
         f'font-family="Comic Neue, Droid Sans Fallback">¯\\_(ツ)_/¯</text></g>')
b.append(text(560, 520, "no enforced", 25, weight="normal") + text(560, 548, "check", 25, weight="normal"))

# ---------------- panel 2: check and retry ----------------
b.append(panel(706, 58, 720, 650, C["peach"], C["peach_s"]))
b.append(tag(1066, 24, "Check and retry", "#FAD5C4", C["peach_s"], size=38))
b.append(text(1066, 128, "tests visible · a successful run is drawn", 22, weight="normal", style='font-style="italic"'))
b.append(robot(988, 150, 1.25))
b.append(rrect(722, 356, 132, 92, 22, C["lav"], INK, 3.5) + lines(788, 394, ["Edit", "code"], 27, gap=30))
b.append(rrect(906, 356, 132, 92, 22, C["mint"], INK, 3.5, 'stroke-dasharray="9 6"') + lines(972, 394, ["Run", "tests"], 27, gap=30))
b.append(rrect(1090, 362, 120, 80, 40, "white", INK, 3.5) + text(1150, 411, "Submit", 27))
b.append(arrow(856, 402, 902, 402, INK, 4) + text(879, 384, "done", 20, weight="normal"))
b.append(arrow(1040, 402, 1086, 402, INK, 4) + text(1063, 384, "pass", 20, weight="normal"))
b.append(arrow(1212, 402, 1240, 402, INK, 4))
b.append(rrect(1248, 314, 160, 176, 18, "#EEF0F6", INK, 3) + text(1328, 352, "Grader", 28) +
         rrect(1282, 372, 92, 96, 12, "white", INK, 3) + check(1328, 420, 1.5))
d = "M972,452 C 972,560 794,560 788,456"
b.append(curve_arrow(d, (788, 456), -1.62, C["red"], 4.5))
b.append(text(880, 592, "fail: here is why", 30, fill=C["red"]))
b.append(rrect(760, 614, 262, 50, 8, "#FFF5F4", C["red"], 2) +
         f'<text x="776" y="646" font-size="19" font-family="DejaVu Sans Mono" fill="{C["red"]}">AssertionError ...</text>')
b.append(text(1110, 604, "dashed: a program", 22, weight="normal", anchor="start"))

# ---------------- panel 3: search ----------------
b.append(panel(1444, 58, 694, 650, C["mint"], C["mint_s"]))
b.append(tag(1791, 24, "Search 3,600 graphs", "#C9EDD5", C["mint_s"], size=38))
# magnifier
b.append(f'<line x1="2010" y1="560" x2="2098" y2="660" stroke="{C["lav_s"]}" stroke-width="30" stroke-linecap="round"/>'
         f'<line x1="2010" y1="560" x2="2098" y2="660" stroke="{INK}" stroke-width="3" stroke-linecap="round" opacity="0.0"/>')
b.append(f'<circle cx="1830" cy="372" r="250" fill="#F7F6FD" stroke="{C["lav_s"]}" stroke-width="26"/>'
         f'<circle cx="1830" cy="372" r="263" fill="none" stroke="{INK}" stroke-width="3"/>'
         f'<circle cx="1830" cy="372" r="237" fill="none" stroke="{INK}" stroke-width="3"/>')
random.seed(3)
cols = ["#B9A8EE", "#8FD3A6", "#7FB6EA"]


def mini(x, y, n, branch=False, loop=False):
    o = ""
    pts = [(x + i * 30, y) for i in range(n)]
    if branch:
        pts[-1] = (x + (n - 1) * 30, y + 22)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        o += f'<line x1="{x1 + 7}" y1="{y1}" x2="{x2 - 7}" y2="{y2}" stroke="{INK}" stroke-width="2.2"/>'
    if branch:
        o += f'<line x1="{pts[-2][0] + 7}" y1="{y}" x2="{pts[-1][0] - 7}" y2="{y - 22}" stroke="{INK}" stroke-width="2.2"/>'
        o += f'<circle cx="{pts[-1][0]}" cy="{y - 22}" r="7" fill="{cols[0]}" stroke="{INK}" stroke-width="2"/>'
    if loop:
        o += f'<path d="M{pts[1][0]},{y + 8} q-15,22 -30,0" fill="none" stroke="{INK}" stroke-width="2"/>'
    for i, (px, py) in enumerate(pts):
        o += f'<circle cx="{px}" cy="{py}" r="7" fill="{cols[(i + n) % 3]}" stroke="{INK}" stroke-width="2"/>'
    return o


spots = [(1640, 210, 3, 0, 0), (1760, 190, 4, 1, 0), (1890, 200, 3, 0, 1), (1630, 300, 4, 0, 1), (1960, 300, 3, 1, 0),
         (1640, 410, 3, 1, 0), (1660, 500, 4, 0, 0), (1780, 550, 3, 0, 1), (1900, 520, 3, 1, 0), (1990, 420, 2, 0, 0),
         (1560, 360, 2, 0, 0), (1720, 120, 2, 0, 0)]
for x, y, n, br, lp in spots:
    if (x - 1830) ** 2 + (y - 372) ** 2 < 215 ** 2:
        b.append(mini(x, y, n, bool(br), bool(lp)))
b.append(text(2010, 250, "...", 34) + text(1700, 610, "...", 34))
# highlighted pick: check and retry + one extra node
b.append(rrect(1750, 318, 214, 98, 16, C["yellow"], INK, 3.5))
# selected graph: edit -> tests -> submit; fail -> reflect -> edit (extras beyond check and retry)
for x, c in [(1790, C["lav"]), (1858, C["mint"]), (1926, "white")]:
    b.append(f'<circle cx="{x}" cy="352" r="14" fill="{c}" stroke="{INK}" stroke-width="2.6"/>')
b.append(arrow(1805, 352, 1842, 352, INK, 2.8, head=8) + arrow(1873, 352, 1910, 352, INK, 2.8, head=8))
b.append(f'<circle cx="1824" cy="394" r="12" fill="{C["pink"]}" stroke="{INK}" stroke-width="2.6"/>')
b.append(f'<circle cx="1766" cy="394" r="11" fill="#F5B841" stroke="{INK}" stroke-width="2.6"/>')
b.append(arrow(1784, 364, 1772, 382, C["lav_s"], 2.6, head=7) + arrow(1764, 382, 1778, 366, C["lav_s"], 2.6, head=7))
b.append(arrow(1850, 365, 1835, 384, C["red"], 2.8, head=7) + arrow(1814, 384, 1800, 366, C["red"], 2.8, head=7))
b.append(robot(1474, 448, 1.15))
b.append(f'<g transform="rotate(-8 1560 600)"><rect x="1530" y="556" width="62" height="80" rx="6" fill="#D9A36A" stroke="{INK}" stroke-width="3"/>'
         f'<rect x="1538" y="568" width="46" height="60" rx="3" fill="white" stroke="{INK}" stroke-width="2"/>'
         f'<rect x="1550" y="550" width="22" height="12" rx="3" fill="#9AA1AA" stroke="{INK}" stroke-width="2"/></g>')
b.append(rrect(1610, 438, 390, 80, 16, "white", INK, 3) +
         lines(1805, 470, ["search's pick: check and retry", "+ reflect on fail + compact on stall"], 22, gap=27))
b.append(arrow(1700, 436, 1760, 418, INK, 3))

open("fig_hg.svg", "w").write(svg(W, H, "\n".join(b)))
print("wrote fig_hg.svg")
