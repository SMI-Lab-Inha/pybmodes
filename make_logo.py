"""Render logo.png from the SVG design using matplotlib (no Cairo needed)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon
from matplotlib.path import Path
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# ── Canvas ──────────────────────────────────────────────────────────────────
# SVG viewBox is 256×256, y-down.  We work in the same coords and invert y.
W = H = 256
fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=200)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # invert y so SVG coords map directly
ax.set_aspect('equal')
ax.axis('off')

# ── Background gradient (approximated with a linear patch) ──────────────────
bg_grad = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(
    np.dstack([bg_grad, bg_grad, bg_grad]),
    extent=[0, W, H, 0], aspect='auto', zorder=0, alpha=0
)
# Solid dark background
bg = patches.FancyBboxPatch((4, 4), W-8, H-8,
                             boxstyle="round,pad=0,rounding_size=36",
                             linewidth=0, zorder=0)
bg.set_facecolor('#0b1e30')
ax.add_patch(bg)
# Subtle diagonal gradient overlay via a second rect
bg2 = patches.FancyBboxPatch((4, 4), W-8, H-8,
                              boxstyle="round,pad=0,rounding_size=36",
                              linewidth=0, zorder=0, alpha=0.35)
bg2.set_facecolor('#14324a')
ax.add_patch(bg2)

# ── Reference circles ────────────────────────────────────────────────────────
for r, ec in [(72, '#214763'), (54, '#1a3b53')]:
    ax.add_patch(Circle((128, 110), r, fill=False,
                         edgecolor=ec, linewidth=1.2, zorder=1))

# ── Ground curve ─────────────────────────────────────────────────────────────
gx = np.array([73, 98, 116, 128, 141, 159, 185], dtype=float)
gy = np.array([194, 183, 180, 181, 182, 186, 196], dtype=float)
ax.plot(gx, gy, color='#264d68', lw=5,
        solid_capstyle='round', zorder=2)

# ── Tower (tapered trapezoid) ────────────────────────────────────────────────
tower_pts = np.array([[120, 186], [136, 186], [132, 82], [124, 82]])
ax.add_patch(Polygon(tower_pts, closed=True, zorder=3,
                     facecolor='#c0d2e4', edgecolor='none'))

# ── Base ellipse ─────────────────────────────────────────────────────────────
ax.add_patch(Ellipse((128, 186), 36, 11,
                      facecolor='#7b93a8', zorder=4, alpha=0.9))

# ── Tower 1st FA mode shape ──────────────────────────────────────────────────
# SVG path: M128 186 C125 170, 121 153, 122 135 C123 118, 130 101, 139 86
tx = np.array([128, 125, 121, 122, 122, 123, 130, 139, 139], dtype=float)
ty = np.array([186, 170, 153, 135, 135, 118, 101, 86,  86],  dtype=float)
# Smooth cubic bezier via many points
def cubic_bezier(p0, p1, p2, p3, n=100):
    t = np.linspace(0, 1, n)
    return ((1-t)**3)[:,None]*p0 + 3*((1-t)**2*t)[:,None]*p1 + \
           3*((1-t)*t**2)[:,None]*p2 + (t**3)[:,None]*p3

seg1 = cubic_bezier(np.array([128,186]), np.array([125,170]),
                     np.array([121,153]), np.array([122,135]))
seg2 = cubic_bezier(np.array([122,135]), np.array([123,118]),
                     np.array([130,101]), np.array([139,86]))
ms_tower = np.vstack([seg1, seg2])

# Glow layer
ax.plot(ms_tower[:,0], ms_tower[:,1], color='#57f0cf',
        lw=9, alpha=0.18, zorder=5, solid_capstyle='round')
# Main line with gradient-like colour
ax.plot(ms_tower[:,0], ms_tower[:,1],
        lw=4.5, zorder=6, solid_capstyle='round',
        color='#3ee8cf')

# ── Blades ────────────────────────────────────────────────────────────────────
blades = [
    ((128, 82), (71,  57)),
    ((128, 82), (189, 95)),
    ((128, 82), (120, 24)),
]
for (x0,y0),(x1,y1) in blades:
    ax.plot([x0,x1],[y0,y1], color='#d8e5ef', lw=8,
            solid_capstyle='round', zorder=3)

# ── Blade 1st flap mode shape (solid, upper blade) ───────────────────────────
seg_b = cubic_bezier(np.array([128,82]), np.array([126,65]),
                      np.array([136,48]), np.array([149,38]))
ax.plot(seg_b[:,0], seg_b[:,1], color='#57f0cf',
        lw=8, alpha=0.18, zorder=5, solid_capstyle='round')
ax.plot(seg_b[:,0], seg_b[:,1], color='#3ee8cf',
        lw=4, zorder=6, solid_capstyle='round')

# ── Blade dashed mode shape (lower-left blade) ────────────────────────────────
seg_d = cubic_bezier(np.array([128,82]), np.array([118,77]),
                      np.array([104,72]), np.array([90,68]))
ax.plot(seg_d[:,0], seg_d[:,1], color='#85fff2',
        lw=2.2, zorder=6, alpha=0.6,
        linestyle=(0, (4, 5)), dash_capstyle='round')

# ── Hub ───────────────────────────────────────────────────────────────────────
ax.add_patch(Circle((128, 82), 11, facecolor='#9cf7ef', zorder=7))
ax.add_patch(Circle((128, 82),  5, facecolor='#0f2436', zorder=8))

# ── Wordmark ──────────────────────────────────────────────────────────────────
ax.text(128, 226, '', ha='center', va='center')  # anchor
ax.text(127.5, 226, 'py',
        ha='right', va='center',
        fontsize=27, fontweight='bold',
        fontfamily='sans-serif', color='#9cf7ef', zorder=10)
ax.text(128.5, 226, 'bmodes',
        ha='left', va='center',
        fontsize=27, fontweight='bold',
        fontfamily='sans-serif', color='#f3f7fb', zorder=10)

fig.savefig('logo.png', dpi=200, bbox_inches='tight',
            facecolor='#0b1e30', edgecolor='none', pad_inches=0)
plt.close(fig)
print("logo.png written.")
