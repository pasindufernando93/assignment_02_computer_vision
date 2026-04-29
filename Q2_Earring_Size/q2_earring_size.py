import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Camera parameters 
FOCAL_LENGTH_MM = 8.0       # mm
PIXEL_SIZE_MM   = 0.0022    # mm  (2.2 µm)
DISTANCE_MM     = 720.0     # mm  (lens to object plane)

# Scale factor
SCALE = PIXEL_SIZE_MM * DISTANCE_MM / FOCAL_LENGTH_MM   # mm/pixel

print("=" * 55)
print("Camera geometry")
print("=" * 55)
print(f"  Focal length      f = {FOCAL_LENGTH_MM} mm")
print(f"  Pixel size        p = {PIXEL_SIZE_MM*1000:.1f} µm")
print(f"  Object distance   Z = {DISTANCE_MM} mm")
print("\n  Scale factor = p × Z / f")
print(f"              = {PIXEL_SIZE_MM*1000:.1f}µm × {DISTANCE_MM}mm / {FOCAL_LENGTH_MM}mm")
print(f"              = {SCALE*1000:.1f} µm/pixel")
print(f"              = {SCALE:.4f} mm/pixel")


# Load image and detect earrings
img_bgr = cv2.imread('assets/earrings.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Otsu threshold to isolate dark earrings from white background
_, thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological closing to fill small holes inside earrings
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours and keep the two largest
contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # left→right

print("\n" + "=" * 55)
print("Earring measurements")
print("=" * 55)

earring_data = []
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # Minimum enclosing circle for diameter estimate
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)

    # Fit ellipse for more accurate shape description
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        ell_cx, ell_cy = ellipse[0]
        ell_w, ell_h   = ellipse[1]  
    else:
        ell_w = w; ell_h = h  # noqa: E702

    # Convert to real-world dimensions
    W_mm       = w * SCALE
    H_mm       = h * SCALE
    diam_mm    = 2 * radius * SCALE
    ell_W_mm   = ell_w * SCALE
    ell_H_mm   = ell_h * SCALE
    area_px    = cv2.contourArea(cnt)
    area_mm2   = area_px * (SCALE ** 2)

    earring_data.append({
        'bbox_px': (x, y, w, h), 'radius_px': radius,
        'center_px': (cx, cy), 'ellipse': ellipse,
        'w_mm': W_mm, 'h_mm': H_mm,
        'diam_mm': diam_mm, 'area_mm2': area_mm2,
        'ell_w_mm': ell_W_mm, 'ell_h_mm': ell_H_mm,
    })

    print(f"\n  Earring {i+1}:")
    print(f"    Bounding box  : {w} × {h} pixels")
    print(f"    Bounding box  : {W_mm:.2f} × {H_mm:.2f} mm  (real world)")
    print(f"    Enclosing dia : {2*radius:.1f} px  →  {diam_mm:.2f} mm")
    print(f"    Ellipse axes  : {ell_w:.1f} × {ell_h:.1f} px  →  "
          f"{ell_W_mm:.2f} × {ell_H_mm:.2f} mm")
    print(f"    Contour area  : {area_px:.0f} px²  →  {area_mm2:.2f} mm²")
    print(f"    Estimated dia : ≈ {diam_mm:.1f} mm")

# Consistency check — both earrings should be same size
print("\n" + "=" * 55)
print("Consistency check (both earrings same pair → same size)")
print("=" * 55)
d1, d2 = earring_data[0]['diam_mm'], earring_data[1]['diam_mm']
print(f"  Earring 1 diameter: {d1:.2f} mm")
print(f"  Earring 2 diameter: {d2:.2f} mm")
print(f"  Difference:         {abs(d1-d2):.2f} mm  ({100*abs(d1-d2)/d1:.1f}%)")
print(f"  Mean diameter:      {(d1+d2)/2:.2f} mm")


# FIGURE
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Q2 — Earring Size Estimation via Pinhole Camera Model",
             fontsize=12, fontweight='bold')

# Panel 1: Original image with annotations
axes[0].imshow(img_rgb)
axes[0].set_title("Original image\nwith detected boundaries", fontsize=10)
axes[0].axis('off')

COLORS = ['#185FA5', '#A32D2D']
for i, (ed, col) in enumerate(zip(earring_data, COLORS)):
    x, y, w, h = ed['bbox_px']
    rect = patches.Rectangle((x, y), w, h,
                               linewidth=2, edgecolor=col,
                               facecolor='none', linestyle='--')
    axes[0].add_patch(rect)

    cx, cy = ed['center_px']
    r = ed['radius_px']
    circ = patches.Circle((cx, cy), r,
                            linewidth=2, edgecolor=col,
                            facecolor='none')
    axes[0].add_patch(circ)

    axes[0].text(x + w/2, y - 20,
                 f"E{i+1}: ⌀{ed['diam_mm']:.1f}mm\n{ed['w_mm']:.1f}×{ed['h_mm']:.1f}mm",
                 ha='center', fontsize=9, color=col, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'))

# Panel 2: Thresholded mask
axes[1].imshow(thresh_clean, cmap='gray')
axes[1].set_title("Otsu threshold mask\n(earrings isolated)", fontsize=10)
axes[1].axis('off')
for i, (ed, col) in enumerate(zip(earring_data, COLORS)):
    x, y, w, h = ed['bbox_px']
    rect = patches.Rectangle((x, y), w, h,
                               linewidth=2, edgecolor=col, facecolor='none')
    axes[1].add_patch(rect)
    axes[1].text(x + w/2, y - 15, f'{w}×{h} px',
                 ha='center', fontsize=9, color=col, fontweight='bold')

# Panel 3: Camera model diagram
ax3 = axes[2]
ax3.set_xlim(-1, 12); ax3.set_ylim(-4, 4); ax3.set_aspect('equal')  # noqa: E702
ax3.axis('off')
ax3.set_title("Pinhole camera model\n(not to scale)", fontsize=10)

# Draw optical axis
ax3.annotate('', xy=(11, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='#888780', lw=1.5))
ax3.text(11.1, 0, 'Z', fontsize=10, va='center', color='#888780')

# Lens (at z=0)
ax3.plot([0, 0], [-2, 2], color='#185FA5', lw=3)
ax3.text(0, 2.3, 'Lens\n(f=8mm)', fontsize=8, ha='center', color='#185FA5')

# Object plane (at z=Z=720mm → scaled to x=8)
OBJ = 8
ax3.plot([OBJ, OBJ], [-3, 3], color='#0F6E56', lw=2, linestyle='--')
ax3.text(OBJ, 3.3, 'Object\n(Z=720mm)', fontsize=8, ha='center', color='#0F6E56')

# Object (earring at object plane)
obj_h = 2.0
ax3.annotate('', xy=(OBJ, obj_h), xytext=(OBJ, 0),
             arrowprops=dict(arrowstyle='->', color='#0F6E56', lw=2))
ax3.text(OBJ + 0.2, obj_h/2, 'X\n(real)', fontsize=8, color='#0F6E56')

# Image plane (at z=-f → scaled to x=-0.09)
IMG = -0.7
ax3.plot([IMG, IMG], [-1.5, 1.5], color='#A32D2D', lw=2)
ax3.text(IMG, 1.8, 'Sensor\n(f=8mm)', fontsize=8, ha='center', color='#A32D2D')

# Image of object
img_h = -obj_h * abs(IMG) / OBJ
ax3.annotate('', xy=(IMG, img_h), xytext=(IMG, 0),
             arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=1.5))
ax3.text(IMG - 0.3, img_h/2, "x'\n(px)", fontsize=8, ha='right', color='#A32D2D')

# Ray from object tip through lens to image
ax3.plot([OBJ, IMG], [obj_h, img_h], color='#BA7517', lw=1.2, linestyle=':')
ax3.plot([OBJ, 0], [obj_h, 0], color='#BA7517', lw=1.2, linestyle=':')

# Formula
ax3.text(3.5, -2.8,
         r'$X = x_{px} \times p \times Z / f$' + '\n'
         r'$= x_{px} \times 2.2\mu m \times 720mm / 8mm$' + '\n'
         r'$= x_{px} \times 198\ \mu m/px$',
         fontsize=8.5, ha='center', va='center',
         bbox=dict(facecolor='#F1EFE8', edgecolor='#D3D1C7', alpha=0.9, pad=5))

plt.tight_layout()
plt.savefig('Q2_Earring_Size/outputs/q2_output.png', dpi=150, bbox_inches='tight')
print("\nSaved → /tmp/q2_output.png")
plt.show()
