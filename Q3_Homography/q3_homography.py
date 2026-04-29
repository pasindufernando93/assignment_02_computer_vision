import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

c1 = cv2.imread('assets/c1.jpg', cv2.IMREAD_REDUCED_COLOR_4)
c2 = cv2.imread('assets/c2.jpg', cv2.IMREAD_REDUCED_COLOR_4)

c1_rgb = cv2.cvtColor(c1, cv2.COLOR_BGR2RGB)
c2_rgb = cv2.cvtColor(c2, cv2.COLOR_BGR2RGB)

print(f"c1 size: {c1.shape[1]} × {c1.shape[0]} px")
print(f"c2 size: {c2.shape[1]} × {c2.shape[0]} px")

H_OUT, W_OUT = c2.shape[:2]   # warp target size = c2's dimensions



# (a) Manual homography — hardcoded correspondence points
pts1 = np.float32([
    [188.0, 416.3],
    [251.4, 629.7],
    [338.4, 624.3],
    [162.9, 366.3],
    [399.3, 484.1],
    [127.4, 470.0],
    [353.3, 429.9],
    [375.4, 129.7],
])

pts2 = np.float32([
    [173.0, 355.1],
    [178.0, 577.7],
    [263.2, 595.4],
    [162.0, 300.3],
    [358.7, 476.2],
    [100.4, 390.9],
    [328.9, 411.9],
    [429.4, 128.1],
])

# Compute homography using DLT
H_manual, mask_manual = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

print("\n" + "="*55)
print("(a) Manual homography matrix H:")
print("="*55)
print(H_manual)
print(f"\nRANSAC inliers: {mask_manual.sum()}/{len(pts1)}")

# Warp c1 to the perspective of c2
warped_manual = cv2.warpPerspective(c1, H_manual, (W_OUT, H_OUT))
warped_manual_rgb = cv2.cvtColor(warped_manual, cv2.COLOR_BGR2RGB)

# (b) Difference image — warped manual vs c2
diff_manual = cv2.absdiff(warped_manual, c2)

# Create mask: only where warped image has content
warp_mask = (warped_manual.sum(axis=2) > 0).astype(np.uint8) * 255

# Amplify differences for visibility (scale to use full range)
diff_manual_amp = diff_manual.copy()
diff_manual_amp = cv2.convertScaleAbs(diff_manual_amp, alpha=3.0)
diff_manual_rgb = cv2.cvtColor(diff_manual, cv2.COLOR_BGR2RGB)
diff_manual_amp_rgb = cv2.cvtColor(diff_manual_amp, cv2.COLOR_BGR2RGB)

# Threshold: only keep significant differences (>30 per channel)
diff_gray_manual = cv2.cvtColor(diff_manual, cv2.COLOR_BGR2GRAY)
_, diff_thresh_manual = cv2.threshold(diff_gray_manual, 30, 255, cv2.THRESH_BINARY)

print("\n" + "="*55)
print("(b) Difference image (manual homography)")
print("="*55)
print(f"  Mean abs difference : {diff_manual.mean():.3f}")
print(f"  Pixels > threshold  : {(diff_gray_manual > 30).sum():,} "
      f"({100*(diff_gray_manual>30).mean():.1f}%)")


# (c) SIFT keypoint detection and matching
g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=3000)
kp1, des1 = sift.detectAndCompute(g1, None)
kp2, des2 = sift.detectAndCompute(g2, None)

print("\n" + "="*55)
print("(c) SIFT keypoint detection")
print("="*55)
print(f"  c1 keypoints: {len(kp1)}")
print(f"  c2 keypoints: {len(kp2)}")

# Brute-force matching with ratio test 
bf   = cv2.BFMatcher(cv2.NORM_L2)
raw  = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in raw if m.distance < 0.75 * n.distance]

print(f"\n  Raw matches         : {len(raw)}")
print(f"  After ratio test    : {len(good)}  (ratio=0.75)")

# Draw matches (limit to 60 for clarity)
match_img = cv2.drawMatches(
    c1, kp1, c2, kp2,
    good[:60], None,
    matchColor=(0, 200, 100),
    singlePointColor=(100, 100, 255),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
match_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)


# (d) Auto homography from SIFT matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H_auto, mask_auto = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

ransac_inliers = mask_auto.ravel().astype(bool)
print("\n" + "="*55)
print("(d) Auto homography (SIFT + RANSAC)")
print("="*55)
print(f"  RANSAC inliers : {ransac_inliers.sum()} / {len(good)} good matches")
print(f"  Inlier ratio   : {100*ransac_inliers.mean():.1f}%")
print("\n  Homography matrix H_auto:")
print(H_auto)

# Warp c1 to c2's perspective using auto homography
warped_auto = cv2.warpPerspective(c1, H_auto, (W_OUT, H_OUT))
warped_auto_rgb = cv2.cvtColor(warped_auto, cv2.COLOR_BGR2RGB)

# Difference image
diff_auto = cv2.absdiff(warped_auto, c2)
diff_auto_amp = cv2.convertScaleAbs(diff_auto, alpha=3.0)
diff_auto_rgb     = cv2.cvtColor(diff_auto, cv2.COLOR_BGR2RGB)
diff_auto_amp_rgb = cv2.cvtColor(diff_auto_amp, cv2.COLOR_BGR2RGB)
diff_gray_auto    = cv2.cvtColor(diff_auto, cv2.COLOR_BGR2GRAY)

print(f"\n  Mean abs difference (auto)   : {diff_auto.mean():.3f}")
print(f"  Mean abs difference (manual) : {diff_manual.mean():.3f}")
print(f"  Auto improvement             : "
      f"{100*(diff_manual.mean()-diff_auto.mean())/diff_manual.mean():.1f}%")


# FIGURE 1 — Main results grid
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Q3 — Homography, Warping & Circuit Board Comparison",
             fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

panels = [
    # Row 0
    (c1_rgb,               gs[0,0], 'c1 — original'),
    (c2_rgb,               gs[0,1], 'c2 — target'),
    (warped_manual_rgb,    gs[0,2], '(a) Warped c1 (manual pts)'),
    (diff_manual_amp_rgb,  gs[0,3], '(b) Diff ×3 (manual)\nmean={:.2f}'.format(diff_manual.mean())),
    # Row 1
    (warped_auto_rgb,      gs[1,0], '(d) Warped c1 (SIFT auto)'),
    (diff_auto_amp_rgb,    gs[1,1], '(d) Diff ×3 (auto)\nmean={:.2f}'.format(diff_auto.mean())),
    # Comparison side-by-side
    (diff_manual_amp_rgb,  gs[1,2], 'Manual diff (×3)'),
    (diff_auto_amp_rgb,    gs[1,3], 'Auto diff (×3)'),
]

for (im, pos, title) in panels:
    ax = fig.add_subplot(pos)
    ax.imshow(im); ax.set_title(title, fontsize=8.5); ax.axis('off')

# Annotate manual correspondence points on row 2
ax_pts1 = fig.add_subplot(gs[2, 0])
ax_pts1.imshow(c1_rgb)
ax_pts1.scatter(pts1[:,0], pts1[:,1], c='#E24B4A', s=60, zorder=5,
                edgecolors='white', linewidths=1)
for i, (x, y) in enumerate(pts1):
    ax_pts1.text(x+5, y-8, str(i+1), fontsize=7, color='#E24B4A', fontweight='bold')
ax_pts1.set_title('(a) Manual pts — c1', fontsize=8.5); ax_pts1.axis('off')

ax_pts2 = fig.add_subplot(gs[2, 1])
ax_pts2.imshow(c2_rgb)
ax_pts2.scatter(pts2[:,0], pts2[:,1], c='#185FA5', s=60, zorder=5,
                edgecolors='white', linewidths=1)
for i, (x, y) in enumerate(pts2):
    ax_pts2.text(x+5, y-8, str(i+1), fontsize=7, color='#185FA5', fontweight='bold')
ax_pts2.set_title('(a) Manual pts — c2', fontsize=8.5); ax_pts2.axis('off')

# Thresholded diff comparison
ax_t1 = fig.add_subplot(gs[2, 2])
thresh_m = np.zeros_like(c2_rgb)
thresh_m[:,:,0] = diff_thresh_manual
ax_t1.imshow(thresh_m); ax_t1.set_title('Manual diff thresholded\n(>30 intensity)', fontsize=8.5)
ax_t1.axis('off')

_, diff_thresh_auto = cv2.threshold(diff_gray_auto, 30, 255, cv2.THRESH_BINARY)
ax_t2 = fig.add_subplot(gs[2, 3])
thresh_a = np.zeros_like(c2_rgb)
thresh_a[:,:,0] = diff_thresh_auto
ax_t2.imshow(thresh_a); ax_t2.set_title('Auto diff thresholded\n(>30 intensity)', fontsize=8.5)
ax_t2.axis('off')

plt.savefig('Q3_Homography/outputs/q3_main.png', dpi=140, bbox_inches='tight')
print("\nSaved → q3_main.png")
plt.show()


# FIGURE 2 — SIFT matches
fig2, ax = plt.subplots(1, 1, figsize=(16, 6))
ax.imshow(match_rgb)
ax.set_title(f'(c) SIFT Matches — {len(good)} good matches after ratio test '
             f'(showing 60, {ransac_inliers.sum()} RANSAC inliers)',
             fontsize=11, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('Q3_Homography/outputs/q3_sift_matches.png', dpi=140, bbox_inches='tight')
print("Saved → q3_sift_matches.png")
plt.show()


