import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# Load data 
D = np.genfromtxt("assets/lines.csv", delimiter=',', skip_header=1)

X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all  = X_cols.flatten()   # 300 points
Y_all  = Y_cols.flatten()

print(f"Total points: {len(X_all)}")
print(f"X range: [{X_all.min():.3f}, {X_all.max():.3f}]")
print(f"Y range: [{Y_all.min():.3f}, {Y_all.max():.3f}]")


# Total Least Squares (TLS) line fitting
def fit_line_tls(x, y):
    x, y   = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()


    A = np.column_stack([x - mx, y - my])  
    _, _, Vt = np.linalg.svd(A)
    a, b     = Vt[-1]                       

    c = -(a * mx + b * my)

    norm = np.sqrt(a**2 + b**2)
    return a / norm, b / norm, c / norm


def point_to_line_dist(x, y, a, b, c):
    return np.abs(a * x + b * y + c)  


def line_y(x_range, a, b, c):
    return -(a * x_range + c) / b


# (a) TLS on Line 1 data only (x1, y1 columns)
x1 = D[:, 0]
y1 = D[:, 3]

a1, b1, c1 = fit_line_tls(x1, y1)

print("\n" + "="*55)
print("(a) TLS fit on Line 1 data (x1, y1)")
print("="*55)
print(f"  Line equation: {a1:.6f}·x + {b1:.6f}·y + {c1:.6f} = 0")
print(f"  Slope (dy/dx): {-a1/b1:.6f}")
print(f"  Intercept:     {-c1/b1:.6f}")
print(f"  Normal vector: [{a1:.6f}, {b1:.6f}]")

# Residuals
dists1 = point_to_line_dist(x1, y1, a1, b1, c1)
print(f"  Mean perp distance: {dists1.mean():.6f}")
print(f"  Max  perp distance: {dists1.max():.6f}")


# (b) RANSAC to find 3 lines from all 300 mixed points
def ransac_line(x, y, n_iters=2000, threshold=0.3, min_inliers=20):
    
    best_inliers = None
    best_count   = 0
    n = len(x)

    for _ in range(n_iters):
        # Sample 2 distinct points
        idx = np.random.choice(n, 2, replace=False)
        xs, ys = x[idx], y[idx]

        # Fit TLS to these 2 points
        a, b, c = fit_line_tls(xs, ys)

        # Count inliers
        dists   = point_to_line_dist(x, y, a, b, c)
        inliers = dists < threshold

        if inliers.sum() > best_count:
            best_count   = inliers.sum()
            best_inliers = inliers

    # Refit using all inliers for best accuracy
    a, b, c = fit_line_tls(x[best_inliers], y[best_inliers])
    return a, b, c, best_inliers


# Run RANSAC 3 times
remaining_mask = np.ones(len(X_all), dtype=bool)
lines   = []
masks   = []

print("\n" + "="*55)
print("(b) RANSAC — finding 3 lines from 300 mixed points")
print("="*55)

for line_num in range(3):
    xr = X_all[remaining_mask]
    yr = Y_all[remaining_mask]

    a, b, c, inlier_local = ransac_line(xr, yr, n_iters=2000, threshold=0.3)

    # Map local inliers back to global indices
    global_indices          = np.where(remaining_mask)[0]
    inlier_global           = np.zeros(len(X_all), dtype=bool)
    inlier_global[global_indices[inlier_local]] = True

    # Remove these inliers from future iterations
    remaining_mask[inlier_global] = False

    lines.append((a, b, c))
    masks.append(inlier_global)

    slope     = -a / b
    intercept = -c / b
    print(f"\n  Line {line_num+1}:")
    print(f"    ax+by+c=0 : {a:.6f}·x + {b:.6f}·y + {c:.6f} = 0")
    print(f"    Slope     : {slope:.6f}")
    print(f"    Intercept : {intercept:.6f}")
    print(f"    Inliers   : {inlier_global.sum()} / {len(X_all)} points")
    dists = point_to_line_dist(X_all[inlier_global],
                                Y_all[inlier_global], a, b, c)
    print(f"    Mean perp dist: {dists.mean():.6f}")


# FIGURE — Results
fig = plt.figure(figsize=(14, 6))
fig.suptitle("Q1 — Line Fitting with Total Least Squares and RANSAC", fontsize=12, fontweight='bold')
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

COLORS = ['#185FA5', '#0F6E56', '#A32D2D']
LIGHT  = ['#E6F1FB', '#E1F5EE', '#FCEBEB']

#(a) TLS on Line 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(x1, y1, c='#185FA5', s=18, alpha=0.7, label='Line 1 data', zorder=3)

x_plot = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 200)
ax1.plot(x_plot, line_y(x_plot, a1, b1, c1),
         color='#A32D2D', lw=2, label=f'TLS fit\ny={-a1/b1:.3f}x + {-c1/b1:.3f}')

# Draw a few perpendicular distance lines to illustrate TLS
for xi, yi in zip(x1[::15], y1[::15]):
    t  = -(a1*xi + b1*yi + c1)
    xf = xi + a1*t
    yf = yi + b1*t
    ax1.plot([xi, xf], [yi, yf], color='#888780', lw=0.8, alpha=0.6)

ax1.set_title('(a) TLS fit — Line 1 data only', fontsize=10)
ax1.set_xlabel('x', fontsize=9)
ax1.set_ylabel('y', fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.text(0.03, 0.97,
         f'a={a1:.4f}\nb={b1:.4f}\nc={c1:.4f}',
         transform=ax1.transAxes, fontsize=8,
         va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='#F1EFE8', alpha=0.8))

# (b) RANSAC — 3 lines
ax2 = fig.add_subplot(gs[0, 1])

for i, ((a, b, c), mask, col, lcol) in enumerate(zip(lines, masks, LIGHT, COLORS)):
    # Inlier points
    ax2.scatter(X_all[mask], Y_all[mask],
                c=col, edgecolors=lcol, linewidths=0.5,
                s=18, alpha=0.85, label=f'Line {i+1} inliers ({mask.sum()}pts)', zorder=3)
    # Fitted line
    xm = X_all[mask]
    x_plot = np.linspace(xm.min() - 0.3, xm.max() + 0.3, 200)
    ax2.plot(x_plot, line_y(x_plot, a, b, c), color=lcol, lw=2)

# Outliers (remaining points not assigned)
outlier_mask = ~(masks[0] | masks[1] | masks[2])
if outlier_mask.sum() > 0:
    ax2.scatter(X_all[outlier_mask], Y_all[outlier_mask],
                c='#888780', s=10, alpha=0.4, label=f'Outliers ({outlier_mask.sum()})', zorder=2)

ax2.set_title('(b) RANSAC — 3 lines from 300 mixed points', fontsize=10)
ax2.set_xlabel('x', fontsize=9)
ax2.set_ylabel('y', fontsize=9)
ax2.legend(fontsize=7.5, loc='upper right')
ax2.grid(True, alpha=0.3)

# Annotate each line with its parameters
for i, (a, b, c) in enumerate(lines):
    ax2.text(0.03, 0.97 - i*0.12,
             f'L{i+1}: y={-a/b:.3f}x+{-c/b:.3f}',
             transform=ax2.transAxes, fontsize=8,
             color=COLORS[i], va='top', fontweight='bold')

plt.savefig('Q1_Line_Fitting/outputs/q1_output.png', dpi=150, bbox_inches='tight')
print("\nSaved → /tmp/q1_output.png")
plt.show()

# Print summary table
print("\n" + "="*55)
print("SUMMARY — All 3 RANSAC Lines")
print("="*55)
print(f"{'Line':<6} {'a':>10} {'b':>10} {'c':>10} {'slope':>10} {'intercept':>10} {'inliers':>8}")
print("-"*64)
for i, ((a,b,c), mask) in enumerate(zip(lines, masks)):
    print(f"  L{i+1}  {a:>10.5f} {b:>10.5f} {c:>10.5f} "
          f"{-a/b:>10.5f} {-c/b:>10.5f} {mask.sum():>8}")
