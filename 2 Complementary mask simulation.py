import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2

# === 解決中文顯示問題的程式碼 ===
# 選擇一個支援中文的字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
# ==================================

# ---------- small helper: 2D fftshift ----------
def myfftshift(X):
    return np.roll(np.roll(X, X.shape[0]//2, axis=0),
                   X.shape[1]//2, axis=1)

# ========== parameters ==========
N = 200          # FFT grid size
D = 80           # circular aperture diameter (in pixels)
CROP = 80        # 中央要看的區域大小 (CROP x CROP)，可以自己調

# ========== build aperture masks ==========
x = np.arange(N)
y = np.arange(N)
xx, yy = np.meshgrid(x, y)
cx, cy = N//2, N//2

# Mask A: transparent circular aperture
A = ((xx - cx)**2 + (yy - cy)**2 <= (D/2)**2).astype(float)

# Mask A': complement of A (整個方形都可以透，只是圓孔挖掉)
A_comp = 1.0 - A

# Open aperture (full square)
A_open = np.ones((N, N))

# ========== Fraunhofer diffraction via FFT ==========
def fraunhofer(mask):
    E = fft2(mask)
    E = myfftshift(E)
    I = np.abs(E)**2
    # I /= np.max(I)    # normalize to [0,1]
    return I

I_A      = fraunhofer(A)
I_A_comp = fraunhofer(A_comp)
I_max = max(np.max(I_A), np.max(I_A_comp))
I_A /= I_max
I_A_comp /= I_max
I_open   = fraunhofer(A_open)

# field sum: E_A + E_A'
E_A      = myfftshift(fft2(A))
E_A_comp = myfftshift(fft2(A_comp))
E_sum    = E_A + E_A_comp
I_sum    = np.abs(E_sum)**2
# I_sum   /= np.max(I_sum)

# ========== 中央裁切，放大看 pattern ==========
def crop_center(I, M):
    cx, cy = I.shape[0]//2, I.shape[1]//2
    h = M//2
    return I[cx-h:cx+h, cy-h:cy+h]

I_A_c      = crop_center(I_A,      CROP)
I_A_comp_c = crop_center(I_A_comp, CROP)
I_open_c   = crop_center(I_open,   CROP)
I_sum_c    = crop_center(I_sum,    CROP)

# ========== Matplotlib 繪圖 (2x2) ==========
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 繪圖參數：使用 gamma 調整，讓暗處可見
gamma = 0.25 

# === 繪製函數 ===
def plot_pattern(ax, I_cropped, title):
    # Gamma 調整
    I_disp = I_cropped**gamma
    
    ax.imshow(I_disp, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([]) # 隱藏刻度

# Row 0, Col 0: 圓孔 A
plot_pattern(axes[0, 0], I_A_c, "圓孔 A (Aperture A) 繞射強度 $I_A$")

# Row 0, Col 1: 圓板 A'
plot_pattern(axes[0, 1], I_A_comp_c, "圓板 A' (Complement A') 繞射強度 $I_{A'}$")

# Row 1, Col 0: 場疊加 (E_A + E_A')
plot_pattern(axes[1, 0], I_sum_c, "場疊加 $|E_A + E_{A'}|^2$")

# Row 1, Col 1: 全透光
plot_pattern(axes[1, 1], I_open_c, "全透光 (Open Aperture) 繞射強度 $I_{Open}$")


# 調整子圖間距
plt.tight_layout()

# 顯示圖形
plt.savefig(f'{N}_point_binary_transparency.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== Babinet check ==========
E_open = myfftshift(fft2(A_open))
err = np.sum(np.abs(E_A + E_A_comp - E_open)) / np.sum(np.abs(E_open))
print("Babinet relative error =", err)