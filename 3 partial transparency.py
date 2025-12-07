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
CROP = 80        # 中央要看的區域大小 (CROP x CROP)

# ========== build aperture masks ==========
x = np.arange(N)
y = np.arange(N)
xx, yy = np.meshgrid(x, y)
cx, cy = N//2, N//2

# Mask A: transparent circular aperture (binary)
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
    # m = np.max(I)
    # if m > 0:
    #     I = I / m      # normalize to [0,1]，避免除以 0
    return I

# ========== 中央裁切，放大看 pattern ==========
def crop_center(I, M):
    cx, cy = I.shape[0]//2, I.shape[1]//2
    h = M//2
    return I[cx-h:cx+h, cy-h:cy+h]

# ========== 共同的 open aperture pattern ==========
I_open   = fraunhofer(A_open)
I_open_c = crop_center(I_open, CROP)

# ========== 定義三個 case（用「強度透過率」T） ==========
# T_A, T_Ap 是強度透過率；實際乘在 mask 上的是振幅 sqrt(T)
cases = [
    ("Case 1: Limiting case (T_A = 1.0, T_A' = 0.0)", 1.0, 0.0),
    ("Case 2: Partial transparency (T_A = 0.64, T_A' = 0.04)", 0.64, 0.04),
    ("Case 3: Partial transparency (T_A = 0.36, T_A' = 0.16)", 0.36, 0.16),
]

# ========== Matplotlib 初始化 (3行 x 4列) ==========
# figsize (16, 12) 確保有足夠的空間放置 12 張圖
fig, axes = plt.subplots(len(cases), 4, figsize=(16, 12)) 
plt.rcParams['font.size'] = 10

# Matplotlib 輔助函數：繪製裁切後的強度圖案
def plot_intensity(ax, I_cropped, title):
    # 使用 gamma 調整
    gamma = 0.25
    I_disp = I_cropped**gamma

    # sqrt for better observation
    I_disp = np.sqrt(I_disp)
    
    # 使用 imshow 繪製灰階圖案
    im = ax.imshow(I_disp, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    
    # 隱藏刻度，讓圖形看起來更像 VPython 的箱子圖
    ax.set_xticks([])
    ax.set_yticks([])

# ========== 逐個 case 模擬並畫圖 ==========
for idx, (case_name, T_A, T_Ap) in enumerate(cases):
    # 振幅透過率
    tA = np.sqrt(T_A)
    tAp = np.sqrt(T_Ap)

    # 對應到實際的「振幅 mask」
    maskA      = tA  * A + tAp * A_comp
    maskA_comp = tA * A_comp + tAp * A

    # Fraunhofer patterns
    I_A      = fraunhofer(maskA)
    I_A_comp = fraunhofer(maskA_comp)
    I_max = max(np.max(I_A), np.max(I_A_comp))
    I_A /= I_max
    I_A_comp /= I_max

    # 場的疊加 (E_A + E_A')
    E_A      = myfftshift(fft2(maskA))
    E_A_comp = myfftshift(fft2(maskA_comp))
    E_sum    = E_A + E_A_comp
    I_sum    = np.abs(E_sum)**2
    m_sum = np.max(I_sum)
    if m_sum > 0:
        I_sum = I_sum / m_sum

    # 中央裁切
    I_A_c      = crop_center(I_A,      CROP)
    I_A_comp_c = crop_center(I_A_comp, CROP)
    I_sum_c    = crop_center(I_sum,    CROP)

    # ---------- Matplotlib 繪製 4 個畫面（第 idx 行） ----------
    
    # Column 0: Mask A
    plot_intensity(axes[idx, 0], I_A_c,
                   f"[{case_name}]\nMask A (sqrt I)")
    
    # Column 1: Mask A'
    plot_intensity(axes[idx, 1], I_A_comp_c,
                   f"[{case_name}]\nMask A' (sqrt I)")
    
    # Column 2: Field Sum (A + A')
    plot_intensity(axes[idx, 2], I_sum_c,
                   f"[{case_name}]\nField Sum (|E_A + E_A'|^2) (sqrt I)")
    
    # Column 3: Open Aperture (所有 case 相同，用於比較 Babinet's principle)
    plot_intensity(axes[idx, 3], I_open_c,
                   f"[{case_name}]\nOpen Aperture (Reference) (sqrt I)")

    print(case_name, "simulated with partial transparency.")
    # ========== Babinet check ==========
    E_open = myfftshift(fft2(A_open))
    err = np.sum(np.abs(E_A + E_A_comp - E_open)) / np.sum(np.abs(E_open))
    print("Babinet relative error =", err)

plt.tight_layout()

plt.savefig(f'{N}_point_partial_transparency.png', dpi=300, bbox_inches='tight')
plt.show()