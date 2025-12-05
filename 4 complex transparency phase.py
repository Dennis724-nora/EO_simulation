import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2

# === 解決中文顯示問題的程式碼 (保持不變) ===
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
# ==================================

# ---------- small helper: 2D fftshift (保持不變) ----------
def myfftshift(X):
    # 將零頻率分量移到中心
    return np.roll(np.roll(X, X.shape[0]//2, axis=0),
                   X.shape[1]//2, axis=1)

# ========== parameters (使用你原始的參數名稱) ==========
N = 200        # FFT grid size
D = 80         # circular aperture diameter (in pixels)
CROP = 80      # 中央要看的區域大小 (CROP x CROP)
PHI_0 = np.pi / 2 # 相位屏引入的相移 (90 度)
C = 1.0        # Constant uniform field C (設為 1 + 0i)

# ========== build aperture masks (使用你原始的變數名稱 A, A_comp) ==========
x = np.arange(N)
y = np.arange(N)
xx, yy = np.meshgrid(x, y)
cx, cy = N//2, N//2

# Mask A_binary (A): transparent circular aperture (1 inside, 0 outside)
A_binary = ((xx - cx)**2 + (yy - cy)**2 <= (D/2)**2).astype(float)

# Mask A_comp_binary (A_comp): complement of A_binary (0 inside, 1 outside)
A_comp_binary = 1.0 - A_binary

# Open aperture (full square, C=1) - 必須是複數型態
A_open = np.ones((N, N), dtype=complex)

# ========== 定義複數傳輸函數 (與 Case 4 邏輯一致) ==========

# 屏 A: 圓孔內相移 PHI_0 (e^(i PHI_0))，圓孔外 1 (無相移)
# 這裡的 A(x) 是純相位屏，振幅 a(x)=1 處處透光
A_complex = A_binary * np.exp(1j * PHI_0) + A_comp_binary * 1.0
maskA = A_complex # 使用你原始的 maskA 變數名稱

# 屏 B (互補屏): B(x) = C - A(x) = 1 - A_complex
# 這裡的 B(x) 是複雜的振幅-相位屏，圓孔外的傳輸率為 0 (不透光)
maskB = C - A_complex # 使用一個新的 maskB 變數名稱，以區分你原程式碼的 maskA_comp

# ========== Fraunhofer diffraction via FFT (修改為返回 E 和 I) ==========
def fraunhofer(mask):
    # 進行 FFT，結果 E 是複數電場
    E = fft2(mask)
    E = myfftshift(E)
    I = np.abs(E)**2 # 強度 I = |E|^2
    return E, I

# ========== 中央裁切 (保持不變) ==========
def crop_center(I, M):
    cx, cy = I.shape[0]//2, I.shape[1]//2
    h = M//2
    return I[cx-h:cx+h, cy-h:cy+h]

# ========== Matplotlib 輔助函數：繪製裁切後的強度圖案 (保持不變) ==========
def plot_intensity(ax, I_cropped, title):
    gamma = 0.25
    I_disp = I_cropped**gamma
    
    im = ax.imshow(I_disp, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    
    ax.set_xticks([])
    ax.set_yticks([])

# ========== 模擬與繪圖 ==========

# 1. 計算三個場的繞射電場 E 和強度 I
E_A, I_A = fraunhofer(maskA)
E_B, I_B = fraunhofer(maskB)
E_open, I_open = fraunhofer(A_open)

# 2. 廣義 Babinet 檢查: E_sum = E_A + E_B
E_sum = E_A + E_B
I_sum = np.abs(E_sum)**2

# 3. 歸一化
# Babinet 原理關注中央點以外的區域，但整體歸一化有助於圖像對比。
# 以全透光光圈 I_open 的最大值來歸一化 Field Sum，確保其中心亮點為 1
I_open_max = np.max(I_open)
if I_open_max > 0:
    I_sum_norm = I_sum / I_open_max
else:
    I_sum_norm = I_sum

# 對 A 和 B 的圖案歸一化，以確保它們的非中心區域可以清晰比較
I_max_A_B = max(np.max(I_A), np.max(I_B))
if I_max_A_B > 0:
    I_A_norm = I_A / I_max_A_B
    I_B_norm = I_B / I_max_A_B
else:
    I_A_norm = I_A
    I_B_norm = I_B


# 4. 中央裁切
I_A_c = crop_center(I_A_norm, CROP)
I_B_c = crop_center(I_B_norm, CROP)
I_sum_c = crop_center(I_sum_norm, CROP)
I_open_c = crop_center(I_open, CROP) # 參考圖不需要再次歸一化

# 5. 繪圖 (1 行 x 4 列)
fig, axes = plt.subplots(1, 4, figsize=(16, 4)) 
case_name = f"Complex Babinet ($\phi_0 = \pi/2$)"

plot_intensity(axes[0], I_A_c, f"[{case_name}]\n|E_A|^2 (Phase Screen A)")
plot_intensity(axes[1], I_B_c, f"[{case_name}]\n|E_B|^2 (Complement B = C - A)")
plot_intensity(axes[2], I_sum_c, f"[{case_name}]\nField Sum (|E_A + E_B|^2)")
plot_intensity(axes[3], I_open_c, f"[{case_name}]\nOpen Aperture (Reference)")

# 調整子圖間距
plt.tight_layout()

# ========== Babinet check ==========
E_open = myfftshift(fft2(A_open))
err = np.sum(np.abs(E_A + E_B - E_open)) / np.sum(np.abs(E_open))

print(f"--- 複數 Babinet 模擬結果 ---")
print(f"屏 A (Phase Screen) 的最大強度: {np.max(I_A):.2f}")
print(f"屏 B (Complement) 的最大強度: {np.max(I_B):.2f}")
print(f"電場疊加 |E_A + E_B|^2 的最大強度: {np.max(I_sum):.2f}")
print(f"全透光光圈 |E_open|^2 的最大強度: {np.max(I_open):.2f}")
print(f"Babinet 相對誤差 max(|E_A + E_B - E_open|) / max(|E_open|) = {err:.4e}")


# 顯示 Matplotlib 圖形
# 
plt.savefig('complex_babinet_result.png', dpi=300, bbox_inches='tight')
plt.show()