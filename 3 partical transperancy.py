from vpython import *
import numpy as np
from numpy.fft import fft2

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
    m = np.max(I)
    if m > 0:
        I = I / m      # normalize to [0,1]，避免除以 0
    return I

# ========== 中央裁切，放大看 pattern ==========
def crop_center(I, M):
    cx, cy = I.shape[0]//2, I.shape[1]//2
    h = M//2
    return I[cx-h:cx+h, cy-h:cy+h]

# ========== display helper (with center & gamma) ==========
def draw_intensity(scene, I, title):
    # 使用 HTML 放大 title 字體 + 粗體
    scene.title = "<b><font size=3>" + title + "</font></b>"
    scene.lights = []
    scene.ambient = color.gray(0.9)

    # gamma 調整，讓暗暗的 ring / side lobe 也看得見
    gamma = 0.25
    I_disp = I**gamma

    M = I.shape[0]
    scene.center = vector(M/2, M/2, 0)

    for i in range(M):
        for j in range(M):
            val = float(I_disp[i, j])
            box(canvas=scene,
                pos=vector(i, j, 0),
                length=1, height=1, width=0.1,
                color=vector(val, val, val))

# ========== 共同的 open aperture pattern ==========
I_open   = fraunhofer(A_open)
I_open_c = crop_center(I_open, CROP)

# ========== 定義三個 case（用「強度透過率」T） ==========
# T_A, T_Ap 是強度透過率；實際乘在 mask 上的是振幅 sqrt(T)
cases = [
    ("Case 1: Limiting case (T_A = 1.0, T_A' = 0.0)", 1.0, 0.0),
    ("Case 2: Partial transparency (T_A = 0.8, T_A' = 0.2)", 0.8, 0.2),
    ("Case 3: Partial transparency (T_A = 0.6, T_A' = 0.4)", 0.6, 0.4),
]

# ========== 逐個 case 模擬並畫圖 ==========
for idx, (case_name, T_A, T_Ap) in enumerate(cases):
    # 振幅透過率（因為 I ∝ |E|^2）
    tA  = np.sqrt(T_A)
    tAp = np.sqrt(T_Ap)

    # 對應到實際的「振幅 mask」
    maskA      = tA  * A
    maskA_comp = tAp * A_comp

    # Fraunhofer patterns
    I_A      = fraunhofer(maskA)
    I_A_comp = fraunhofer(maskA_comp)

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

    # ---------- VPython 畫 4 個畫面（每個 case 一列） ----------
    y_offset = idx * 430   # 每個 case 往下排一列

    scene1 = canvas(width=400, height=400, align='left',
                    x=0,   y=y_offset)
    scene2 = canvas(width=400, height=400, align='left',
                    x=410, y=y_offset)
    scene3 = canvas(width=400, height=400, align='left',
                    x=820, y=y_offset)
    scene4 = canvas(width=400, height=400, align='left',
                    x=1230, y=y_offset)

    draw_intensity(scene1, I_A_c,
                   f"[{case_name}] Pattern through mask A")
    draw_intensity(scene2, I_A_comp_c,
                   f"[{case_name}] Pattern through mask A'")
    draw_intensity(scene3, I_sum_c,
                   f"[{case_name}] Pattern through A + A' (field sum)")
    draw_intensity(scene4, I_open_c,
                   f"[{case_name}] Pattern through open aperture")

    # 只有在 T_A = 1, T_A' = 0 以外的 case，打印提示
    if T_A == 1.0 and T_Ap == 0.0:
        print(case_name, "(A' 完全不透光，此時只剩下 A 的 pattern，sum = A)")
    else:
        print(case_name, "simulated with partial transparency.")

# b12901058 add
input("Press Enter to close the VPython window...")