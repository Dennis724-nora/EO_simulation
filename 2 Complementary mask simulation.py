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
    I /= np.max(I)    # normalize to [0,1]
    return I

I_A      = fraunhofer(A)
I_A_comp = fraunhofer(A_comp)
I_open   = fraunhofer(A_open)

# field sum: E_A + E_A'
E_A      = myfftshift(fft2(A))
E_A_comp = myfftshift(fft2(A_comp))
E_sum    = E_A + E_A_comp
I_sum    = np.abs(E_sum)**2
I_sum   /= np.max(I_sum)

# ========== 中央裁切，放大看 pattern ==========
def crop_center(I, M):
    cx, cy = I.shape[0]//2, I.shape[1]//2
    h = M//2
    return I[cx-h:cx+h, cy-h:cy+h]

I_A_c      = crop_center(I_A,      CROP)
I_A_comp_c = crop_center(I_A_comp, CROP)
I_open_c   = crop_center(I_open,   CROP)
I_sum_c    = crop_center(I_sum,    CROP)

# ========== display helper (with center & gamma) ==========
def draw_intensity(scene, I, title):
    # 使用 HTML 放大 title 字體 + 粗體
    scene.title = "<b><font size=6>" + title + "</font></b>"
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

# ========== 4 個畫面 (水平排列成 2×2) ==========
scene1 = canvas(width=400, height=400)
scene2 = canvas(width=400, height=400)
scene3 = canvas(width=400, height=400)
scene4 = canvas(width=400, height=400)

draw_intensity(scene1, I_A_c,
               "Pattern through mask A (Fraunhofer Airy disk, cropped center)")
draw_intensity(scene2, I_A_comp_c,
               "Pattern through mask A' (complement aperture, cropped center)")
draw_intensity(scene3, I_sum_c,
               "Pattern through both masks A and A' simultaneously (cropped)")
draw_intensity(scene4, I_open_c,
               "Pattern through open aperture (square, cropped center)")


# ========== Babinet check ==========
E_open = myfftshift(fft2(A_open))
err = np.max(np.abs(E_A + E_A_comp - E_open)) / np.max(np.abs(E_open))
print("Babinet relative error =", err)

# b12901058 add
input("Press Enter to close the VPython window...")