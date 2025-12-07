import matplotlib.pyplot as plt
from numpy import * 

# === 解決中文顯示問題的程式碼 ===
# 選擇一個支援中文的字體
# (Windows/macOS/Linux 可能需要替換成系統中實際存在的字體名稱)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS'] # 將常用中文字體加入
plt.rcParams['axes.unicode_minus'] = False # 解決負號'-'顯示為方塊的問題
# ==================================

N = 100
R, lamda = 1.0, 500E-9
d = 100E-6
dx, dy = d/N, d/N

# circle: 0
# cover by little circle: 1 (圓環)
# 1's compliment: 2 (小圓孔)

# 1.創建 3x2 的 Matplotlib 畫布和子圖陣列
fig, axes = plt.subplots(3, 2, figsize=(10, 15)) 
plt.rcParams['font.size'] = 10

titles_I = [
    "Stage 0: 圓孔繞射 (強度 $I$)",
    "Stage 1: 圓環繞射 (強度 $I$)",
    "Stage 2: 小圓孔繞射 (強度 $I$)"
]
titles_E = [
    "Stage 0: 圓孔繞射 (電場絕對值 $|E|$)",
    "Stage 1: 圓環繞射 (電場絕對值 $|E|$)",
    "Stage 2: 小圓孔繞射 (電場絕對值 $|E|$)"
]

for stage in range(3):
    side = linspace(-0.01*pi, 0.01*pi, N)
    x,y = meshgrid(side,side)

    # calculate the electric field of diffraction of the aperture
    k = 2*pi/lamda
    A = zeros((N,N))
    for ix in range(N):
        for iy in range(N):
            # 檢查是否在光圈總範圍內 (直徑 N)
            is_in_big_circle = ((ix-N/2)**2 + (iy-N/2)**2) <= (N/2)**2 
            
            # 決定當前點 (ix, iy) 是否對繞射場有貢獻 (即是否在光圈內)
            should_add = False
            if stage == 0 and is_in_big_circle:
                # 圓孔
                should_add = True
            elif stage == 1 and is_in_big_circle and ((ix-N/2)**2 + (iy-N/2)**2) >= (N/4)**2:
                # 圓環 (在大圓內，但不在小圓內)
                should_add = True
            elif stage == 2 and ((ix-N/2)**2 + (iy-N/2)**2) < (N/4)**2:
                # 小圓孔 (在小圓內)
                should_add = True
            
            if should_add:
                 A += cos((k*x/R) * (ix*dx - d/2) + (k*y/R) * (iy*dy - d/2)) / R

    E_field = A

    # 打印特定點的數值
    print("E at 41,54 in stage", stage, "=", E_field[41][54])
    print("intensity at 41,54 in stage", stage, "=", abs(E_field[41][54])**2)
    print()

    # === 左列：強度 I = |E|^2 ===
    Inte_I = abs(E_field) ** 2
    maxI_I = amax(Inte_I)
    
    # Matplotlib 繪圖
    ax_I = axes[stage, 0] # 選擇第 stage 行, 第 0 列的子圖
    ax_I.imshow(Inte_I / maxI_I, 
                cmap='gray', 
                extent=[side.min(), side.max(), side.min(), side.max()])
    ax_I.set_title(titles_I[stage])
    ax_I.set_xlabel(r'$\theta_x$ (rad)') # 標註座標軸為角度 (rad)
    ax_I.set_ylabel(r'$\theta_y$ (rad)')

    # === 右列：電場絕對值 |E| ===
    Inte_E = abs(E_field)
    maxI_E = amax(Inte_E)
    
    # Matplotlib 繪圖
    ax_E = axes[stage, 1]
    ax_E.imshow(Inte_E / maxI_E, 
                cmap='gray', 
                extent=[side.min(), side.max(), side.min(), side.max()])
    ax_E.set_title(titles_E[stage])
    ax_E.set_xlabel(r'$\theta_x$ (rad)')
    ax_E.set_ylabel(r'$\theta_y$ (rad)')
    
plt.tight_layout()
plt.show()