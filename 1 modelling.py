from vpython import *

####################################
# 幾何位置設定
####################################
x_source = 0.0
x_lens1  = 1.0
x_lens2  = 2.0            # Lens 2 在 masks 前
x_maskA  = 2.8            # Mask A
x_maskB  = 3.6            # Mask A'（跟 A 平行，但遠很多）
x_screen = 5.0

r_beam_small = 0.10       # 初始細光束
r_beam_big   = 0.30       # lens2 之後的大光束

####################################
# 場景設定
####################################
scene = canvas(title="Optical system: source → lens1 → lens2 → masks → screen",
               width=1000, height=500,
               background=color.white)

scene.center = vector((x_source + x_screen)/2, 0, 0)
scene.camera.pos  = vector((x_source + x_screen)/2, 0, 6)
scene.camera.axis = vector(0, 0, -6)

scene.userspin = True
scene.userzoom = True

####################################
# 1. 光源：球 + 初始細 cylinder
####################################
source_ball = sphere(pos=vector(x_source - 0.2, 0, 0),
                     radius=0.12,
                     color=vector(1, 0.8, 0.2),
                     emissive=True)

initial_beam = cylinder(pos=source_ball.pos,
                        axis=vector(x_lens1 - (x_source - 0.2), 0, 0),
                        radius=r_beam_small,
                        color=vector(1, 0.9, 0.4),
                        opacity=0.7)

label(pos=source_ball.pos + vector(0, 0.4, 0),
      text="Source", box=False, height=12)

####################################
# 2. Lens 1：兩個 cone 頂對頂
####################################
lens1_plate = cylinder(pos=vector(x_lens1, 0, 0),
                       axis=vector(0.04, 0, 0),
                       radius=0.9 * r_beam_big,
                       color=vector(0.6, 0.8, 1.0),
                       opacity=0.4)

# 左 cone：頂點在 lens1，往左開，小底
cone_left = cone(pos=vector(x_lens1, 0, 0),
                 axis=vector(-0.7, 0, 0),       # 向左
                 radius=0.7 * r_beam_small,     # 底比較小
                 color=vector(0.8, 0.9, 1.0),
                 opacity=0.3)

# 右 cone：頂點同樣在 lens1，往右開，大底
cone_right = cone(pos=vector(x_lens1, 0, 0),
                  axis=vector(0.9, 0, 0),       # 向右
                  radius=0.95 * r_beam_big,     # 底比較大
                  color=vector(0.8, 0.9, 1.0),
                  opacity=0.3)

label(pos=vector(x_lens1, 1.0, 0),
      text="Lens 1\n(converge + expand)", box=False, height=11)

####################################
# 3. Lens 2：collimate + 放大光束
####################################
lens2_plate = cylinder(pos=vector(x_lens2, 0, 0),
                       axis=vector(0.04, 0, 0),
                       radius=r_beam_big * 1.1,
                       color=vector(0.6, 0.9, 0.9),
                       opacity=0.4)

# Lens2 之後到 Mask A 的平行大光束
collimated_beam = cylinder(pos=vector(x_lens2, 0, 0),
                           axis=vector(x_maskA - x_lens2, 0, 0),
                           radius=r_beam_big,
                           color=vector(0.7, 0.9, 1.0),
                           opacity=0.4)

label(pos=vector(x_lens2, 1.0, 0),
      text="Lens 2\n(collimate + magnify)", box=False, height=11)

####################################
# 4. Complementary masks：兩片平行板，距離拉開很多
####################################
# Mask A
maskA_plate = box(pos=vector(x_maskA, 0, 0),
                  size=vector(0.04, 1.4, 1.4),
                  color=vector(0.6, 0.6, 0.6),
                  opacity=0.3)

maskA_aperture = cylinder(pos=vector(x_maskA - 0.02, 0, 0),
                          axis=vector(0.04, 0, 0),
                          radius=0.35,
                          color=vector(0.2, 0.5, 1.0),
                          opacity=1.0)

label(pos=vector(x_maskA, 1.1, 0),
      text="Mask A", box=False, height=11)

# Mask A'（明顯在後面一大段距離，但與 Mask A 平行）
maskB_plate = box(pos=vector(x_maskB, 0, 0),
                  size=vector(0.04, 1.4, 1.4),
                  color=vector(0.85, 0.85, 0.85),
                  opacity=0.9)

maskB_block = cylinder(pos=vector(x_maskB - 0.02, 0, 0),
                       axis=vector(0.04, 0, 0),
                       radius=0.35,
                       color=vector(0.1, 0.1, 0.1),
                       opacity=0.9)

label(pos=vector(x_maskB, -1.1, 0),
      text="Mask A'", box=False, height=11)

# 中間 beam
beam_between_masks = cylinder(pos=vector(x_maskA, 0, 0),
                              axis=vector(x_maskB - x_maskA, 0, 0),
                              radius=r_beam_big,
                              color=vector(0.7, 0.9, 1.0),
                              opacity=0.2)

####################################
# 5. Screen + Airy pattern 示意
####################################
screen = box(pos=vector(x_screen, 0, 0),
             size=vector(0.05, 1.8, 1.8),
             color=vector(0.95, 0.95, 1.0),
             opacity=0.7)

label(pos=vector(x_screen, 1.2, 0),
      text="Screen", box=False, height=12)

beam_to_screen = cylinder(pos=vector(x_maskB, 0, 0),
                          axis=vector(x_screen - x_maskB, 0, 0),
                          radius=r_beam_big,
                          color=vector(0.7, 0.9, 1.0),
                          opacity=0.25)

# Airy pattern（示意）
center_pos = vector(x_screen + 0.01, 0, 0)
central_spot = sphere(pos=center_pos,
                      radius=0.09,
                      color=vector(0.2, 0.4, 1.0),
                      opacity=0.9)

airy_radii = [0.14, 0.24, 0.34, 0.44]
airy_thickness = 0.035

for i, r in enumerate(airy_radii):
    ring_color = vector(0.2, 0.4, 1.0) if i % 2 == 0 else vector(0.7, 0.8, 1.0)
    ring(pos=center_pos,
         axis=vector(1, 0, 0),
         radius=r,
         thickness=airy_thickness,
         color=ring_color,
         opacity=0.85)

label(pos=center_pos + vector(0.2, -1.1, 0),
      text="Airy pattern (schematic)", box=False, height=11)

print("Updated system geometry (bigger mask spacing & double-cone lens1) complete.")

# b12901058 add
input("Press Enter to close the VPython window...")