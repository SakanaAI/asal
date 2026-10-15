import taichi as ti

ti.init()

nch = 9
w, h = 100, 100
cs = 2
nim = nch//3
img_w, img_h = w * cs * nim, h * cs

state = ti.Vector.field(n=3, dtype=ti.f32, shape=(nim, w, h))
img = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h))

@ti.kernel
def init_state():
    # inits each of the 3 images with 1,g,b r,1,b g,r,1 gradients
    for _, i, j in state:
        state[0,i,j].rgb = [1,i/w,j/h]
        state[1,i,j].rgb = [j/h,1,i/w]
        state[2,i,j].rgb = [i/w,j/h,1]

@ti.kernel
def wrap_img():
    for i, j in img:
        offset = (i//cs) // w
        wind = (i//cs) % w
        hind = (j//cs) % h
        img[i,j].rgb = state[offset, wind, hind].rgb

init_state()
wrap_img()

# Creates a GUI of the size of the gray-scale image
gui = ti.GUI('Wrapped Image', (img_w, img_h))
while gui.running:
    gui.set_image(img)
    gui.show()
