import taichi as ti

# arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.metal)

max_num_particles = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)

per_vertex_color = ti.Vector.field(3, ti.f32, shape=max_num_particles)

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    num_particles[None] += 1


def main():
    window = ti.ui.Window("Circles", (768, 768), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    new_particle(0.3, 0.3, False)
    new_particle(0.3, 0.4, False)
    new_particle(0.4, 0.4, False)

    radius = 0.02  # initial radius

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            elif e.key == ti.ui.LMB:
                pos = window.get_cursor_pos()
                new_particle(pos[0], pos[1], int(window.is_pressed(ti.ui.SHIFT)))

        canvas.set_background_color((1, 1, 1))
        canvas.circles(x, radius=radius, color=(0, 0, 0))

        with gui.sub_window("Options", 0.05, 0.05, 0.9, 0.2) as w:
            radius = w.slider_float("Circle Radius", radius, 0.01, 0.05)

        window.show()

if __name__ == "__main__":
    main()
