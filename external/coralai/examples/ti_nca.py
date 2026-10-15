import torch
import torch.nn as nn
import taichi as ti
from timeit import Timer

VISUALIZE = False
"""
TODO:
- Perception kernel of offsets
- Multi neighborhood? - concentric circles?
- Custom activation functions
    - Lookup table of functions
    - Coefficient and a bias
    - (x1*f1(x) + b1) + (k2*f2(x) + b2) + (x3*f3(x) + b3) + ... + (xN*fN(x) + bN)
    .... Is this a NN for a neuron?
    - Man just use NEAT
    

"""
# arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=ti.metal)
w, h = 400, 400
n_ch = 3
kernel_radius = 3
n_im = n_ch//3  # 3 channels per image, nim images next to each other widthwise
cell_size = 2
img_w, img_h = w * cell_size * n_im, h * cell_size
render_buffer = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=(img_w, img_h)
        )

@ti.kernel
def write_to_renderer(state: ti.types.ndarray(dtype=ti.f32)):
    for i, j in render_buffer:
        subimg_index = i // (w * cell_size)
        offset = subimg_index * 3
        xind = (i//cell_size) % w
        yind = (j//cell_size) % h
        for ch in ti.static(range(3)):
            render_buffer[i, j][ch] = state[offset+ch, xind, yind]

@ti.kernel
def add_noise(strength: ti.f32, state: ti.types.ndarray()):
    for i, j in ti.ndrange(w, h):
        for ch in ti.static(range(3)):
            state[ch, i, j] += ti.random(float) * strength

@ti.kernel
def draw_rad_zero(
        pos_x: ti.f32,
        pos_y: ti.f32,
        radius: ti.i32,
        state: ti.types.ndarray()
    ):
    ind_x = int(pos_x * w)
    ind_y = int(pos_y * h)
    offset = int(pos_x * n_im) * 3
    for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
        for ch in ti.static(range(3)):
            if (i**2) + j**2 < radius**2:
                state[offset+ch, (i + ind_x * n_im) % w, (j + ind_y) % h] +=1

# torch.Size([10, 400, 400]) torch.Size([10, 10, 3, 3])
@ti.kernel
def conv2d(state: ti.types.ndarray(ndim=3), weights: ti.types.ndarray(ndim=4), out: ti.types.ndarray(ndim=3)):
    for o_chid, i, j in ti.ndrange(n_ch, w, h):
        o_chsum = 0.0
        for in_chid, offi, offj in ti.ndrange(n_ch, (-kernel_radius, kernel_radius+1), (-kernel_radius, kernel_radius+1)):
            ci = (i + offi) % w
            cj = (j + offj) % h
            # Calculate the distance from the center
            dist = ti.sqrt(offi**2 + offj**2)
            # Adjust the weights based on the distance
            adjusted_weight = weights[in_chid, o_chid, offi, offj] / (dist + 1)  # Adding 1 to avoid division by zero
            o_chsum += adjusted_weight * state[in_chid, ci, cj]
        out[o_chid, i, j] = o_chsum

@ti.kernel
def special_activation(state: ti.types.ndarray()):
    for I in ti.grouped(state):
        state[I] = ti.exp(state[I]) + 1.0

@ti.kernel
def inverse_sigmoid(state: ti.types.ndarray()):
    for I in ti.grouped(state):
        state[I] = ti.log(state[I] / (1.0 - state[I]))

@ti.data_oriented
class NCA(nn.Module):
    def __init__(self, channel_count, visualize = True):
        super(NCA, self).__init__()
        self.state = torch.zeros(channel_count, w, h)
        self.weights = torch.randn(channel_count, channel_count, kernel_radius*2+1, kernel_radius*2+1)
        self.convout = torch.zeros(channel_count, w, h)
        self.conv = nn.Conv2d(
            channel_count,
            channel_count,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.paused = False
        self.brush_radius = 5
        self.drawing = False
        self.perturbing_weights = False
        self.perturbation_strength = 0.1
        self.noise_strength = 0.00
        self.visualize = visualize

    def perturb_weights(self):
        self.weights += torch.randn_like(self.weights) * self.perturbation_strength
        self.conv.weight.data = self.weights

    def ch_norm_(self, input_tensor):
        # The input tensor shape is assumed to be (width, height, num_channels).
        # Compute mean and variance.
        mean = input_tensor.mean(dim=(1, 2), keepdim=True)
        var = input_tensor.var(dim=(1, 2), keepdim=True, unbiased=False)
        
        # The in-place normalization is done next.
        # Subtract the mean and divide by the standard deviation in place.
        input_tensor.sub_(mean).div_(torch.sqrt(var + 1e-5))

    def forward(self, x):
        conv2d(x, self.weights, self.convout)
        x=self.convout
        x = nn.ReLU()(x)
        self.ch_norm_(x)
        x = torch.sigmoid(x)
        x = x.squeeze(0)
        return x

    def apply_rules(self):
        self.state = self.forward(self.state)

    def draw(self, window=None):
        if self.drawing:
            pos = window.get_cursor_pos()
            draw_rad_zero(pos[0], pos[1], self.brush_radius, self.state)

    def update(self, window=None):
        add_noise(self.noise_strength, self.state)

        if self.visualize:
            self.check_input(window)
            self.draw(window)
            write_to_renderer(self.state)

        if self.perturbing_weights:
            self.perturb_weights()
    
    def check_input(self, window):
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and window.is_pressed(ti.ui.SHIFT):
                self.drawing = True
            elif e.key == ti.ui.SPACE:
                self.state *= 0.0
            # elif e.key == 'r':
            #     self.perturbing_weights = True

        for e in window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False
            # elif e.key == 'r':
            #     self.perturbing_weights = False

        
def main_vis(img_w, img_h, num_ch):
    model = NCA(num_ch, visualize=True)
    window = ti.ui.Window("NCA", (img_w, img_h), fps_limit=400, vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    steps_per_frame = 1

    while window.running:
        if not model.paused:
            for _ in range(steps_per_frame):
                model.apply_rules()
            model.update(window)

        canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / img_w, img_w)
        opt_h = min(180 / img_h, img_h)
        with gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as w:
            model.brush_radius = w.slider_int("Brush Radius", model.brush_radius, 1, 200)
            model.noise_strength = w.slider_float("Noise Strength", model.noise_strength, 0.0, 5.0)
            model.perturbation_strength = w.slider_float("Perturbation Strength", model.perturbation_strength, 0.0, 5.0)
            steps_per_frame = w.slider_int("Steps per Frame", steps_per_frame, 1, 100)
            model.paused = w.checkbox("Pause", model.paused)
            model.perturbing_weights = w.checkbox("Perturb Weights", model.perturbing_weights)
        canvas.set_image(render_buffer)
        window.show()

if __name__ == "__main__":
    if VISUALIZE:
        main_vis(img_w, img_h, num_ch=n_ch)
    else:  
        model = NCA(n_ch, visualize=False)
        model.apply_rules()
        # Initialize Timer object with the function to measure
        t = Timer(model.apply_rules)

        # Measure time taken for 1000 calls
        time_taken = t.timeit(number=1000)

        # Calculate FPS
        fps = 1000 / time_taken

        print(f"Frames per second: {fps}")
