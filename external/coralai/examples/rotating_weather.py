import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import importlib

device = torch.device('cpu')

def torch_rot_and_signals():
    def generate_signal(freq, period, amplitude, rot, X, Y):
        return (amplitude * torch.sin(freq * X * torch.cos(rot) - Y * torch.sin(rot) + period) +
                amplitude * torch.cos(freq * X * torch.sin(rot) + Y * torch.cos(rot) + period))

    num_signals = 5
    min_freq, max_freq = 0.5, 10
    min_rot, max_rot = 0, np.pi
    min_amp, max_amp = 1, 1

    freqs = (max_freq - min_freq) * torch.rand(num_signals).to(device) + min_freq
    rots = (max_rot - min_rot) * torch.rand(num_signals).to(device) + min_rot
    amps = (max_amp - min_amp) * torch.rand(num_signals).to(device) + min_amp
    periods = torch.zeros(num_signals).to(device)

    width, height = 100, 100

    x = torch.linspace(0, np.pi*2, width).to(device)
    y = torch.linspace(0, np.pi*2, height).to(device)
    X, Y = torch.meshgrid(x, y)

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])
    fig = plt.figure(figsize=(8, 8))

    ax_main = plt.subplot(gs[0, 1])
    ax_x = plt.subplot(gs[1, 1])
    ax_y = plt.subplot(gs[0, 0])
    ax_signals = [plt.subplot(gs[1, 0], position=[i/num_signals, 0, 1/num_signals, 1]) for i in range(num_signals)]
    
    for ax in ax_signals:
        ax.axis('off')

    def update(period):
        signals = [generate_signal(freqs[i], period, amps[i], rots[i], X, Y) for i in range(num_signals)]
        combined_signal = sum(signals)

        # Main combined signal
        ax_main.clear()
        ax_main.imshow(combined_signal.cpu(), cmap='gray')
            
        #  X and Y frequencies
        ax_x.clear()
        ax_y.clear()
        for s in signals:
            ax_x.plot(x.cpu(), s[height//2, :].cpu(), color='red', linestyle='dashed', alpha=0.5)
            ax_y.plot(s[:, width//2].cpu(), y.cpu(), color='red', linestyle='dashed', alpha=0.5)
        ax_x.plot(x.cpu(), combined_signal[height//2, :].cpu(), color='black')
        ax_y.plot(combined_signal[:, width//2].cpu(), y.cpu(), color='black')
        ax_y.invert_xaxis()

        # Set y-axis limits
        ax_x.set_ylim([-max_amp*len(signals), max_amp*len(signals)])
        ax_y.set_xlim([-max_amp*len(signals), max_amp*len(signals)])
        #  2D individual signals
        #  for i, s in enumerate(signals):
        #      ax_signals[i].imshow(s.cpu(), cmap='gray')

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi/min_freq, 100), interval=25)
    plt.tight_layout()
    plt.show()


def torch_rot():

    def generate_signal(freq, period, amplitude, rot, X, Y):
        return (amplitude * torch.sin(freq * X * torch.cos(rot) - Y * torch.sin(rot) + period) +
                amplitude * torch.cos(freq * X * torch.sin(rot) + Y * torch.cos(rot) + period))

    num_signals = 4
    min_freq, max_freq = 0.2, 3
    min_rot, max_rot = 0, np.pi
    min_amp, max_amp = 1, 1

    freqs = torch.rand(num_signals) * (max_freq - min_freq) + min_freq
    rots = torch.rand(num_signals) * (max_rot - min_rot) + min_rot
    amps = torch.rand(num_signals) * (max_amp - min_amp) + min_amp
    periods = torch.zeros(num_signals)

    width, height = 200, 200
    x = torch.linspace(0, np.pi*2, width)
    y = torch.linspace(0, np.pi*2, height)
    X, Y = torch.meshgrid(x, y)

    def update(period):
        signals = [generate_signal(freqs[i], period, amps[i], rots[i], X, Y) for i in range(num_signals)]
        combined_signal = sum(signals)
        ax.clear()
        ax.imshow(combined_signal, cmap='gray')


    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 100), interval=50)

    plt.show()


def separate_xy_signals():
    num_x_signals = 2
    num_y_signals = 2
    num_signals = num_x_signals + num_y_signals
    min_freq, max_freq = 0.2, 3
    min_rot, max_rot = 0, np.pi
    min_amp, max_amp = 1, 1
    min_period, max_period = 0, np.pi*2

    freqs = (max_freq - min_freq) * torch.rand(num_signals).to(device) + min_freq
    # rots = (max_rot - min_rot) * torch.rand(num_signals).to(device) + min_rot
    amps = (max_amp - min_amp) * torch.rand(num_signals).to(device) + min_amp
    periods = (max_period - min_period) * torch.rand(num_signals).to(device) + min_period

    width, height = 100, 100
    x = torch.linspace(0, np.pi*2, width).to(device)
    y = torch.linspace(0, np.pi*2, height).to(device)

    x_signals = [amps[i] * torch.sin(freqs[i] * x + periods[i]) for i in range(num_x_signals)]
    y_signals = [amps[i] * torch.sin(freqs[i] * y + periods[i]) for i in range(num_y_signals)]
    x_combined = sum(x_signals)
    y_combined = sum(y_signals)

    # X, Y = torch.meshgrid(x, y)
    # combined_signal = X# * x_combined# + Y * y_combined
    X = x_combined.unsqueeze(0).repeat(height, 1)
    Y = y_combined.unsqueeze(1).repeat(1, width)
    combined_signal = X + Y

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])

    fig = plt.figure(figsize=(8, 8))
    ax_combined = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[1, 1])
    ax_y = fig.add_subplot(gs[0, 0])
    ax_mini = fig.add_subplot(gs[1, 0])

    ax_combined.set_title('Combined Signal')
    ax_x.set_title('X Signals')
    ax_y.set_title('Y Signals')
    # ax_minix4.set_title('X Signals x 4')

    # ax_x.set_ylim([-max_amp*len(signals), max_amp*len(signals)])
    # ax_y.set_xlim([-max_amp*len(signals), max_amp*len(signals)])

    ax_combined.imshow(combined_signal.cpu().numpy(), cmap='gray')

    for ix in range(num_x_signals):
        ax_x.plot(x, x_signals[ix], color='red', linestyle='dashed', alpha=0.5)
        
    for iy in range(num_y_signals):
        ax_y.plot(y_signals[iy], y, color='red', linestyle='dashed', alpha=0.5)

    ax_x.plot(x, x_combined, color='blue')
    ax_y.plot(y_combined, y, color='blue')
    ax_y.invert_xaxis()
    plt.show()

def wacky_rot():
    # Shared parameters and signal generation
    num_signals = 4
    min_freq, max_freq = 0.1, np.pi*2
    min_rot, max_rot = 0, np.pi
    min_amp, max_amp = 1, 1
    noise_scale = 0.05  # Added noise scale

    freqs = torch.rand(num_signals) * (max_freq - min_freq) + min_freq
    rots = torch.rand(num_signals) * (max_rot - min_rot) + min_rot
    amps = torch.rand(num_signals) * (max_amp - min_amp) + min_amp
    periods = torch.zeros(num_signals)

    width, height = 200, 200
    min_angle, max_angle = -np.pi, np.pi
    x = torch.linspace(min_angle, max_angle, width)
    y = torch.linspace(min_angle, max_angle, height)
    X, Y = torch.meshgrid(x, y)

    def generate_signal(freq, period, amplitude, rot, X, Y):
        noise = torch.randn_like(X) * noise_scale  # Added noise
        return (amplitude * torch.sin(freq * X * torch.cos(rot) - Y * torch.sin(rot) + period) +
                amplitude * torch.cos(freq * X * torch.sin(rot) + Y * torch.cos(rot) + period)) + noise

    # Visualization using GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.25])  # 3 rows and 2 columns

    ax1 = fig.add_subplot(gs[0, 0])  # Top-left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right plot
    ax3 = fig.add_subplot(gs[1, :])  # Bottom plot spanning both columns and 2 rows for thinness

    def update(period):
        # 1D wacky signals
        noise = torch.randn_like(x) * noise_scale
        unrotated_signals = [amps[i] * torch.sin(freqs[i] * x + period) + noise for i in range(num_signals)]
        x_prime = [torch.cos(rots[i]) * x - torch.sin(rots[i]) * unrotated_signals[i] + noise for i in range(num_signals)]
        y_prime = [torch.sin(rots[i]) * x + torch.cos(rots[i]) * unrotated_signals[i] for i in range(num_signals)]

        ax1.clear()
        for i in range(num_signals):
            ax1.plot(x_prime[i], y_prime[i])
        ax1.set_xlim([min_angle, max_angle])  # Set x limits
        ax1.set_ylim([min_angle, max_angle])  # Set y limits
        
        # 2D signals
        signals = [generate_signal(freqs[i], period, amps[i], rots[i], X, Y) for i in range(num_signals)]
        combined_signal = sum(signals)
        ax2.clear()
        ax2.imshow(combined_signal, cmap='gray', vmin=-max_amp*num_signals, vmax=max_amp*num_signals)  # Set min and max values for colormap

        # Sum of unrotated signals
        sum_signal = sum(unrotated_signals)
        ax3.clear()
        for signal in unrotated_signals:
            ax3.plot(x.numpy(), signal.numpy(), 'r--', alpha=0.3)
        ax3.plot(x.numpy(), sum_signal.numpy(), 'b-', linewidth=2)
        ax3.set_xlim([min_angle, max_angle])  # Set x limits
        ax3.set_ylim([-max_amp*num_signals, max_amp*num_signals])  # Set y limits

    ani = FuncAnimation(fig, update, frames=np.linspace(min_angle*2, max_angle*2, 100), interval=50)
    plt.tight_layout()
    plt.show()

# separate_xy_signals()
# torch_rot()
# torch_rot_and_signals()
wacky_rot()


