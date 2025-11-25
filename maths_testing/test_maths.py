#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

H, W = 256, 256                # synthetic test image size
FREQS = [4, 8, 16, 32]         # geometric ladder
N = 4                          # number of phase shifts
DELTA = np.array([2*np.pi*n/N for n in range(N)])

# mask thresholds
Th1 = 0.20                     # visibility threshold (gamma = B/A)
Th2 = 20                       # absolute modulation threshold

SIMULATION_MODE = True         # disables saturation-based masking

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_sinusoid(freq, delta):
    """
    Generate clean synthetic phase-shifted fringe patterns.
    Returns H×W float images (NOT uint8 — avoids false saturation).
    """
    x = np.linspace(0, 1, W)
    X = np.tile(x, (H, 1))

    # “real” phase pattern: 2π f x
    phi = 2 * np.pi * freq * X

    # intensity: A + B sin(phi + delta)
    A = 110.0                       # mean brightness
    B = 90.0                        # modulation
    return A + B * np.sin(phi + delta)

# Store captured patterns
I = {f: [] for f in FREQS}

for f in FREQS:
    for d in DELTA:
        I[f].append(generate_sinusoid(f, d))
    I[f] = np.stack(I[f], axis=0)   # → shape (N, H, W)


# ============================================================
# PHASE EXTRACTION (per frequency)
# ============================================================

phi_wrapped = {}
A_f = {}
B_f = {}
gamma_f = {}
valid_f = {}

for f in FREQS:

    # sums from Eq.(2)
    S = np.sum(I[f] * np.sin(DELTA)[:, None, None], axis=0)
    C = np.sum(I[f] * np.cos(DELTA)[:, None, None], axis=0)

    # Eq.(2): wrapped phase in (-π, π]
    phi_wrapped[f] = np.arctan2(S, C)

    # Eq.(3): mean intensity
    A_f[f] = np.mean(I[f], axis=0)

    # Eq.(4): modulation amplitude
    B_f[f] = (2.0 / N) * np.sqrt(S**2 + C**2)

    # Eq.(5): visibility
    gamma_f[f] = B_f[f] / A_f[f]

    # mask: visibility and absolute modulation
    mask = (gamma_f[f] > Th1) & (B_f[f] > Th2)

    # disable saturation rule in simulation mode
    if not SIMULATION_MODE:
        sat = (np.max(I[f], axis=0) > 250) | (np.min(I[f], axis=0) < 5)
        mask &= ~sat

    valid_f[f] = mask


# ============================================================
# TEMPORAL UNWRAPPING (Eq. 6–7)
# ============================================================

# Coarsest frequency is unwrapped already
phi_unwrapped = {}
phi_unwrapped[FREQS[0]] = phi_wrapped[FREQS[0]].copy()

# combined mask: AND of per-frequency masks
valid_total = valid_f[FREQS[0]].copy()

# unwrap finer frequencies iteratively
for i in range(1, len(FREQS)):
    f_low = FREQS[i - 1]
    f_high = FREQS[i]
    r = f_high / f_low

    # Eq.(7): integer lift
    k = np.round((r * phi_unwrapped[f_low] - phi_wrapped[f_high]) / (2*np.pi))

    # Eq.(6): lifted phase
    phi_unwrapped[f_high] = phi_wrapped[f_high] + 2*np.pi * k

    # update mask
    valid_total &= valid_f[f_high]

# final output phase
phi_final = phi_unwrapped[FREQS[-1]]
mask_final = valid_total


# ============================================================
# DEBUG / DIAGNOSTICS
# ============================================================

print("Final phase map stats:")
print(f"  shape = {phi_final.shape}")
print(f"  mean phase = {phi_final.mean():.2f} rad")
print(f"  valid pixels = {100*np.mean(mask_final):.1f}%")

# ============================================================
# SAVE PLOTS (headless-friendly)
# ============================================================

plt.figure(figsize=(12, 4))

# Wrapped phase
plt.subplot(1, 3, 1)
plt.title("Wrapped phase (coarsest freq)")
plt.imshow(phi_wrapped[FREQS[0]], cmap="twilight", vmin=-np.pi, vmax=np.pi)
plt.colorbar(fraction=0.046)

# Unwrapped phase
plt.subplot(1, 3, 2)
plt.title("Unwrapped phase (final)")
plt.imshow(phi_final * mask_final, cmap="twilight")
plt.colorbar(fraction=0.046)

# Mask
plt.subplot(1, 3, 3)
plt.title("Valid mask")
plt.imshow(mask_final.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)

plt.tight_layout()
plt.savefig("/home/pi/fringe/maths_testing/phase_debug.png", dpi=150)
print("Saved plot as /home/pi/fringe/maths_testing/phase_debug.png")

plt.figure(figsize=(6,4))
plt.title("Difference: unwrapped - wrapped")
plt.imshow(phi_final - phi_wrapped[FREQS[0]], cmap="viridis")
plt.colorbar()
plt.savefig("/home/pi/fringe/maths_testing/difference.png")

