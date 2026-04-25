"""Explorateur interactif de l'espace latent — version MLX.

Charge le meilleur checkpoint MLX (safetensors) et permet d'explorer le latent
space 2D à la souris. Reproduit l'UX de interac_decompression.py mais avec le
modèle MLX entraîné dans main_mlx.py.
"""
import argparse

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from torchvision import datasets, transforms

from main_mlx import ConvAutoEncoder

DIGIT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def load_model(path: str, latent_dim: int = 2) -> ConvAutoEncoder:
    model = ConvAutoEncoder(latent_dim=latent_dim)
    weights = list(mx.load(path).items())
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    model.eval()
    return model


def map_test_set(model: ConvAutoEncoder, n_samples: int = 2000):
    """Encode un sous-ensemble de MNIST test pour visualiser la distribution latente."""
    tx = transforms.ToTensor()
    test = datasets.MNIST(root="./mnist/", train=False, transform=tx, download=True)
    n = min(n_samples, len(test))
    imgs = np.stack([test[i][0].numpy() for i in range(n)], axis=0)  # (n, 1, 28, 28)
    labels = np.array([int(test[i][1]) for i in range(n)])
    imgs_mx = mx.array(imgs.transpose(0, 2, 3, 1).astype(np.float32))  # NHWC
    out = model.encode(imgs_mx)
    # VAE renvoie (mu, logvar) — on prend mu comme représentation déterministe.
    z = out[0] if isinstance(out, tuple) else out
    mx.eval(z)
    return np.asarray(z), labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mlx_best.safetensors")
    p.add_argument("--samples", type=int, default=2000)
    args = p.parse_args()

    print(f"Chargement du modèle MLX depuis {args.checkpoint}…")
    model = load_model(args.checkpoint)

    print(f"Projection de {args.samples} exemples MNIST dans l'espace latent…")
    z_test, labels = map_test_set(model, n_samples=args.samples)
    x_min, x_max = float(z_test[:, 0].min()), float(z_test[:, 0].max())
    y_min, y_max = float(z_test[:, 1].min()), float(z_test[:, 1].max())
    print(f"  range latent : x ∈ [{x_min:.2f}, {x_max:.2f}], y ∈ [{y_min:.2f}, {y_max:.2f}]")

    fig = plt.figure(figsize=(14, 7))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.07, top=0.92, wspace=0.25)
    gs = plt.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[2, 1, 1])
    ax_latent = fig.add_subplot(gs[0, 0])
    ax_image = fig.add_subplot(gs[0, 1])
    ax_detail = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, :2])
    ax_hist = fig.add_subplot(gs[1, 2])

    # VAE : pas de Tanh, le mu peut sortir de [-1, 1]. On adapte les bornes au scatter.
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)
    ax_latent.set_xlim(*xlim)
    ax_latent.set_ylim(*ylim)
    ax_latent.set_xlabel("Dimension Latente 1")
    ax_latent.set_ylabel("Dimension Latente 2")
    ax_latent.set_title("Espace Latent (MLX)")
    ax_latent.grid(True, linestyle="--", alpha=0.6)
    ax_latent.axhline(0, color="k", alpha=0.3)
    ax_latent.axvline(0, color="k", alpha=0.3)

    for d in range(10):
        m = labels == d
        if m.any():
            ax_latent.scatter(
                z_test[m, 0], z_test[m, 1],
                c=DIGIT_COLORS[d], alpha=0.55, s=24,
                label=str(d), edgecolors="w", linewidths=0.4,
            )
    ax_latent.legend(loc="upper right", fontsize=9, ncol=2)

    cursor, = ax_latent.plot([0], [0], "ro", markersize=9, zorder=10)

    ax_image.set_title("Reconstruction")
    ax_image.axis("off")
    img_disp = ax_image.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)

    ax_detail.set_title("Pixel Values")
    ax_detail.axis("off")
    img_detail = ax_detail.imshow(np.zeros((28, 28)), cmap="plasma", vmin=0, vmax=1)
    plt.colorbar(img_detail, ax=ax_detail, fraction=0.046, pad=0.04)

    ax_info.axis("off")
    info_text = ax_info.text(
        0.02, 0.5, "Clique ou glisse dans l'espace latent.",
        fontsize=11, va="center", transform=ax_info.transAxes,
    )

    def nearest_digit(x: float, y: float):
        d2 = (z_test[:, 0] - x) ** 2 + (z_test[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        return int(labels[i]), float(np.sqrt(d2[i]))

    def update(x: float, y: float):
        z = mx.array([[x, y]], dtype=mx.float32)
        recon = model.decode(z)
        mx.eval(recon)
        img = np.asarray(recon).reshape(28, 28)
        img_disp.set_data(img)
        img_detail.set_data(img)

        ax_hist.clear()
        ax_hist.set_title("Distribution des pixels")
        ax_hist.set_xlim(0, 1)
        ax_hist.hist(img.flatten(), bins=20, range=(0, 1), color="purple", alpha=0.7)

        cursor.set_data([x], [y])
        digit, dist = nearest_digit(x, y)
        info_text.set_text(
            f"Coordonnées latentes : ({x:.2f}, {y:.2f})\n\n"
            f"Chiffre le plus proche : {digit}  (distance {dist:.3f})\n\n"
            f"Statistiques pixels :\n"
            f"  • moyenne : {img.mean():.3f}\n"
            f"  • max     : {img.max():.3f}\n"
            f"  • écart-type : {img.std():.3f}"
        )
        fig.canvas.draw_idle()

    def _clamp(event):
        x = float(np.clip(event.xdata, xlim[0], xlim[1]))
        y = float(np.clip(event.ydata, ylim[0], ylim[1]))
        return x, y

    def on_motion(event):
        if event.inaxes is ax_latent and event.xdata is not None:
            update(*_clamp(event))

    def on_click(event):
        if event.inaxes is ax_latent and event.xdata is not None:
            update(*_clamp(event))

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_click)

    update(0.0, 0.0)
    plt.show()


if __name__ == "__main__":
    main()
