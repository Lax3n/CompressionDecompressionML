"""Compare visuellement originaux / reconstruction PyTorch / reconstruction MLX.

Charge les deux meilleurs checkpoints et les met côte-à-côte sur le même
échantillon de test MNIST.
"""
import argparse

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_unflatten
from torchvision import datasets, transforms

from main import AutoEncoder as TorchAE
from main_mlx import ConvAutoEncoder as MlxModel


def torch_recon(model, x_np, device):
    """x_np: (N, 28, 28). Renvoie (N, 28, 28) après recon."""
    model.eval()
    x = torch.from_numpy(x_np).to(device).view(-1, 28 * 28)
    with torch.no_grad():
        _, decoded = model(x)
    return decoded.cpu().numpy().reshape(-1, 28, 28)


def mlx_recon(model, x_np):
    """x_np: (N, 28, 28). Renvoie (N, 28, 28) après recon."""
    model.eval()
    x = mx.array(x_np[..., None].astype(np.float32))  # NHWC
    out = model(x, deterministic=True) if hasattr(model, "encode") else model(x)
    recon = out[0] if isinstance(out, tuple) else out
    mx.eval(recon)
    return np.asarray(recon).reshape(-1, 28, 28)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-ckpt", default="checkpoints/autoencoder_best.pth")
    p.add_argument("--mlx-ckpt", default="checkpoints/mlx_best.safetensors")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default="img/comparaison/torch_vs_mlx.png")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Données
    test = datasets.MNIST(root="./mnist/", train=False, transform=transforms.ToTensor(), download=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test), size=args.n, replace=False)
    imgs = np.stack([np.asarray(test[i][0]).squeeze() for i in idx], axis=0)  # (N, 28, 28)
    labels = [int(test[i][1]) for i in idx]

    # PyTorch
    print(f"Chargement PyTorch : {args.torch_ckpt}")
    torch_model = TorchAE().to(device)
    ckpt = torch.load(args.torch_ckpt, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    torch_model.load_state_dict(state)
    recon_t = torch_recon(torch_model, imgs, device)

    # MLX
    print(f"Chargement MLX : {args.mlx_ckpt}")
    mlx_model = MlxModel()
    weights = list(mx.load(args.mlx_ckpt).items())
    mlx_model.update(tree_unflatten(weights))
    mx.eval(mlx_model.parameters())
    recon_m = mlx_recon(mlx_model, imgs)

    # MSE par image
    mse_t = ((recon_t - imgs) ** 2).mean(axis=(1, 2))
    mse_m = ((recon_m - imgs) ** 2).mean(axis=(1, 2))
    print(f"  MSE moyenne sur l'échantillon : torch={mse_t.mean():.5f}  mlx={mse_m.mean():.5f}")

    # Plot
    fig, axes = plt.subplots(3, args.n, figsize=(args.n * 1.4, 4.5))
    for i in range(args.n):
        for ax_row, img in zip(axes, [imgs[i], recon_t[i], recon_m[i]]):
            ax = ax_row[i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, i].set_title(str(labels[i]), fontsize=10)
    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel(f"PyTorch\n(MSE={mse_t.mean():.4f})", fontsize=10)
    axes[2, 0].set_ylabel(f"MLX\n(MSE={mse_m.mean():.4f})", fontsize=10)
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Sauvegardé : {args.out}")


if __name__ == "__main__":
    main()
