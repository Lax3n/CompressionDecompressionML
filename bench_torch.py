"""Mesure la MSE moyenne du modèle PyTorch entraîné sur MNIST test (cible à battre)."""
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from main import AutoEncoder

CHECKPOINT = "checkpoints/autoencoder_best.pth"
BATCH = 1024


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = AutoEncoder().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    test_data = datasets.MNIST(
        root="./mnist/", train=False,
        transform=transforms.ToTensor(), download=True,
    )
    loader = DataLoader(test_data, batch_size=BATCH, shuffle=False, num_workers=0)

    total_sq_err = 0.0
    total_pixels = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(-1, 28 * 28).to(device)
            _, decoded = model(x)
            total_sq_err += ((decoded - x) ** 2).sum().item()
            total_pixels += x.numel()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    mse = total_sq_err / total_pixels
    print(f"PyTorch baseline | test MSE = {mse:.8f} | {elapsed*1000:.1f} ms")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    return mse


if __name__ == "__main__":
    main()
