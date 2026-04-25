"""Autoencodeur MNIST en MLX (architecture convolutionnelle).

Objectif : battre la MSE de référence PyTorch (0.03087685) le plus vite possible.
Mesure et logge l'epoch + le temps écoulé au moment où la cible est franchie.
"""
import argparse
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from torchvision import datasets, transforms

TARGET_MSE = 0.03087685  # à battre (PyTorch MLP, 600 epochs)


def load_mnist():
    """Charge MNIST via torchvision et renvoie des arrays MLX en NHWC float32 [0,1]."""
    tx = transforms.ToTensor()
    train = datasets.MNIST(root="./mnist/", train=True, transform=tx, download=True)
    test = datasets.MNIST(root="./mnist/", train=False, transform=tx, download=True)

    def to_mx(ds):
        # ds.data : uint8 (N, 28, 28). On normalise et on ajoute la dimension canal en dernier (NHWC).
        x = ds.data.numpy().astype(np.float32) / 255.0
        x = x[..., None]  # (N, 28, 28, 1)
        return mx.array(x), mx.array(ds.targets.numpy())

    return to_mx(train), to_mx(test)


def _kaiming_init_(layer: nn.Linear, gain: float = math.sqrt(2.0)):
    """Init Kaiming-normal (adaptée aux activations ReLU/GELU) sur une couche linéaire."""
    fan_in = layer.weight.shape[1]
    std = gain / math.sqrt(fan_in)
    layer.weight = mx.random.normal(shape=layer.weight.shape) * std
    if "bias" in layer:
        layer.bias = mx.zeros_like(layer.bias)


def _small_init_(layer: nn.Linear, std: float = 0.01):
    """Init très petite (utile pour les couches qui précèdent une saturation type Tanh)."""
    layer.weight = mx.random.normal(shape=layer.weight.shape) * std
    if "bias" in layer:
        layer.bias = mx.zeros_like(layer.bias)


class MLPVAE(nn.Module):
    """β-VAE MLP : encoder produit (μ, log_var), reparam, decoder reconstruit.

    Pendant l'entraînement : z = μ + ε⊙σ (ε ~ N(0, I)).
    À l'éval : on prend z = μ pour avoir un encodage déterministe comparable.

    Encoder:  784 → 1024 → 512 → 256 → 128 → 32 → (μ, log_var)
    Decoder:  z → 32 → 128 → 256 → 512 → 1024 → 784 (Sigmoid)
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        sizes_enc = [784, 1024, 768, 512, 256, 64]
        sizes_dec = [latent_dim, 64, 256, 512, 1024, 1536, 1024, 784]
        self.enc_layers = [nn.Linear(a, b) for a, b in zip(sizes_enc[:-1], sizes_enc[1:])]
        self.enc_norms = [nn.LayerNorm(s) for s in sizes_enc[1:]]
        self.fc_mu = nn.Linear(sizes_enc[-1], latent_dim)
        self.fc_logvar = nn.Linear(sizes_enc[-1], latent_dim)

        self.dec_layers = [nn.Linear(a, b) for a, b in zip(sizes_dec[:-1], sizes_dec[1:])]
        self.dec_norms = [nn.LayerNorm(s) for s in sizes_dec[1:-1]]

        for layer in self.enc_layers:
            _kaiming_init_(layer)
        # Init petite sur fc_mu / fc_logvar pour démarrer proche de N(0, exp(0)=1)
        _small_init_(self.fc_mu, std=0.01)
        _small_init_(self.fc_logvar, std=0.01)
        for layer in self.dec_layers[:-1]:
            _kaiming_init_(layer)
        _kaiming_init_(self.dec_layers[-1], gain=1.0)

    def encode(self, x):
        h = x.reshape(x.shape[0], -1)
        for layer, norm in zip(self.enc_layers, self.enc_norms):
            h = nn.relu(norm(layer(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # log_var clampé pour éviter exp() qui explose au début.
        logvar = mx.clip(logvar, -8.0, 4.0)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(shape=mu.shape)
        return mu + eps * std

    def decode(self, z):
        h = z
        for layer, norm in zip(self.dec_layers[:-1], self.dec_norms):
            h = nn.relu(norm(layer(h)))
        h = mx.sigmoid(self.dec_layers[-1](h))
        return h.reshape(h.shape[0], 28, 28, 1)

    def __call__(self, x, deterministic: bool = False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


# Alias rétro-compat
MLPAutoEncoder = MLPVAE


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation : feats ← feats * (1 + γ(z)) + β(z).

    γ et β sont des projections linéaires de z, broadcastées sur la dimension spatiale
    (NHWC). Init petite mais non-nulle pour que z module dès la 1re epoch sans saturer.
    """

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(latent_dim, channels)
        self.beta = nn.Linear(latent_dim, channels)
        # Init : gamma_init et beta_init centrés autour de 0, magnitude 0.5 ; biais à 0.
        for layer in (self.gamma, self.beta):
            layer.weight = mx.random.normal(shape=layer.weight.shape) * 0.5
            layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, h, z):
        # h: (B, H, W, C), z: (B, latent_dim)
        b = z.shape[0]
        gamma = self.gamma(z).reshape(b, 1, 1, -1)
        beta = self.beta(z).reshape(b, 1, 1, -1)
        return h * (1.0 + gamma) + beta


class ConvFiLMAutoEncoder(nn.Module):
    """Encoder conv → latent → Decoder conv avec injection FiLM à chaque niveau.

    L'idée : z module les feature maps du décodeur via gamma/beta à chaque conv,
    pas seulement au point d'entrée. Ça force le décodeur à exploiter z (sinon
    FiLM est mort) et brise le mode collapse "image moyenne".

    Encoder:  28×28×1 → 14×14×32 → 7×7×64 → 7×7×128 → flat → 256 → latent (Tanh)
    Decoder:  z → seed (7×7×128) → FiLM+Conv → 14×14×64 → FiLM+Conv → 28×28×32
              → FiLM+Conv → 28×28×1 (Sigmoid)
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_gn1 = nn.GroupNorm(8, 32, pytorch_compatible=True)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_gn2 = nn.GroupNorm(8, 64, pytorch_compatible=True)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc_gn3 = nn.GroupNorm(8, 128, pytorch_compatible=True)
        self.enc_fc1 = nn.Linear(7 * 7 * 128, 256)
        self.enc_fc2 = nn.Linear(256, latent_dim)
        _small_init_(self.enc_fc2, std=0.01)

        # --- Decoder seed: z → 7×7×128 spatial ---
        self.dec_seed = nn.Linear(latent_dim, 7 * 7 * 128)
        _kaiming_init_(self.dec_seed)

        # --- Decoder blocks (Conv → FiLM(z) → GroupNorm → SiLU) ---
        self.dec_conv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dec_film1 = FiLMBlock(latent_dim, 128)
        self.dec_gn1 = nn.GroupNorm(8, 128, pytorch_compatible=True)

        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_film2 = FiLMBlock(latent_dim, 64)
        self.dec_gn2 = nn.GroupNorm(8, 64, pytorch_compatible=True)

        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_film3 = FiLMBlock(latent_dim, 32)
        self.dec_gn3 = nn.GroupNorm(8, 32, pytorch_compatible=True)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        h = nn.silu(self.enc_gn1(self.enc_conv1(x)))
        h = nn.silu(self.enc_gn2(self.enc_conv2(h)))
        h = nn.silu(self.enc_gn3(self.enc_conv3(h)))
        h = h.reshape(h.shape[0], -1)
        h = nn.silu(self.enc_fc1(h))
        return mx.tanh(self.enc_fc2(h))

    def decode(self, z):
        h = self.dec_seed(z).reshape(z.shape[0], 7, 7, 128)
        h = nn.silu(self.dec_gn1(self.dec_film1(self.dec_conv1(h), z)))
        h = nn.silu(self.dec_gn2(self.dec_film2(self.dec_conv2(h), z)))
        h = nn.silu(self.dec_gn3(self.dec_film3(self.dec_conv3(h), z)))
        return mx.sigmoid(self.dec_out(h))

    def __call__(self, x):
        return self.decode(self.encode(x))


ConvAutoEncoder = MLPVAE


def count_params(model: nn.Module) -> int:
    return sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))


def evaluate(model: nn.Module, x_test: mx.array, batch: int = 1024) -> float:
    """MSE déterministe : on prend z = μ (pas d'échantillonnage)."""
    model.eval()
    n = x_test.shape[0]
    total_sq = mx.array(0.0, dtype=mx.float32)
    for i in range(0, n, batch):
        xb = x_test[i : i + batch]
        out = model(xb, deterministic=True)
        recon = out[0] if isinstance(out, tuple) else out
        total_sq = total_sq + ((recon - xb) ** 2).sum()
    mx.eval(total_sq)
    return float(total_sq.item()) / float(n * 28 * 28)


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> mx.array:
    coords = mx.arange(size, dtype=mx.float32) - (size - 1) / 2
    g1d = mx.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    return g1d[:, None] * g1d[None, :]  # (size, size)


def ssim_loss(x: mx.array, y: mx.array, data_range: float = 1.0,
              kernel_size: int = 11, sigma: float = 1.5) -> mx.array:
    """1 - SSIM moyen entre x et y. Inputs en NHWC, valeurs ∈ [0, data_range]."""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    kernel = _gaussian_kernel_2d(kernel_size, sigma).reshape(1, kernel_size, kernel_size, 1)
    pad = kernel_size // 2

    mu_x = mx.conv2d(x, kernel, padding=pad)
    mu_y = mx.conv2d(y, kernel, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = mx.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = mx.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = mx.conv2d(x * y, kernel, padding=pad) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den
    return 1.0 - mx.mean(ssim_map)


def cosine_lr(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 1e-5,
    warmup_steps: int = 0,
) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def cosine_warm_restart_lr(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 1e-5,
    warmup_steps: int = 0,
    n_cycles: int = 3,
    decay_per_cycle: float = 0.6,
) -> float:
    """Cosine avec warm restarts : n cycles, lr_max décroît à chaque restart."""
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    rem_steps = total_steps - warmup_steps
    cycle_len = max(1, rem_steps // n_cycles)
    s = step - warmup_steps
    cycle_idx = min(n_cycles - 1, s // cycle_len)
    s_in_cycle = s - cycle_idx * cycle_len
    cur_lr_max = lr_max * (decay_per_cycle ** cycle_idx)
    progress = s_in_cycle / cycle_len
    return lr_min + 0.5 * (cur_lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=float, default=TARGET_MSE)
    p.add_argument("--save", type=str, default="checkpoints/mlx_best.safetensors")
    p.add_argument("--beta", type=float, default=1e-4, help="poids de la KL (β-VAE)")
    p.add_argument("--cycles", type=int, default=3, help="nombre de cycles cosine warm restart")
    p.add_argument("--noise", type=float, default=0.0, help="bruit gaussien sur l'input (denoising VAE)")
    p.add_argument("--wd", type=float, default=1e-5, help="weight decay AdamW")
    p.add_argument("--ssim", type=float, default=0.5, help="poids de la perte SSIM (1-SSIM)")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                   help="précision de calcul (bf16 = mixed precision sur Apple Silicon)")
    p.add_argument("--eval-every", type=int, default=1, help="évalue toutes les N epochs")
    args = p.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    print("Chargement MNIST…")
    (x_train, _), (x_test, _) = load_mnist()
    print(f"  train: {tuple(x_train.shape)} | test: {tuple(x_test.shape)}")

    model = ConvAutoEncoder(latent_dim=args.latent_dim)
    mx.eval(model.parameters())

    if args.dtype == "bf16":
        # Caste tous les paramètres float du modèle en bfloat16.
        from mlx.utils import tree_flatten, tree_unflatten
        flat = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]
        model.update(tree_unflatten(flat))
        x_train = x_train.astype(mx.bfloat16)
        x_test = x_test.astype(mx.bfloat16)
        mx.eval(model.parameters())
        print(f"Modèle bf16 : {count_params(model):,} paramètres | latent_dim={args.latent_dim}")
    else:
        print(f"Modèle fp32 : {count_params(model):,} paramètres | latent_dim={args.latent_dim}")
    print(f"Cible PyTorch à battre : MSE = {args.target:.8f}")

    n_train = x_train.shape[0]
    steps_per_epoch = math.ceil(n_train / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(steps_per_epoch * 5, total_steps // 10)

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.wd)

    beta = args.beta
    noise_std = args.noise
    ssim_w = args.ssim

    def loss_fn(model, x):
        if noise_std > 0:
            x_in = mx.clip(x + noise_std * mx.random.normal(shape=x.shape), 0.0, 1.0)
        else:
            x_in = x
        recon, mu, logvar = model(x_in, deterministic=False)
        mse = mx.mean((recon - x) ** 2)
        kl = -0.5 * mx.mean(mx.sum(1 + logvar - mu * mu - mx.exp(logvar), axis=1))
        loss = mse + beta * kl
        if ssim_w > 0:
            loss = loss + ssim_w * ssim_loss(recon, x)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    state = [model.state, optimizer.state, mx.random.state]

    def _step(x):
        loss, grads = loss_and_grad(model, x)
        optimizer.update(model, grads)
        return loss

    step = mx.compile(_step, inputs=state, outputs=state)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = "training_log_mlx.csv"
    log_file = open(log_path, "w", buffering=1)  # line-buffered
    log_file.write("epoch,avg_loss,test_mse,epoch_time_s,wall_time_s,lr\n")

    best_mse = float("inf")
    target_hit_epoch = None
    target_hit_time = None
    wall_t0 = time.perf_counter()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        perm = mx.array(np.random.permutation(n_train))
        x_train_shuf = x_train[perm]

        running = 0.0
        n_batches = 0
        ep_t0 = time.perf_counter()

        for i in range(0, n_train, args.batch):
            xb = x_train_shuf[i : i + args.batch]
            optimizer.learning_rate = cosine_warm_restart_lr(
                global_step, total_steps, args.lr,
                warmup_steps=warmup_steps, n_cycles=args.cycles,
            )
            loss = step(xb)
            mx.eval(state)
            running += float(loss.item())
            n_batches += 1
            global_step += 1

        ep_time = time.perf_counter() - ep_t0
        avg_loss = running / n_batches
        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
        test_mse = evaluate(model, x_test, batch=args.batch) if do_eval else float("nan")
        wall = time.perf_counter() - wall_t0

        log_file.write(
            f"{epoch+1},{avg_loss:.8f},{test_mse:.8f},{ep_time:.3f},{wall:.3f},"
            f"{optimizer.learning_rate.item():.6f}\n"
        )

        flag = ""
        if do_eval and test_mse < args.target and target_hit_epoch is None:
            target_hit_epoch = epoch + 1
            target_hit_time = wall
            flag = "  ← BATTU!"

        if do_eval:
            print(
                f"E{epoch+1:3d}/{args.epochs} | train {avg_loss:.6f} | test {test_mse:.6f} "
                f"| epoch {ep_time:.2f}s | total {wall:.1f}s | lr {optimizer.learning_rate.item():.5f}{flag}"
            )
        elif (epoch + 1) % 10 == 0:
            print(
                f"E{epoch+1:3d}/{args.epochs} | train {avg_loss:.6f} | (no eval) "
                f"| epoch {ep_time:.2f}s | total {wall:.1f}s"
            )

        if do_eval and test_mse < best_mse:
            best_mse = test_mse
            # Sauve en fp32 pour rester compatible avec le visualiseur, même si on entraîne en bf16.
            flat = {k: v.astype(mx.float32) for k, v in nn.utils.tree_flatten(model.parameters())}
            mx.save_safetensors(args.save, flat)

    log_file.close()
    wall_total = time.perf_counter() - wall_t0
    print("\n" + "=" * 60)
    print(f"  Cible PyTorch  : MSE = {args.target:.8f}")
    print(f"  Meilleure MLX  : MSE = {best_mse:.8f}")
    if target_hit_epoch is not None:
        print(f"  → BATTU à l'epoch {target_hit_epoch} après {target_hit_time:.1f}s "
              f"({target_hit_time/60:.2f} min)")
        print(f"  → Gain final  : {(args.target - best_mse) / args.target * 100:.1f}% de MSE en moins")
    else:
        print(f"  → Cible non atteinte (manque {best_mse - args.target:.6f})")
    print(f"  Temps total    : {wall_total:.1f}s ({wall_total/60:.2f} min)")
    print(f"  Params         : {count_params(model):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
