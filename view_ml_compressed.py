import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import os
from main import AutoEncoder
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

# Charger le modèle entraîné
model_path = "checkpoints/autoencoder_best.pth"
model = AutoEncoder()
model.load_state_dict(torch.load(model_path))
model.eval()

# Charger les données de test
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root="./mnist/", train=False, transform=transform, download=True)

# Collecter des échantillons de chiffres 3
num_samples = 4000
samples = []
labels = []


i = 0
while len(samples) < num_samples and i < len(test_data):
    img, label = test_data[i]
    if label == 3:  
        samples.append(img.view(-1, 28*28))
        labels.append(label)
    i += 1

# Concaténer les échantillons collectés
samples = torch.cat(samples)

# Obtenir les représentations latentes
with torch.no_grad():
    latents, _ = model(samples)

latents_np = latents.numpy()

# Vérifier que la dimension latente est 2
if latents_np.shape[1] != 2:
    print(f"Attention: La dimension latente est {latents_np.shape[1]}, pas 2. La visualisation 3D nécessite une dimension latente de 2.")
else:
    # Créer une grille pour la visualisation 3D
    x_min, x_max = latents_np[:, 0].min() - 0.2, latents_np[:, 0].max() + 0.2
    y_min, y_max = latents_np[:, 1].min() - 0.2, latents_np[:, 1].max() + 0.2
    
    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    xi, yi = np.meshgrid(xi, yi)

    # Calculer la densité
    positions = np.vstack([xi.ravel(), yi.ravel()])
    kernel = gaussian_kde(latents_np.T)
    zi = np.reshape(kernel(positions).T, xi.shape)

    # Tracer la surface 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='plasma', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Densité')
    ax.set_title(f'Densité de l\'espace latent pour les chiffres {labels[0]}')
    
    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Améliorer la vue 3D
    ax.view_init(elev=35, azim=15)
    
    # Sauvegarder l'image
    plt.savefig(f"latent_space_density_3d_digit{labels[0]}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualisation créée pour {len(samples)} chiffres 2")
    print(f"Image sauvegardée sous 'latent_space_density_3d_digit3.png'")