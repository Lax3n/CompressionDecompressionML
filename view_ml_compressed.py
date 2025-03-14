import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from main import AutoEncoder

def main():
    # Chargement du modèle
    model_path = "checkpoints/autoencoder_best.pth"
    model = AutoEncoder()
    
    # Chargement des poids du modèle
    checkpoint = torch.load(model_path)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Préparation de la transformation
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Chargement du jeu d'entraînement ET du jeu de test
    train_data = datasets.MNIST(root="./mnist/", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="./mnist/", train=False, transform=transform, download=True)
    
    # Création d'un seul jeu de données combiné
    full_dataset = torch.utils.data.ConcatDataset([train_data, test_data])
    print(f"Taille totale du jeu de données: {len(full_dataset)} exemples")
    
    # Création du DataLoader sans multiprocessing pour éviter les erreurs Windows
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=500, shuffle=False, num_workers=0
    )
    
    # Préparation de la figure
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
    ax.set_xlim(-0.8, 0.8)  # Limites légèrement élargies pour accommoder tous les points
    ax.set_ylim(-0.7, 0.9)
    ax.set_xlabel('Dimension Latente 1', fontsize=14)
    ax.set_ylabel('Dimension Latente 2', fontsize=14)
    ax.set_title('Espace Latent MNIST Complet (70 000 exemples)', fontsize=16)
    
    # Couleurs pour les différents chiffres
    digit_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    print("Projection de tous les digits MNIST dans l'espace latent...")
    
    # Traitement de l'ensemble du jeu de données par lots
    latent_vectors = []
    labels = []
    
    # Pour suivre la progression
    total_batches = len(full_loader)
    batch_count = 0
    
    with torch.no_grad():  # Désactivation du calcul de gradient pour l'inférence
        for images, batch_labels in full_loader:
            # Passage avant à travers l'encodeur
            batch_latent, _ = model(images.view(images.size(0), -1))
            
            # Stockage des résultats
            latent_vectors.append(batch_latent)
            labels.append(batch_labels)
            
            # Affichage de la progression
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Traitement: {batch_count}/{total_batches} lots ({batch_count*100/total_batches:.1f}%)")
    
    # Concaténation de tous les lots
    latent_vectors = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"Tracé de {len(labels)} points dans l'espace latent...")
    
    # Traçage de chaque chiffre avec sa couleur correspondante
    for digit in range(10):
        # Récupération des indices de ce chiffre spécifique
        indices = (labels == digit).nonzero(as_tuple=True)[0]
        
        # Extraction des vecteurs latents pour ce chiffre
        digit_latents = latent_vectors[indices]
        
        print(f"Tracé du chiffre {digit}: {len(indices)} exemples")
        
        # Tracé de ces points
        plt.scatter(
            digit_latents[:, 0].cpu().numpy(), 
            digit_latents[:, 1].cpu().numpy(),
            color=digit_colors[digit], 
            alpha=0.3,  # Alpha réduit pour une meilleure visibilité avec plus de points
            s=4,        # Points plus petits pour éviter le surencombrement
            label=f'Chiffre {digit}'
        )
    
    # Ajout d'une légende
    plt.legend(loc='upper right')
    
    # Ajout des axes de coordonnées
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Ajout d'une grille
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Sauvegarde de la visualisation
    plt.savefig("mnist_full_latent_space_map.png", dpi=700)  # DPI plus élevé pour les détails
    plt.show()
    
    # Statistiques sur l'espace latent
    print(f"Nombre total d'échantillons tracés: {len(labels)}")
    print(f"Statistiques de l'espace latent:")
    for digit in range(10):
        digit_count = (labels == digit).sum().item()
        print(f"  Chiffre {digit}: {digit_count} exemples")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Nécessaire sous Windows
    main()