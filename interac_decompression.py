import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from torchvision import datasets, transforms

from main import AutoEncoder

# Charger le modèle
model_path = "checkpoints/autoencoder_best.pth"
model = AutoEncoder()

# Chargement adapté au nouveau format de sauvegarde
checkpoint = torch.load(model_path)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Configurer la figure
fig, (ax_latent, ax_image) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Préparer l'affichage de l'espace latent - strictement [-1,1]²
ax_latent.set_xlim(-1, 1)
ax_latent.set_ylim(-1, 1)
ax_latent.set_xlabel('Dimension 1')
ax_latent.set_ylabel('Dimension 2')
ax_latent.set_title('Espace latent [-1,1]²')
ax_latent.grid(True)
ax_latent.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax_latent.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Point initial dans l'espace latent
latent_point = np.array([0.0, 0.0])
point, = ax_latent.plot(latent_point[0], latent_point[1], 'ro', markersize=10, 
                       picker=10, zorder=100)  # augmenté picker et zorder pour une meilleure sélection

# Préparer l'affichage de l'image reconstruite
ax_image.set_title('Image reconstruite')
ax_image.axis('off')

# Générer l'image initiale
latent_tensor = torch.tensor([latent_point], dtype=torch.float32)
with torch.no_grad():
    reconstructed = model.decoder(latent_tensor)

img_display = ax_image.imshow(reconstructed.numpy().reshape(28, 28), cmap='gray')

# Variables pour le drag and drop
is_dragging = False

# Fonction pour mettre à jour l'image
def update_image(x, y):
    # Assurer que les coordonnées restent dans [-1,1]²
    x = max(min(x, 1), -1)
    y = max(min(y, 1), -1)
    
    latent_vector = np.array([[x, y]], dtype=np.float32)
    latent_tensor = torch.tensor(latent_vector, dtype=torch.float32)
    
    with torch.no_grad():
        reconstructed = model.decoder(latent_tensor)
    
    img_display.set_data(reconstructed.numpy().reshape(28, 28))
    
    # Mettre à jour les coordonnées affichées
    ax_latent.set_title(f'Position: ({x:.2f}, {y:.2f})')
    
    # Forcer le rafraîchissement
    fig.canvas.draw_idle()

# Fonctions pour gérer le drag and drop
def on_pick(event):
    global is_dragging
    if event.artist == point:
        is_dragging = True
        # Conserver le pointeur de la souris lors de la sélection
        fig.canvas.draw_idle()

def on_release(event):
    global is_dragging
    is_dragging = False

def on_motion(event):
    global is_dragging
    if is_dragging and event.inaxes == ax_latent:
        x, y = event.xdata, event.ydata
        
        # Limiter aux bornes [-1,1]²
        x = max(min(x, 1), -1)
        y = max(min(y, 1), -1)
        
        # Mettre à jour la position du point
        point.set_data([x], [y])
        
        # Mettre à jour l'image
        update_image(x, y)

# Connecter les événements
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Ajouter aussi un clic direct dans l'espace latent
def on_click(event):
    if event.inaxes == ax_latent and not is_dragging:
        x, y = event.xdata, event.ydata
        
        # Limiter aux bornes [-1,1]²
        x = max(min(x, 1), -1)
        y = max(min(y, 1), -1)
        
        # Mettre à jour la position du point
        point.set_data([x], [y])
        
        # Mettre à jour l'image
        update_image(x, y)

fig.canvas.mpl_connect('button_press_event', on_click)

# Ajouter tous les chiffres de 0 à 9 avec des couleurs différentes
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root="./mnist/", train=False, transform=transform, download=True)

# Couleurs pour chaque chiffre
colors = ['blue', 'orange', 'green', 'red', 'purple', 
          'brown', 'pink', 'gray', 'olive', 'cyan']

# Collecter des échantillons pour chaque chiffre
for digit in range(10):
    digit_samples = []
    count = 0
    
    # Chercher des exemples du chiffre actuel
    for i in range(len(test_data)):
        if count >= 7000:  # Limiter à 100 exemples par chiffre pour des performances raisonnables
            break
        img, label = test_data[i]
        if label == digit:
            digit_samples.append(img.view(-1, 28*28))
            count += 1
    
    if digit_samples:
        digit_samples = torch.cat(digit_samples)
        with torch.no_grad():
            latents, _ = model(digit_samples)
        latents_np = latents.numpy()
        
        # Ne conserver que les points dans [-1,1]²
        mask = (latents_np[:, 0] >= -1) & (latents_np[:, 0] <= 1) & (latents_np[:, 1] >= -1) & (latents_np[:, 1] <= 1)
        latents_np = latents_np[mask]
        
        if len(latents_np) > 0:
            # Tracer les points pour ce chiffre
            ax_latent.scatter(latents_np[:, 0], latents_np[:, 1], 
                             c=colors[digit], alpha=0.5, s=20,
                             label=f'{digit}', zorder=5)

# Ajouter une légende compacte
ax_latent.legend(loc='upper right', fontsize='small', ncol=2)

# Titre
plt.suptitle('Explorateur de l\'espace latent [-1,1]² avec reconstruction')

plt.tight_layout()
plt.show()