import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from torchvision import datasets, transforms
from main import AutoEncoder

print("Loading model and preparing visualization...")

# Load the model with proper handling of the checkpoint structure
model_path = "checkpoints/autoencoder_best.pth"
model = AutoEncoder()

# Load the full checkpoint and extract just the model weights
checkpoint = torch.load(model_path)
if "state_dict" in checkpoint:
    # Extract model weights from the training checkpoint
    model.load_state_dict(checkpoint["state_dict"])
else:
    # Direct loading as fallback
    model.load_state_dict(checkpoint)

model.eval()

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root="./mnist/", train=False, transform=transform, download=True)

# Create the main figure
fig = plt.figure(figsize=(14, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3)

# Create a grid layout
gs = plt.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[2, 1, 1])

# Create the axes for different parts of the visualization
ax_latent = fig.add_subplot(gs[0, 0])        # Latent space plot
ax_image = fig.add_subplot(gs[0, 1])         # Current image
ax_detail = fig.add_subplot(gs[0, 2])        # Detail view with pixel values
ax_info = fig.add_subplot(gs[1, :2])         # Information panel
ax_hist = fig.add_subplot(gs[1, 2])          # Histogram of pixel values

# Configure the latent space plot
ax_latent.set_xlim(-1, 1)
ax_latent.set_ylim(-1, 1)
ax_latent.set_xlabel('Latent Dimension 1')
ax_latent.set_ylabel('Latent Dimension 2')
ax_latent.set_title('Latent Space Distribution')
ax_latent.grid(True, linestyle='--', alpha=0.7)
ax_latent.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax_latent.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Add a cursor point in the latent space
point, = ax_latent.plot([0], [0], 'ro', markersize=8, zorder=10)

# Configure the image display
ax_image.set_title('Reconstructed Image')
ax_image.axis('off')
img_display = ax_image.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)

# Configure the detail view
ax_detail.set_title('Pixel Values')
ax_detail.axis('off')
img_detail = ax_detail.imshow(np.zeros((28, 28)), cmap='plasma', vmin=0, vmax=1)
plt.colorbar(img_detail, ax=ax_detail, fraction=0.046, pad=0.04)

# Configure the histogram
ax_hist.set_title('Pixel Value Distribution')
ax_hist.set_xlabel('Pixel Value')
ax_hist.set_ylabel('Frequency')
# We'll update the histogram dynamically

# Configure the information panel
ax_info.axis('off')
info_text = ax_info.text(0.02, 0.5, 'Move cursor to explore the latent space', 
                        fontsize=12, va='center', transform=ax_info.transAxes)

# Colors for different digits
digit_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Map MNIST digits to their latent representations
print("Mapping MNIST digits to latent space...")
digit_latents = [[] for _ in range(10)]
for i in range(min(2000, len(test_data))):  # Process a subset for speed
    img, label = test_data[i]
    
    # Extract the label (handle both tensor and int cases)
    if hasattr(label, 'item'):
        digit = label.item()
    else:
        digit = label
    
    # Get the latent representation
    with torch.no_grad():
        latent, _ = model(img.view(1, -1))
    
    # Store if within our display bounds
    if -1 <= latent[0, 0] <= 1 and -1 <= latent[0, 1] <= 1:
        digit_latents[digit].append(latent.numpy()[0])

# Plot the digit distributions in latent space
for digit in range(10):
    if digit_latents[digit]:
        points = np.array(digit_latents[digit])
        ax_latent.scatter(points[:, 0], points[:, 1], 
                         color=digit_colors[digit], 
                         alpha=0.6, 
                         label=f'{digit}',
                         edgecolors='w',
                         linewidths=0.5,
                         s=30)

ax_latent.legend(loc='upper right', fontsize=10)

# Function to find the most likely digit for a latent point
def get_nearest_digit(x, y):
    min_dist = float('inf')
    nearest_digit = -1
    
    # Check against all mapped digits
    for digit in range(10):
        if not digit_latents[digit]:
            continue
        
        points = np.array(digit_latents[digit])
        # Find distance to nearest point of this digit
        distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        min_digit_dist = np.min(distances)
        
        if min_digit_dist < min_dist:
            min_dist = min_digit_dist
            nearest_digit = digit
    
    return nearest_digit, min_dist

# Function to update the visualization
def update_display(x, y):
    # Generate reconstructed image from latent point
    latent_vector = torch.tensor([[x, y]], dtype=torch.float32)
    
    with torch.no_grad():
        reconstructed = model.decoder(latent_vector)
    
    # Get the image as a numpy array
    img_array = reconstructed.numpy().reshape(28, 28)
    
    # Update the main image display
    img_display.set_data(img_array)
    
    # Update the detail view
    img_detail.set_data(img_array)
    
    # Update the histogram
    ax_hist.clear()
    ax_hist.set_title('Pixel Value Distribution')
    ax_hist.set_xlabel('Pixel Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.hist(img_array.flatten(), bins=20, range=(0, 1), color='purple', alpha=0.7)
    ax_hist.set_xlim(0, 1)
    
    # Update the cursor position
    point.set_data([x], [y])
    
    # Get information about the point
    nearest_digit, distance = get_nearest_digit(x, y)
    
    # Update information panel
    info_str = (
        f"Latent Coordinates: ({x:.2f}, {y:.2f})\n\n"
        f"Nearest Digit Class: {nearest_digit}\n"
        f"Distance to Nearest Example: {distance:.3f}\n\n"
        f"Image Statistics:\n"
        f"  • Mean Pixel Value: {np.mean(img_array):.3f}\n"
        f"  • Max Pixel Value: {np.max(img_array):.3f}\n"
        f"  • Min Pixel Value: {np.min(img_array):.3f}\n"
        f"  • Standard Deviation: {np.std(img_array):.3f}\n"
    )
    info_text.set_text(info_str)
    
    # Redraw the figure
    fig.canvas.draw_idle()

# Event handlers for interactive exploration
def on_mouse_move(event):
    if event.inaxes == ax_latent:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # Constrain to bounds
            x = max(min(x, 1), -1)
            y = max(min(y, 1), -1)
            update_display(x, y)

def on_mouse_click(event):
    if event.inaxes == ax_latent:
        x, y = event.xdata, event.ydata
        # Constrain to bounds
        x = max(min(x, 1), -1)
        y = max(min(y, 1), -1)
        update_display(x, y)

# Connect event handlers
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_mouse_click)

# Optional: Add a grid of reference points in the latent space
grid_density = 5
x_grid = np.linspace(-1, 1, grid_density)
y_grid = np.linspace(-1, 1, grid_density)

for x in x_grid:
    for y in y_grid:
        ax_latent.plot(x, y, 'k.', alpha=0.2, markersize=3)

# Initial display update (center of latent space)
print("Initialization complete! Move your cursor in the latent space to explore.")
update_display(0, 0)

plt.show()