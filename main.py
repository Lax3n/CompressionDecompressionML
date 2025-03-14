import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import os
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for MNIST
transform = transforms.Compose([transforms.ToTensor()])

# Optimized AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_size=28*28, latent_dim=2):  # Maximum compression with latent_dim=2
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_size), nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def main():
    # Argument parsing for resume training
    parser = argparse.ArgumentParser(description='Train or resume training of autoencoder')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start from')
    args = parser.parse_args()
    
    # Load MNIST dataset
    train_data = datasets.MNIST(root="./mnist/", train=True, transform=transform, download=True)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model, loss, and optimizer
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Create directory for checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # Resume from checkpoint if specified
    best_loss = float('inf')
    start_epoch = args.start_epoch
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            
            # Check if the checkpoint contains state dictionary only or full training state
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                # Full training state
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                # Only state dictionary
                model.load_state_dict(checkpoint)
                print(f"=> loaded model state from '{args.resume}'")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    # CSV logging setup - append if resuming, create new if not
    csv_mode = 'a' if args.resume and start_epoch > 0 else 'w'
    with open('training_log.csv', csv_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header only if creating a new file
        if csv_mode == 'w':
            writer.writerow(['Epoch', 'Avg Loss', 'Time (s)', 'Compression Ratio'])
        
        # Training loop
        EPOCH = args.epochs
        
        for epoch in range(start_epoch, EPOCH):
            start_time = time.time()
            running_loss = 0.0
            
            model.train()
            for x, _ in train_loader:
                x = x.view(-1, 28*28).to(device)
                
                # Forward pass
                encoded, decoded = model(x)
                loss = criterion(decoded, x)
                
                # Backward and optimize
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            epoch_time = time.time() - start_time
            
            # Compression ratio calculation (28*28 / latent_dim)
            compression_ratio = (28*28) / 2  # If using latent_dim=2
            
            # Log results
            writer.writerow([epoch + 1, avg_loss, epoch_time, compression_ratio])
            print(f'Epoch: {epoch+1}/{EPOCH} | Avg loss: {avg_loss:.8f} | Time: {epoch_time:.2f}s | Compression: {compression_ratio:.1f}x')
            
            # Save best model with full training state
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, "checkpoints/autoencoder_best.pth")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, f"checkpoints/autoencoder_epoch_{epoch+1}.pth")
            
            # Update learning rate
            scheduler.step(avg_loss)
    
    # Load best model for visualization
    checkpoint = torch.load("checkpoints/autoencoder_best.pth")
    model.load_state_dict(checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint)
    model.eval()
    
    # Visualization
    N_TEST_IMG = 15  # Increased to see more examples
    view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).float() / 255.0
    view_data = view_data.to(device)
    
    with torch.no_grad():
        encoded_data, decoded_data = model(view_data)
        
    # Convert to numpy for visualization
    decoded_data = decoded_data.cpu().numpy()
    encoded_data = encoded_data.cpu().numpy()
    
    # Plot original vs reconstructed images
    fig, ax = plt.subplots(2, N_TEST_IMG, figsize=(N_TEST_IMG*2, 4))
    for i in range(N_TEST_IMG):
        ax[0, i].imshow(np.reshape(view_data[i].cpu().numpy(), (28, 28)), cmap="gray")
        ax[1, i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap="gray")
        ax[0, i].set_title("Original")
        ax[1, i].set_title("Reconstructed")
        ax[0, i].set_xticks([]); ax[0, i].set_yticks([])
        ax[1, i].set_xticks([]); ax[1, i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig("reconstruction.png")
    plt.show()
    
    # Visualize the latent space if it's 2D
    if model.encoder[-2].out_features == 2:
        # Get encoded representations for the entire dataset
        encoded_points = []
        labels = []
        
        test_loader = DataLoader(train_data, batch_size=1000, shuffle=False)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(-1, 28*28).to(device)
                encoded, _ = model(x)
                encoded_points.append(encoded.cpu().numpy())
                labels.append(y.numpy())
        
        encoded_points = np.vstack(encoded_points)
        labels = np.concatenate(labels)
        
        # Plot the 2D latent space
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encoded_points[:, 0], encoded_points[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title("2D Latent Space Visualization")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.savefig("latent_space.png")
        plt.show()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()