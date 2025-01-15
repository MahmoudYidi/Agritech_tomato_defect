import torch
import numpy as np
import matplotlib.pyplot as plt
from auto_vae import VAE, generate_reconstructions, normalize_new_tensor
import pickle

# Load normalization components
with open('normalization_components.pkl', 'rb') as f:
    components = pickle.load(f)
components = {key: torch.tensor(value) for key, value in components.items()}

# Load the input tensor and normalize
#data_dir = "/mnt/c/Users/User/Desktop/Network_Isolation/KernelPCA/output/ful_anom_tensor.pt"
data_dir = "/mnt/c/Users/User/Desktop/Network_Isolation/KernelPCA/output/ful_test_tensor.pt"
hsi_tensor = torch.load(data_dir)
hsi_tensor = normalize_new_tensor(hsi_tensor, components)
hsi_tensor = hsi_tensor.permute(0, 3, 1, 2)  # Rearrange for model input

# Define the model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_channels=4).to(device)
model.load_state_dict(torch.load('auto_vae2.pth'))
model.eval()

# Prepare data for a single sample and run the model
hsi_tensor = hsi_tensor.to(device)
sample = hsi_tensor[0].unsqueeze(0)  # Select the first sample and add batch dimension
original, reconstructions, _, _, _ = generate_reconstructions(model, sample, 1)

# Compute the error between ground truth and reconstruction
error = (original - reconstructions).abs().squeeze().cpu()

# Calculate reconstruction loss
reconstruction_loss = error.sum().item()  # Sum of absolute errors

# Print message if the loss exceeds the threshold
if reconstruction_loss > 1900:
    print("Anomalous Tomato")

# Visualize the error overlayed on the original image
original_img = original.squeeze().cpu().permute(1, 2, 0)[:, :, :3]  # Select first 3 bands for RGB-like display
recon_img = reconstructions.squeeze().cpu().permute(1, 2, 0)[:, :, :3]
error_img = error.permute(1, 2, 0).numpy()

# Define a threshold for highlighting errors
threshold = 0.035  # Adjust this value based on your specific use case
error_highlight = (error_img > threshold).astype(np.float32)

# Create a red overlay where the error exceeds the threshold
red_overlay = np.zeros_like(original_img)
red_overlay[:, :, 0] = error_highlight[:, :, 0]  # Red channel
red_overlay[:, :, 1] = 0  # Green channel
red_overlay[:, :, 2] = 0  # Blue channel

# Combine original image with red overlay
highlighted_img = np.clip(original_img + red_overlay, 0, 1)  # Ensure values stay within [0, 1]

# Set up the plot with subplots
plt.figure(figsize=(15, 5))

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title('Ground Truth')
plt.axis('off')

# Plot the reconstruction
plt.subplot(1, 3, 2)
plt.imshow(recon_img)
plt.title('Reconstructed')
plt.axis('off')

# Plot the highlighted image
plt.subplot(1, 3, 3)
plt.imshow(highlighted_img)
plt.title('Error Highlight')
plt.axis('off')

plt.tight_layout()
plt.show()
