import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import cv2
import numpy as np
import numpy as np
import pickle


# Define VAE model #20 (128,16)
class VAE(nn.Module):
    def __init__(self, input_channels=16, latent_dim=100):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # Output: [16, 105, 105]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: [32, 53, 53]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: [64, 27, 27]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: [128, 14, 14]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: [256, 7, 7]
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 7 * 7)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [128, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [64, 27, 27]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [32, 53, 53]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [16, 105, 105]
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [input_channels, 210, 210]
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.decoder(x)
        return x, mu, logvar
    
    
def loss_function(recon_x, x, mu, logvar, beta):
    #MSE = F.mse_loss(recon_x, x, reduction='sum')
    MSE =F.l1_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD, MSE, KLD

def generate_reconstructions(model, data, beta):
    with torch.no_grad():
        output, mu, logvar = model(data)
        data = nn.functional.interpolate(data, size=output.size()[2:], mode='bilinear', align_corners=False)
        loss, mse, kld = loss_function(output, data, mu, logvar, beta)
    return data, output, loss, mse, kld


def plot_groundtruth_vs_prediction(model, dataloader, device, num_images=8):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))

    for i, data in enumerate(dataloader):
        if i >= num_images:
            break

        data = data[0].to(device)
        
        with torch.no_grad():
            reconstructions, mu, logvar = model(data)
        
        # Extract the first channel from the original and reconstructed data
        original_first_channel = data[:, 0, :, :].cpu().numpy()
        reconstructed_first_channel = reconstructions[:, 0, :, :].cpu().numpy()

        # Plotting the original and reconstructed images
        axes[i, 0].imshow(original_first_channel[0], cmap='gray')
        axes[i, 0].set_title(f"Ground Truth {i + 1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed_first_channel[0], cmap='gray')
        axes[i, 1].set_title(f"Reconstructed {i + 1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def thresholding(hsi_tensor, threshold):
    # Apply thresholding to each channel of each image
    thresholded_tensor = torch.zeros_like(hsi_tensor)

    # Iterate over images, channels, and pixels to apply thresholding
    for i in range(hsi_tensor.shape[0]):  # Iterate over images
        for j in range(hsi_tensor.shape[3]):  # Iterate over channels
            # Apply thresholding condition
            thresholded_tensor[i, :, :, j] = torch.where(hsi_tensor[i, :, :, j] < threshold,
                                                        hsi_tensor[i, :, :, j],
                                                        torch.zeros_like(hsi_tensor[i, :, :, j]))
    
    return thresholded_tensor

def thresholding_bin(hsi_tensor, threshold):
    # Initialize the output tensor with zeros
    thresholded_tensor = torch.zeros_like(hsi_tensor)

    # Iterate over images and channels to apply thresholding
    for i in range(hsi_tensor.shape[0]):  # Iterate over images
        for j in range(hsi_tensor.shape[3]):  # Iterate over channels
            # Apply thresholding condition
            thresholded_tensor[i, :, :, j] = torch.where(hsi_tensor[i, :, :, j] < threshold,
                                                         torch.ones_like(hsi_tensor[i, :, :, j]),
                                                         torch.zeros_like(hsi_tensor[i, :, :, j]))
    
    return thresholded_tensor

def thresholding_otsu(hsi_tensor):
    # Apply adaptive thresholding using Otsu's method to each channel of each image
    thresholded_tensor = torch.zeros_like(hsi_tensor)

    for i in range(hsi_tensor.shape[0]):  # Iterate over images
        for j in range(hsi_tensor.shape[3]):  # Iterate over channels
            # Convert tensor to numpy array
            hsi_channel_np = hsi_tensor[i, :, :, j].numpy().astype(np.uint8)
            # Apply Otsu's thresholding
            _, thresholded_np = cv2.threshold(hsi_channel_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Convert numpy array back to tensor
            thresholded_tensor[i, :, :, j] = torch.from_numpy(thresholded_np).float()
    
    return thresholded_tensor

def normalize_ori_hsi_tensor(hsi_tensor):
    # Convert the input tensor to float
    hsi_tensor = hsi_tensor.float()
    
    # Standard normalization
    mean = hsi_tensor.mean(dim=(0, 1, 2), keepdim=True)
    std = hsi_tensor.std(dim=(0, 1, 2), keepdim=True)
    #standard_normalized_tensor = (hsi_tensor - mean) / std
    standard_normalized_tensor = hsi_tensor
    #standard_normalized_tensor = hsi_tensor
    # Min-max normalization on standard normalized tensor
    min_val = standard_normalized_tensor.amin(dim=(0, 1, 2), keepdim=True)
    max_val = standard_normalized_tensor.amax(dim=(0, 1, 2), keepdim=True)
    # Print the min and max values to inspect them
    print(f"Min values: {min_val}")
    print(f"Max values: {max_val}")
    min_max_normalized_tensor = (standard_normalized_tensor - min_val) / (max_val - min_val)
    print(f"Min normalized values: {min_max_normalized_tensor.amin(dim=(0, 1, 2))}")
    print(f"Max normalized values: {min_max_normalized_tensor.amax(dim=(0, 1, 2))}")

    # Save normalization components
    normalization_components = {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val
    }

    return min_max_normalized_tensor, normalization_components


def normalize_new_tensor(new_tensor, components):
    # Convert the input tensor to float for precision
    new_tensor = new_tensor.float()
    
    # Standard normalization
    #new_standard_normalized_tensor = (new_tensor - components['mean']) / components['std']
    new_standard_normalized_tensor = new_tensor
    
    # Min-max normalization on standard normalized tensor
    new_min_max_normalized_tensor = (new_standard_normalized_tensor - components['min']) / (components['max'] - components['min'])
    
    return new_min_max_normalized_tensor

#with open('normalization_components.pkl', 'wb') as f:
  #  pickle.dump(normalization_components, f)
  
#with open('normalization_components.pkl', 'rb') as f:
   # saved_components = pickle.load(f)
   
#saved_components = {key: torch.tensor(value) for key, value in saved_components.items()}

def dilate_hyperspectral_images(tensor, kernel_size=10, iterations=1):
    # Convert the tensor to a numpy array
    tensor_np = tensor.numpy()

    # Get the dimensions of the tensor
    num_images, height, width, num_bands = tensor_np.shape

    # Define the structuring element for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Initialize an array to store the dilated images
    dilated_tensor_np = np.zeros_like(tensor_np)

    # Process each image and each band separately
    for img_idx in range(num_images):
        for band_idx in range(num_bands):
            # Extract the band
            band = tensor_np[img_idx, :, :, band_idx]

            # Apply dilation
            dilated_band = cv2.dilate(band, kernel, iterations=iterations)

            # Store the dilated band back into the result array
            dilated_tensor_np[img_idx, :, :, band_idx] = dilated_band

    # Convert the numpy array back to a PyTorch tensor
    dilated_tensor = torch.from_numpy(dilated_tensor_np)

    return dilated_tensor

def generate_reconstructions_error(model, data, beta):
    with torch.no_grad():
        output, mu, logvar = model(data)
        data = nn.functional.interpolate(data, size=output.size()[2:], mode='bilinear', align_corners=False)
        loss, mse, kld = loss_function(output, data, mu, logvar, beta)
        
        # Calculate absolute difference between ground truth and predicted reconstructions
        reconstruction_diff = torch.abs(output - data)
        
    return data, output, loss, mse, kld, reconstruction_diff


def inspect_normalized_tensor(normalized_tensor):
    # Check the min and max values across the entire tensor
    min_val = normalized_tensor.amin()
    max_val = normalized_tensor.amax()
    print(f"Overall Min value: {min_val.item()}, Overall Max value: {max_val.item()}")

    # Check the min and max values for each channel
    for i in range(normalized_tensor.shape[3]):
        channel_min = normalized_tensor[..., i].amin()
        channel_max = normalized_tensor[..., i].amax()
        print(f"Channel {i}: Min value: {channel_min.item()}, Max value: {channel_max.item()}")

    # Check if the value 1 exists in the tensor
    unique_values = torch.unique(normalized_tensor)
    print("Unique values in the tensor:")
    print(unique_values)

    # Plot the distribution of the values for each channel
    for i in range(normalized_tensor.shape[3]):
        channel_values = normalized_tensor[..., i].flatten().cpu().numpy()
        plt.hist(channel_values, bins=50, alpha=0.6, label=f'Channel {i}')
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of values in the normalized tensor')
    plt.show()

def find_and_visualize_coordinates(tensor, value=1.0, channel=0):
    # Find the indices where the value is equal to the specified value
    coordinates = (tensor == value).nonzero(as_tuple=False)
    
    if coordinates.numel() == 0:
        print(f"No occurrences of the value {value} found in the tensor.")
    else:
        print(f"Found {coordinates.size(0)} occurrences of the value {value}.")
        
        tensor_np = tensor.numpy()
        
        for idx in range(coordinates.size(0)):
            coords = coordinates[idx]
            batch, y, x, ch = coords.tolist()
            if ch == channel:
                plt.imshow(tensor_np[batch, :, :, channel], cmap='gray')
                plt.scatter([x], [y], color='red')
                plt.title(f"Image {batch}, Channel {channel}, Coordinate: ({y}, {x})")
                plt.show()