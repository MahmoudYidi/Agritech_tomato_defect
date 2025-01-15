import torch
import random

# Load the tensors from the specified directories
data_dir_test = "/mnt/c/Users/User/Desktop/Network_Isolation/KernelPCA/output/ful_test_tensor.pt"
hsi_tensor_test = torch.load(data_dir_test)

data_dir_anom = "/mnt/c/Users/User/Desktop/Network_Isolation/KernelPCA/output/ful_anom_tensor.pt"
hsi_tensor_anom = torch.load(data_dir_anom)

# Select the 1st, 2nd, and 5th samples from each tensor
selected_samples = [
    hsi_tensor_test[0],  # 1st sample from test tensor
    hsi_tensor_test[2],  # 2nd sample from test tensor
    hsi_tensor_test[9],  # 5th sample from test tensor
    hsi_tensor_anom[0],  # 1st sample from anomaly tensor
    hsi_tensor_anom[9],  # 2nd sample from anomaly tensor
    hsi_tensor_anom[17]   # 5th sample from anomaly tensor
]

# Randomly shuffle the selected samples
random.shuffle(selected_samples)

# Stack the shuffled samples into a new tensor
new_hsi_tensor = torch.stack(selected_samples)

# Define the path to save the new tensor
output_path = "shuffled_samples.pt"

# Save the tensor to the specified path
torch.save(new_hsi_tensor, output_path)

print(f"New shuffled tensor saved at: {output_path}")
