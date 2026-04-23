
# In[1]:
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Set base directory paths
input_base_dir = 'OASIS_2'  # Replace with the actual path to your OASIS_2 folder
output_base_dir = 'Output'  # Replace with the desired output directory

# List of class names (directories)
class_names = ['Converted','Demented', 'Nondemented']

# Function to save axial, coronal, and sagittal views for each .nii file
def save_slices_for_nii(nii_file_path, output_class_dir, file_index):
    # Load the .nii file
    nifti_image = nib.load(nii_file_path)
    image_data = nifti_image.get_fdata()

    # Remove the extra dimension (size 1)
    image_data = np.squeeze(image_data)

    # Create output directories for axial, coronal, and sagittal views
    axial_dir = os.path.join(output_class_dir, 'axial')
    coronal_dir = os.path.join(output_class_dir, 'coronal')
    sagittal_dir = os.path.join(output_class_dir, 'sagittal')

    os.makedirs(axial_dir, exist_ok=True)
    os.makedirs(coronal_dir, exist_ok=True)
    os.makedirs(sagittal_dir, exist_ok=True)

    # Get the number of axial slices (along the Z-axis)
    num_axial_slices = image_data.shape[2]

    # Save all axial slices (slices along the Z-axis)
    for i in range(num_axial_slices):
        axial_slice = image_data[:, :, i]
        plt.imsave(os.path.join(axial_dir, f'{file_index}_{i+1}.png'), axial_slice, cmap='gray', format='png')

    # Resize and save coronal slices (slices along the Y-axis)
    num_coronal_slices = image_data.shape[1]
    step_size_coronal = max(1, num_coronal_slices // num_axial_slices)
    for i in range(num_axial_slices):
        coronal_slice = image_data[:, i * step_size_coronal, :]
        plt.imsave(os.path.join(coronal_dir, f'{file_index}_{i+1}.png'), coronal_slice, cmap='gray', format='png')

    # Resize and save sagittal slices (slices along the X-axis)
    num_sagittal_slices = image_data.shape[0]
    step_size_sagittal = max(1, num_sagittal_slices // num_axial_slices)
    for i in range(num_axial_slices):
        sagittal_slice = image_data[i * step_size_sagittal, :, :]
        plt.imsave(os.path.join(sagittal_dir, f'{file_index}_{i+1}.png'), sagittal_slice, cmap='gray', format='png')

# Loop through each class (Demented, Nondemented, etc.)
for class_name in class_names:
    input_class_dir = os.path.join(input_base_dir, class_name)
    output_class_dir = os.path.join(output_base_dir, class_name)

    # Loop through each .nii file in the class folder
    nii_files = [f for f in os.listdir(input_class_dir) if f.endswith('.nii')]
    
    for file_index, nii_file in enumerate(nii_files, start=1):
        nii_file_path = os.path.join(input_class_dir, nii_file)
        
        # Save slices for the .nii file
        save_slices_for_nii(nii_file_path, output_class_dir, file_index)

print("Processing complete!")

# In[2]:
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# In[3]:
class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # List all classes
        self.classes = sorted(os.listdir(root_dir))
        
        # Initialize lists to hold file paths and labels
        self.axial_paths = []
        self.coronal_paths = []
        self.sagittal_paths = []
        self.labels = []

        # Iterate through each class directory
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            
            # Load axial, coronal, and sagittal images
            axial_dir = os.path.join(class_dir, 'axial')
            coronal_dir = os.path.join(class_dir, 'coronal')
            sagittal_dir = os.path.join(class_dir, 'sagittal')
            
            # Store paths and labels of axial
            for img_name in os.listdir(axial_dir):
                self.axial_paths.append(os.path.join(axial_dir, img_name))
                self.sagittal_paths.append(os.path.join(sagittal_dir, img_name))
                self.coronal_paths.append(os.path.join(coronal_dir, img_name))
                self.labels.append(label)
                

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load images
        axial_image = Image.open(self.axial_paths[idx]).convert("RGB")
        coronal_image = Image.open(self.coronal_paths[idx]).convert("RGB")
        sagittal_image = Image.open(self.sagittal_paths[idx]).convert("RGB")
        
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            axial_image = self.transform(axial_image)
            coronal_image = self.transform(coronal_image)
            sagittal_image = self.transform(sagittal_image)

        return axial_image, coronal_image, sagittal_image, label

# In[4]:
# Define your image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# In[5]:
dataset = AlzheimerDataset(root_dir='2D Datasets/OASIS_1/', transform=transform)

# In[6]:
# Creating a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# In[7]:
# Example of iterating through the dataloader
for axial, coronal, sagittal, labels in dataloader:
    print(axial.shape, coronal.shape, sagittal.shape, labels.shape)
    break

# ### Model

# In[8]:
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# In[9]:
# Define the Region-Specific Attention Module (simplified as a placeholder)
class RegionSimamModule(nn.Module):
    def __init__(self):
        super(RegionSimamModule, self).__init__()
        # Placeholder for the attention module, to be replaced with the actual implementation
        self.attention = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
    def forward(self, x, region_mask=None):
        # Apply attention mechanism here (simplified for now)
        return self.attention(x)

class MultiLevelAttentionNet(torch.nn.Module):
    def __init__(self):
        super(MultiLevelAttentionNet, self).__init__()

        # Modify the input channels to 960
        self.encoder_local = nn.Sequential(
            nn.Conv2d(960, 64, kernel_size=3, padding=1),  # Change input channels to 960
            nn.ReLU(),
            RegionSimamModule()  # Region-Specific Attention
        )

        # Middle layers: Capture more global features
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Higher layers: Capture global brain structure
        self.encoder_global = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x, region_mask=None):
        # Pass through local feature extractor
        local_features = self.encoder_local(x)

        # Pass through middle layers (global context)
        middle_features = self.middle(local_features)

        # Pass through higher layers
        global_features = self.encoder_global(middle_features)

        return global_features

# Channel Attention Module (CAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #print('avg_pool:', self.avg_pool.shape)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #print('max_pool:', self.max_pool.shape)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        #print('max_pool_fc1:', self.fc1.shape)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)
        #print('max_pool_fc2:', self.fc2.shape)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        #print('avg_out:', self.avg_out.shape)
        #print('max_out:', self.max_out.shape)
        out = avg_out + max_out
        #print('out:', self.out.shape)
        return torch.sigmoid(out) * x

# Spatial Attention Module (SAM)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        #print('SpatialAttention Conv 1:', self.conv1.shape)
        self.sigmoid = nn.Sigmoid()
        #print('SpatialAttention Sigmoid:', self.sigmoid.shape)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print('SpatialAttention avg_out:', avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #print('SpatialAttention max_out:', max_out.shape)
        out = torch.cat([avg_out, max_out], dim=1)
        #print('SpatialAttention Concat:', out.shape)
        out = self.conv1(out)
        #print('SpatialAttention out Conv:', out.shape)
        return self.sigmoid(out) * x

# Residual CBAM Module
class ResidualCBAM(nn.Module):
    def __init__(self, in_channels):
        super(ResidualCBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)  # Adjust to 256 if necessary
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x + residual  # Residual connection

class MultiViewMobileNetV3WithResidualCBAM(nn.Module):
    def __init__(self, num_classes):
        super(MultiViewMobileNetV3WithResidualCBAM, self).__init__()

        # Load pretrained MobileNetV3 for each view (axial, sagittal, coronal)
        self.mobilenetv3_axial = models.mobilenet_v3_large(pretrained=True)
        self.mobilenetv3_sagittal = models.mobilenet_v3_large(pretrained=True)
        self.mobilenetv3_coronal = models.mobilenet_v3_large(pretrained=True)

        # Define the Multi-Level Attention Network
        self.attention_net = MultiLevelAttentionNet()

        # Residual CBAM module (applied after feature extraction from MobileNetV3)
        self.cbam_axial = ResidualCBAM(960)  # Apply CBAM before reducing channels
        self.cbam_sagittal = ResidualCBAM(960)
        self.cbam_coronal = ResidualCBAM(960)

        # 1x1 Convolution to adjust the number of channels before passing to 3D conv
        self.conv_1x1 = nn.Conv2d(768, 2880, kernel_size=1, stride=1, padding=0)

        # Define the 3D convolutional layers
        self.conv3d_1 = nn.Conv3d(in_channels=2880, out_channels=512, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_3 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), padding=1)

        # Define batch normalization
        self.bn1 = nn.BatchNorm3d(512)
        self.bn2 = nn.BatchNorm3d(256)
        self.bn3 = nn.BatchNorm3d(128)

        # Define a pooling layer
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # [batch_size, channels, 1, 1, 1]

        # Define the fully connected layer
        self.fc = nn.Linear(128, num_classes)  # 128 is the output channels from last conv layer

    def forward(self, axial, sagittal, coronal):
        # Extract features using MobileNetV3 for all views
        axial_features = self.mobilenetv3_axial.features(axial)
        sagittal_features = self.mobilenetv3_sagittal.features(sagittal)
        coronal_features = self.mobilenetv3_coronal.features(coronal)

        # Apply Residual CBAM to each feature map BEFORE applying attention or 3D convolutions
        axial_features = self.cbam_axial(axial_features)
        sagittal_features = self.cbam_sagittal(sagittal_features)
        coronal_features = self.cbam_coronal(coronal_features)

        # Apply Multi-Level Attention Network to each feature map
        axial_features = self.attention_net(axial_features)
        sagittal_features = self.attention_net(sagittal_features)
        coronal_features = self.attention_net(coronal_features)

        # Concatenate the feature maps from the three views
        # Ensure that the resulting tensor has the correct shape: [batch_size, 768, height, width]
        combined_features = torch.cat([axial_features, sagittal_features, coronal_features], dim=1)  # Shape: [batch_size, 768, height, width]

        # Apply 1x1 convolution to adjust channels to 2880 for 3D convolution
        adjusted_features = self.conv_1x1(combined_features)  # Shape: [batch_size, 2880, height, width]

        # Reshaping the adjusted features to add a depth dimension (for 3D convolution)
        adjusted_features = adjusted_features.unsqueeze(2)  # Shape: [batch_size, 2880, 1, height, width]

        # Pass through 3D convolutional layers with batch normalization
        x = F.relu(self.bn1(self.conv3d_1(adjusted_features)))  # Shape: [batch_size, 512, 1, height, width]
        x = F.relu(self.bn2(self.conv3d_2(x)))  # Shape: [batch_size, 256, 1, height, width]
        x = F.relu(self.bn3(self.conv3d_3(x)))  # Shape: [batch_size, 128, 1, height, width]

        # Apply pooling
        x = self.pool(x)  # Shape: [batch_size, 128, 1, 1, 1]

        # Flatten
        x = x.view(x.size(0), -1)  # Shape: [batch_size, 128]

        # Pass through the fully connected layer
        x = self.fc(x)
        return x

# In[10]:
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# In[11]:
num_classes = 4  # Example: 3 classes in the classification problem
model = MultiViewMobileNetV3WithResidualCBAM(num_classes=num_classes).to(device)
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ### Training 

# In[12]:
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

num_epochs = 100
for epoch in range(num_epochs):
    start_time = time.time()  # Start the timer
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    all_labels = []
    all_preds = []

    # Use tqdm to display a progress bar for each epoch
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for axial_images, sagittal_images, coronal_images, labels in pbar:
            # Move data to GPU if available
            axial_images = axial_images.to(device)
            sagittal_images = sagittal_images.to(device)
            coronal_images = coronal_images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(axial_images, sagittal_images, coronal_images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track predictions and true labels
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Update the progress bar description with loss
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    epoch_loss = running_loss / len(dataloader)
    
    # Print epoch details along with the time taken
    epoch_time = time.time() - start_time  # End the timer
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, '
          f'Time: {epoch_time:.2f}s')

# ### Testing of Model

# In[13]:
def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to calculate gradients during testing
        for axial_images, sagittal_images, coronal_images, labels in test_loader:  # Assuming the DataLoader returns images and labels
            axial_images = axial_images.to(device)
            sagittal_images = sagittal_images.to(device)
            coronal_images = coronal_images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(axial_images, sagittal_images, coronal_images)
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

# In[14]:
test_dataset = AlzheimerDataset(root_dir='2D Datasets/OASIS_1/', transform=None)
# Creating a DataLoader
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# In[15]:
# Assuming you have a DataLoader `test_loader`
accuracy, precision, recall, f1 = test_model(model, test_dataloader, device)

# In[16]:
import matplotlib.pyplot as plt

# Sample data: Replace these with your actual arrays
epochs = list(range(1, 11))  # Assuming 20 epochs
training_loss = [0.9741, 0.6481, 0.4004, 0.2482, 0.1604, 0.1217, 0.0988, 0.0850, 0.0742, 0.0682]
training_accuracy = [0.5319, 0.7070, 0.8353, 0.9054, 0.9413, 0.9568, 0.9649, 0.9701, 0.9740, 0.9757]

# Plotting
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss', color='red', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, label='Training Accuracy', color='blue', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# In[17]:
