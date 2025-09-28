import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import torch.nn.functional as F
import cv2

# --- Configuration ---
DATA_DIR = '/kaggle/input/breast-histopathology-images'
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 128
EPOCHS = 30 # For demonstration, you might want to run for more epochs
LEARNING_RATE = 1e-4
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.15

# --- 1. Data Loading and Preprocessing ---
print("--- Starting Data Loading and Preprocessing ---")

# Find all image paths
all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)
print(f"Total number of images found: {len(all_image_paths)}")

# Extract labels from file paths
labels = [int(os.path.basename(path).split('_class')[1].split('.')[0]) for path in all_image_paths]

# Create a DataFrame for easier handling
data_df = pd.DataFrame({'path': all_image_paths, 'label': labels})

# Display class distribution
print("\nClass Distribution:")
print(data_df['label'].value_counts())

# Split data into training and testing sets
train_val_df, test_df = train_test_split(data_df, test_size=TEST_SPLIT_SIZE, stratify=data_df['label'], random_state=42)

# Split training data into training and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_SPLIT_SIZE, stratify=train_val_df['label'], random_state=42)

print(f"\nTraining set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# --- 2. Custom Dataset Class ---
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 3. Data Augmentation and Transforms ---
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create Dataset instances
train_dataset = BreastCancerDataset(train_df, transform=train_transform)
val_dataset = BreastCancerDataset(val_df, transform=val_test_transform)
test_dataset = BreastCancerDataset(test_df, transform=val_test_transform)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("\nDataLoaders created successfully.")

# --- 4. Model Architecture with Attention ---
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.relu(self.conv1(x))
        attn = self.sigmoid(self.conv2(attn))
        return x * attn, attn

class AttnResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(AttnResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove the final fully connected layer
        self.attention = Attention(num_features)
        self.classifier = nn.Linear(num_features, 2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        features = self.resnet.layer4(x)
        
        attended_features, attn_map = self.attention(features)
        out = nn.functional.relu(attended_features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out, attn_map

print("\n--- Model Architecture Defined ---")
model = AttnResNet50(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")


# --- 5. Weighted Loss Function and Optimizer ---
class_counts = data_df['label'].value_counts().sort_index()
weights = 1.0 / class_counts
class_weights = torch.FloatTensor(weights.values).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

print("\n--- Loss Function and Optimizer Initialized ---")
print(f"Class weights for loss function: {class_weights.cpu().numpy()}")

# --- 6. Training and Validation Loop ---
print("\n--- Starting Model Training ---")

best_val_accuracy = 0.0
model_save_path = 'best_breast_cancer_model.pth'

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        train_pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"\nEpoch {epoch+1}/{EPOCHS} - Training Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds)
    val_recall = recall_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path} with validation accuracy: {best_val_accuracy:.4f}")

print("\n--- Training Finished ---")

# --- 7. Testing and Evaluation ---
print("\n--- Loading Best Model for Testing ---")
model.load_state_dict(torch.load(model_save_path))
model.eval()

test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    test_pbar = tqdm(test_loader, desc="[Testing]")
    for inputs, labels in test_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_probs.extend(probabilities.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds)
test_recall = recall_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)

print("\n--- Final Test Results ---")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()

# --- 8. Visualize Attention ---
print("\n--- Visualizing Attention Maps ---")

# Find the 5 most confident correct predictions
test_df['predicted_label'] = test_preds
test_df['true_label'] = test_labels
test_df['probability'] = [p[l] for p, l in zip(test_probs, test_preds)]

correctly_predicted_df = test_df[test_df['predicted_label'] == test_df['true_label']]
top_5_confident_samples = correctly_predicted_df.nlargest(5, 'probability')

def visualize_attention(model, image_path, true_label, predicted_label, probability):
    """
    Generates and displays an attention map for a given image.
    """
    # Pre-process the image
    transform = val_test_transform
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get the attention map
    model.eval()
    with torch.no_grad():
        _, attn_map = model(image_tensor)

    # Post-process for visualization
    attn_map = attn_map.squeeze().cpu().numpy()
    attn_map = cv2.resize(attn_map, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

    # Normalize the attention map
    attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
    attn_map = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)

    # Overlay the attention map on the original image
    original_image = np.array(image.resize(IMAGE_SIZE))
    superimposed_img = cv2.addWeighted(original_image, 0.6, attn_map, 0.4, 0)

    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(attn_map)
    plt.title('Attention Map')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Superimposed Image')
    plt.axis('off')
    
    plt.suptitle(f'True Label: {true_label} | Predicted Label: {predicted_label} | Confidence: {probability:.4f}', fontsize=16)
    plt.show()

for index, row in top_5_confident_samples.iterrows():
    visualize_attention(model, row['path'], row['true_label'], row['predicted_label'], row['probability'])