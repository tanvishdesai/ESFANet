# -*- coding: utf-8 -*-
"""
EFSANetV2 (Revised): Breast Cancer Histopathology Classification with a 
Hybrid Attention Model built on a Pretrained Backbone.

This script implements a revised workflow for classifying breast cancer
histopathology images.

Key Revisions:
1.  Replaces the from-scratch CNN with a powerful, pretrained DenseNet121 backbone
    to leverage transfer learning.
2.  Increases image resolution to 96x96 to preserve fine-grained details.
3.  Adapts the novel Edge-Frequency Attention and Transformer modules to operate on the
    rich features extracted by the pretrained backbone.
4.  Aligns hyperparameters (batch size, learning rate) with proven successful baselines.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import glob

warnings.filterwarnings("ignore")

# --- Configuration ---
# <--- MODIFICATION: Aligned hyperparameters with successful baselines
DATA_DIR = 'breakhis'  # Updated to match local dataset structure
IMAGE_SIZE = 96
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.15
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")
print(f"PyTorch Version: {torch.__version__}")

# --- 1. Data Loading and Preprocessing ---

class HistopathologyDataset(Dataset):
    """Custom PyTorch Dataset for histopathology images."""
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
            
        return image, torch.tensor(label, dtype=torch.long)

# --- 2. Model Architecture: EFSANet (Revised) ---

# 2.1 Custom Attention Module: Edge-Frequency Attention (Unchanged)
class EdgeFrequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeFrequencyAttention, self).__init__()
        # Edge-Aware Spatial Attention
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_x = nn.Parameter(sobel_kernel_x.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_kernel_y.repeat(in_channels, 1, 1, 1), requires_grad=False)
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # Frequency-Aware Channel Attention
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False), # Using a higher reduction ratio
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial Attention
        grayscale_x = torch.mean(x, dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)
        edge_x = F.conv2d(grayscale_x, self.sobel_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(grayscale_x, self.sobel_y, padding=1, groups=x.size(1))
        edge_map = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        spatial_attention = self.edge_conv(edge_map)

        # Channel Attention
        fft_map = torch.fft.fft2(x, norm='ortho')
        fft_amp = torch.abs(fft_map)
        channel_vec = F.adaptive_avg_pool2d(fft_amp, 1).squeeze(-1).squeeze(-1)
        channel_attention = self.channel_fc(channel_vec).unsqueeze(-1).unsqueeze(-1)

        # Combine attentions with a residual connection
        x_att = x * spatial_attention * channel_attention
        return x + x_att

# 2.2 Transformer Encoder Block (Unchanged)
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 2.3 The Full Revised EFSANet Model
class EFSANet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EFSANet, self).__init__()
        
        # <--- MODIFICATION: Use a powerful pretrained DenseNet121 backbone
        base_model = models.densenet121(pretrained=pretrained)
        self.backbone = base_model.features
        
        # Get the number of output features from the backbone
        num_features = base_model.classifier.in_features # This is 1024 for DenseNet121
        
        # <--- MODIFICATION: Adapt your novel components to the backbone's output
        self.attention = EdgeFrequencyAttention(num_features)
        
        # For a 96x96 input, DenseNet121 features are 3x3. 3*3 = 9 patches.
        # The feature dimension (d_model) is 1024.
        self.pos_embedding = nn.Parameter(torch.randn(1, 9, num_features)) 
        self.transformer_block = TransformerEncoderBlock(d_model=num_features, nhead=8)

        # <--- MODIFICATION: Classifier now takes the rich 1024-dim features
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. Extract rich features using the pretrained backbone
        features = self.backbone(x)
        
        # 2. Apply your novel attention mechanism
        attended_features = self.attention(features)
        
        # 3. Prepare for Transformer
        b, c, h, w = attended_features.shape
        transformer_input = attended_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        
        # Add positional embedding
        transformer_input = transformer_input + self.pos_embedding
        
        # 4. Process with Transformer
        transformer_output = self.transformer_block(transformer_input)

        # 5. Classify using the mean of the transformer outputs
        final_vector = transformer_output.mean(dim=1)
        logits = self.classifier(final_vector)
            
        return logits

# --- 3. Focal Loss for Class Imbalance (Unchanged) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# --- 4. Training and Evaluation Logic (Unchanged) ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            print("!!! LOSS IS NaN. STOPPING TRAINING. !!!")
            return np.nan, np.nan
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return epoch_loss, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

# --- 5. Main Execution Block ---

def get_label_from_breakhis_path(path):
    """
    Extract label from BreakHis dataset path structure.
    BreakHis naming convention: SOB_[B|M]_[subtype]-[patient]-[slide]-[magnification]-[patch].png
    Where B = Benign, M = Malignant
    """
    filename = os.path.basename(path)

    # Extract the category indicator (B for benign, M for malignant)
    # Format: SOB_B_... or SOB_M_...
    parts = filename.split('_')
    if len(parts) >= 2:
        category = parts[1]  # This should be 'B' or 'M'
        if category == 'B':
            return 0  # Benign
        elif category == 'M':
            return 1  # Malignant

    # Fallback: try to determine from directory structure
    path_parts = path.split(os.sep)
    for part in path_parts:
        if part.lower() == 'benign':
            return 0
        elif part.lower() == 'malignant':
            return 1

    # If we can't determine, print warning and default to benign
    print(f"Warning: Could not determine label for {path}")
    return 0

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Step 1: Loading and splitting BreakHis data...")
    all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)
    print(f"Found {len(all_image_paths)} images")

    # Filter to only include valid image paths (exclude any non-png files that might have been matched)
    all_image_paths = [path for path in all_image_paths if path.lower().endswith('.png')]

    # Extract labels using BreakHis naming convention
    labels = [get_label_from_breakhis_path(path) for path in all_image_paths]

    data_df = pd.DataFrame({'path': all_image_paths, 'label': labels})

    # Print class distribution
    class_counts = data_df['label'].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")

    train_val_df, test_df = train_test_split(data_df, test_size=TEST_SPLIT_SIZE, stratify=data_df['label'], random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_SPLIT_SIZE, stratify=train_val_df['label'], random_state=RANDOM_SEED)

    print(f"Dataset split:")
    print(f"  - Training:   {len(train_df)} images")
    print(f"  - Validation: {len(val_df)} images")
    print(f"  - Testing:    {len(test_df)} images")

    # Image transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = HistopathologyDataset(train_df, transform=data_transforms['train'])
    val_dataset = HistopathologyDataset(val_df, transform=data_transforms['val'])
    test_dataset = HistopathologyDataset(test_df, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\nStep 2: Initializing the revised EFSANet model...")
    model = EFSANet(num_classes=2).to(DEVICE)
    
    class_counts = train_df['label'].value_counts()
    benign_count = class_counts.get(0, 0)
    malignant_count = class_counts.get(1, 0)
    alpha = benign_count / (benign_count + malignant_count + 1e-6)  # Alpha for benign class
    print(f"Class distribution in training set: Benign: {benign_count}, Malignant: {malignant_count}")
    print(f"Using Focal Loss with calculated alpha: {alpha:.4f} (for benign class)")
    
    criterion = FocalLoss(alpha=alpha, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print("\nStep 3: Starting model training...")
    best_f1_score = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        if np.isnan(train_loss):
            print("Training halted due to NaN loss.")
            break
            
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} Summary | Time: {elapsed_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val F1:   {val_f1:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

        if val_f1 > best_f1_score:
            print(f"Validation F1 improved from {best_f1_score:.4f} to {val_f1:.4f}. Saving model...")
            best_f1_score = val_f1
            torch.save(model.state_dict(), 'best_model_revised.pth')

    print("\nTraining finished.")
    
    if os.path.exists('best_model_revised.pth'):
        print("\nStep 4: Performing final evaluation on the test set...")
        model.load_state_dict(torch.load('best_model_revised.pth'))
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, DEVICE)
        
        print("\n--- Test Set Results ---")
        print(f"  - Accuracy:  {test_acc:.4f}")
        print(f"  - Precision: {test_prec:.4f} (Macro)")
        print(f"  - Recall:    {test_rec:.4f} (Macro)")
        print(f"  - F1 Score:  {test_f1:.4f} (Macro)")
        
        # Detailed Classification Report
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Generating test report"):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
    else:
        print("\nSkipping final evaluation because no model was saved.")

if __name__ == '__main__':
    main()