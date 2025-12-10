# -*- coding: utf-8 -*-
"""
EFSANetV2 (Revised): Breast Cancer Histopathology Classification with 
Ablation Study and Attention Visualization Support

Key Features:
1. Configurable attention modes: 'full', 'edge_only', 'frequency_only', 'none'
2. Attention visualization for paper figures
3. Component ablation study support
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
import concurrent.futures

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_DIR = '/kaggle/input/breast-histopathology-images'
IMAGE_SIZE = 96
BATCH_SIZE = 128 
EPOCHS = 30 
LEARNING_RATE = 1e-4
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.15
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============== ABLATION STUDY CONFIGURATION ==============
# Options: 'full', 'edge_only', 'frequency_only', 'none'
ATTENTION_MODE = 'full'  # <-- CHANGE THIS FOR ABLATION STUDY
# ==========================================================

# ============== VISUALIZATION CONFIGURATION ==============
ENABLE_VISUALIZATION = True  # Set to True to generate attention visualizations
NUM_VIZ_SAMPLES = 5  # Number of samples to visualize
# =========================================================

print(f"Using device: {DEVICE}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Attention Mode: {ATTENTION_MODE}")
print(f"Visualization Enabled: {ENABLE_VISUALIZATION}")

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

# --- 2. Model Architecture: EFSANet (Revised with Ablation Support) ---

class EdgeFrequencyAttention(nn.Module):
    """
    Edge-Frequency Attention Module with Ablation Study Support
    
    Args:
        in_channels: Number of input channels
        mode: 'full', 'edge_only', 'frequency_only', or 'none'
    """
    def __init__(self, in_channels, mode='full'):
        super(EdgeFrequencyAttention, self).__init__()
        self.mode = mode
        
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
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # For visualization
        self.last_edge_map = None
        self.last_channel_vec = None
        self.last_spatial_attention = None
        self.last_channel_attention = None

    def forward(self, x):
        if self.mode == 'none':
            return x
        
        # Spatial Attention (Edge-based)
        spatial_attention = None
        if self.mode in ['full', 'edge_only']:
            grayscale_x = torch.mean(x, dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)
            edge_x = F.conv2d(grayscale_x, self.sobel_x, padding=1, groups=x.size(1))
            edge_y = F.conv2d(grayscale_x, self.sobel_y, padding=1, groups=x.size(1))
            edge_map = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
            spatial_attention = self.edge_conv(edge_map)
            
            # Store for visualization
            self.last_edge_map = edge_map.detach()
            self.last_spatial_attention = spatial_attention.detach()

        # Channel Attention (Frequency-based)
        channel_attention = None
        if self.mode in ['full', 'frequency_only']:
            fft_map = torch.fft.fft2(x, norm='ortho')
            fft_amp = torch.abs(fft_map)
            channel_vec = F.adaptive_avg_pool2d(fft_amp, 1).squeeze(-1).squeeze(-1)
            channel_attention = self.channel_fc(channel_vec).unsqueeze(-1).unsqueeze(-1)
            
            # Store for visualization
            self.last_channel_vec = channel_vec.detach()
            self.last_channel_attention = channel_attention.detach()

        # Apply attention based on mode
        if self.mode == 'full':
            x_att = x * spatial_attention * channel_attention
        elif self.mode == 'edge_only':
            x_att = x * spatial_attention
        elif self.mode == 'frequency_only':
            x_att = x * channel_attention
        else:
            x_att = x
            
        return x + x_att

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

class AgenticEFSANet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, attention_mode='full'):
        super(AgenticEFSANet, self).__init__()
        
        # 1. Backbone
        base_model = models.densenet121(pretrained=pretrained)
        self.backbone = base_model.features
        num_features = base_model.classifier.in_features 
        
        # 2. Attention (Your Novelty) with ablation support
        self.attention = EdgeFrequencyAttention(num_features, mode=attention_mode)
        
        # 3. Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Agentic Classifier Head (Evidence)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)
        attended_features = self.attention(features)
        
        pooled = self.pool(attended_features)
        flat = torch.flatten(pooled, 1)
        
        evidence = self.classifier(flat)
        return evidence

# --- 3. Evidential Deep Learning Loss (EDL) ---

def kl_divergence(alpha, num_classes, device=None):
    if device is None:
        device = alpha.device
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term

def edl_loss_function(output, target, epoch, num_classes, annealing_step, device):
    evidence = output
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    belief = alpha / S

    y = F.one_hot(target, num_classes).float()
    loss_mse = torch.sum((y - belief) ** 2 + ((alpha * (S - alpha)) / (S * S * (S + 1))), dim=1)
    
    annealing_coef = min(1, epoch / annealing_step)
    
    alpha_tilde = y + (1 - y) * alpha
    kl = kl_divergence(alpha_tilde, num_classes, device=device)
    
    return torch.mean(loss_mse + annealing_coef * kl.squeeze())

class EDLLoss(nn.Module):
    def __init__(self, num_classes=2, annealing_step=10, device='cuda'):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
        self.epoch = 0

    def forward(self, output, target):
        return edl_loss_function(output, target, self.epoch, self.num_classes, self.annealing_step, self.device)

# --- 4. Attention Visualization Functions ---

def denormalize_image(img_tensor):
    """Denormalize image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def visualize_attention_components(model, dataloader, device, num_samples=5, save_dir='attention_viz'):
    """
    Visualize attention mechanism components for paper figures
    
    Creates visualizations showing:
    - Original image
    - Edge map (Sobel output)
    - Frequency channel importance
    - Spatial attention map
    - Final attended features
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_visualized = 0
    
    print(f"\nGenerating attention visualizations...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if samples_visualized >= num_samples:
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass to populate attention maps
            _ = model(inputs)
            
            # Get attention module
            attention_module = model.attention
            
            for i in range(min(inputs.size(0), num_samples - samples_visualized)):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Attention Visualization - Sample {samples_visualized + 1} (Label: {labels[i].item()})', 
                            fontsize=16, fontweight='bold')
                
                # 1. Original Image
                orig_img = denormalize_image(inputs[i])
                axes[0, 0].imshow(orig_img)
                axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                # 2. Edge Map (if available)
                if attention_module.last_edge_map is not None and ATTENTION_MODE in ['full', 'edge_only']:
                    edge_map = attention_module.last_edge_map[i].mean(dim=0).cpu().numpy()
                    im1 = axes[0, 1].imshow(edge_map, cmap='hot')
                    axes[0, 1].set_title('Edge Map (Sobel)', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
                else:
                    axes[0, 1].text(0.5, 0.5, 'Edge Attention\nDisabled', 
                                   ha='center', va='center', fontsize=14)
                    axes[0, 1].axis('off')
                
                # 3. Spatial Attention Map
                if attention_module.last_spatial_attention is not None and ATTENTION_MODE in ['full', 'edge_only']:
                    spatial_att = attention_module.last_spatial_attention[i].mean(dim=0).cpu().numpy()
                    im2 = axes[0, 2].imshow(spatial_att, cmap='jet')
                    axes[0, 2].set_title('Spatial Attention Weight', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
                else:
                    axes[0, 2].text(0.5, 0.5, 'Spatial Attention\nDisabled', 
                                   ha='center', va='center', fontsize=14)
                    axes[0, 2].axis('off')
                
                # 4. Frequency Channel Importance
                if attention_module.last_channel_vec is not None and ATTENTION_MODE in ['full', 'frequency_only']:
                    channel_vec = attention_module.last_channel_vec[i].cpu().numpy()
                    axes[1, 0].bar(range(len(channel_vec)), channel_vec)
                    axes[1, 0].set_title('Frequency Channel Importance', fontsize=12, fontweight='bold')
                    axes[1, 0].set_xlabel('Channel Index')
                    axes[1, 0].set_ylabel('Importance')
                    axes[1, 0].grid(True, alpha=0.3)
                else:
                    axes[1, 0].text(0.5, 0.5, 'Frequency Attention\nDisabled', 
                                   ha='center', va='center', fontsize=14)
                    axes[1, 0].axis('off')
                
                # 5. Channel Attention Weight
                if attention_module.last_channel_attention is not None and ATTENTION_MODE in ['full', 'frequency_only']:
                    channel_att = attention_module.last_channel_attention[i].squeeze().cpu().numpy()
                    axes[1, 1].bar(range(len(channel_att)), channel_att)
                    axes[1, 1].set_title('Channel Attention Weight', fontsize=12, fontweight='bold')
                    axes[1, 1].set_xlabel('Channel Index')
                    axes[1, 1].set_ylabel('Weight')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Channel Attention\nDisabled', 
                                   ha='center', va='center', fontsize=14)
                    axes[1, 1].axis('off')
                
                # 6. Combined Attention Overlay
                if ATTENTION_MODE != 'none':
                    axes[1, 2].imshow(orig_img)
                    if attention_module.last_spatial_attention is not None:
                        spatial_overlay = attention_module.last_spatial_attention[i].mean(dim=0).cpu().numpy()
                        # Resize to match image dimensions
                        from scipy.ndimage import zoom
                        h, w = orig_img.shape[:2]
                        spatial_overlay_resized = zoom(spatial_overlay, 
                                                      (h/spatial_overlay.shape[0], w/spatial_overlay.shape[1]), 
                                                      order=1)
                        axes[1, 2].imshow(spatial_overlay_resized, cmap='jet', alpha=0.5)
                    axes[1, 2].set_title('Attention Overlay', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                else:
                    axes[1, 2].imshow(orig_img)
                    axes[1, 2].set_title('No Attention Applied', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'attention_sample_{samples_visualized + 1}_label_{labels[i].item()}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                samples_visualized += 1
                print(f"  Saved visualization {samples_visualized}/{num_samples}")
                
                if samples_visualized >= num_samples:
                    break
    
    print(f"All visualizations saved to '{save_dir}/' directory")

# --- 5. Training and Evaluation Logic ---

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

def evaluate_agentic(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    uncertainties = []
    correctness = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            evidence = outputs
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            belief = alpha / S
            uncertainty = 2 / S
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            uncertainties.extend(uncertainty.cpu().numpy().flatten())
            
            batch_correct = preds.eq(labels).cpu().numpy()
            correctness.extend(batch_correct)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1, uncertainties, correctness

# --- 6. Main Execution Block ---
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Step 1: Loading and splitting data (Parallel)...")
    
    def scan_subdir(subdir):
        """Scans a subdirectory for PNG images and returns their paths."""
        return list(glob.iglob(os.path.join(subdir, '**', '*.png'), recursive=True))

    all_image_paths = []
    
    try:
        subdirs = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]
    except FileNotFoundError:
        print(f"Error: DATA_DIR '{DATA_DIR}' not found.")
        return

    print(f"  Found {len(subdirs)} subdirectories to scan.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(scan_subdir, subdir): subdir for subdir in subdirs}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(subdirs), desc="Scanning Dirs", unit="dir"):
            all_image_paths.extend(future.result())

    if not all_image_paths and not subdirs:
        print("  No subdirectories found, falling back to global scan...")
        all_image_paths = list(glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True))

    print(f"  Total images found: {len(all_image_paths)}")

    labels = []
    print("  Extracting labels...")
    for path in tqdm(all_image_paths, desc="Extracting Labels", unit="img"):
        try:
            label_str = os.path.basename(path).split('_class')[1].split('.')[0]
            labels.append(int(label_str))
        except (IndexError, ValueError):
            print(f"Warning: Could not extract label from {path}")
            labels.append(0)

    if len(labels) != len(all_image_paths):
        print("Error: Label count mismatch!")
        return

    data_df = pd.DataFrame({'path': all_image_paths, 'label': labels})

    train_val_df, test_df = train_test_split(data_df, test_size=TEST_SPLIT_SIZE, stratify=data_df['label'], random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_SPLIT_SIZE, stratify=train_val_df['label'], random_state=RANDOM_SEED)

    print(f"Dataset split:")
    print(f"  - Training:   {len(train_df)} images")
    print(f"  - Validation: {len(val_df)} images")
    print(f"  - Testing:    {len(test_df)} images")

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

    print(f"\nStep 2: Initializing EFSANet model with attention mode: {ATTENTION_MODE}...")
    model = AgenticEFSANet(num_classes=2, attention_mode=ATTENTION_MODE).to(DEVICE)
    
    class_counts = np.bincount(train_df['label'])
    print(f"Class distribution in training set: Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    print(f"Using Evidential Deep Learning (EDL) Loss")
    
    criterion = EDLLoss(num_classes=2, annealing_step=10, device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print("\nStep 3: Starting model training...")
    best_f1_score = 0.0
    
    for epoch in range(EPOCHS):
        criterion.epoch = epoch
        start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        if np.isnan(train_loss):
            print("Training halted due to NaN loss.")
            break
            
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate_agentic(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} Summary | Time: {elapsed_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val F1:   {val_f1:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")

        if val_f1 > best_f1_score:
            print(f"Validation F1 improved from {best_f1_score:.4f} to {val_f1:.4f}. Saving model...")
            best_f1_score = val_f1
            model_name = f'best_model_{ATTENTION_MODE}.pth'
            torch.save(model.state_dict(), model_name)

    print("\nTraining finished.")
    
    model_name = f'best_model_{ATTENTION_MODE}.pth'
    if os.path.exists(model_name):
        print("\nStep 4: Performing final evaluation on the test set...")
        model.load_state_dict(torch.load(model_name))

        test_loss, test_acc, test_prec, test_rec, test_f1, u_scores, correct_mask = evaluate_agentic(model, test_loader, criterion, DEVICE)

        # Uncertainty Analysis
        import seaborn as sns

        u_scores = np.array(u_scores)
        correct_mask = np.array(correct_mask)

        u_correct = u_scores[correct_mask == 1]
        u_wrong = u_scores[correct_mask == 0]

        print(f"Average Uncertainty on Correct Predictions: {np.mean(u_correct):.4f}")
        print(f"Average Uncertainty on Wrong Predictions:   {np.mean(u_wrong):.4f}")

        plt.figure(figsize=(10, 6))
        sns.kdeplot(u_correct, fill=True, label='Correct Predictions', color='green')
        sns.kdeplot(u_wrong, fill=True, label='Wrong Predictions', color='red')
        plt.title(f"Uncertainty Distribution ({ATTENTION_MODE.replace('_', ' ').title()})")
        plt.xlabel("Uncertainty Score (u)")
        plt.legend()
        plt.savefig(f"uncertainty_plot_{ATTENTION_MODE}.png")
        print(f"Uncertainty plot saved as uncertainty_plot_{ATTENTION_MODE}.png")
                
        print("\n--- Test Set Results ---")
        print(f"  - Accuracy:  {test_acc:.4f}")
        print(f"  - Precision: {test_prec:.4f} (Macro)")
        print(f"  - Recall:    {test_rec:.4f} (Macro)")
        print(f"  - F1 Score:  {test_f1:.4f} (Macro)")
        
        # Classification Report
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
        print(classification_report(all_labels, all_preds, target_names=['Class 0 (No IDC)', 'Class 1 (IDC)']))
        
        # Generate Attention Visualizations
        if ENABLE_VISUALIZATION and ATTENTION_MODE != 'none':
            print("\n" + "="*60)
            print("Step 5: Generating Attention Visualizations for Paper")
            print("="*60)
            visualize_attention_components(model, test_loader, DEVICE, 
                                          num_samples=NUM_VIZ_SAMPLES,
                                          save_dir=f'attention_viz_{ATTENTION_MODE}')
        
        # Save ablation results
        results_summary = {
            'attention_mode': ATTENTION_MODE,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'avg_uncertainty_correct': np.mean(u_correct),
            'avg_uncertainty_wrong': np.mean(u_wrong)
        }
        
        results_df = pd.DataFrame([results_summary])
        results_file = f'ablation_results_{ATTENTION_MODE}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\nAblation study results saved to {results_file}")
        
    else:
        print("\nSkipping final evaluation because no model was saved.")

if __name__ == '__main__':
    main()