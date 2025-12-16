# -*- coding: utf-8 -*-
"""
DST-AgenticNet v2: Dynamic Uncertainty-Gated Fusion
Track 1: Agentic and Generative AI Paradigm Shift implementation.
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
import glob

warnings.filterwarnings("ignore")

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
# Options: 'RGB_ONLY', 'NAIVE_FUSION', 'GATED_FUSION' (The proposed method)
EXPERIMENT_MODE = 'GATED_FUSION' 

DATA_DIR = '/kaggle/input/breast-histopathology-images'
IMAGE_SIZE = 96
BATCH_SIZE = 128
EPOCHS = 20 # Reduced slightly for demo
WARMUP_EPOCHS = 3 # Train agents independently first
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

print(f"Running Experiment: {EXPERIMENT_MODE}")

# --- 1. Metrics: ECE (Expected Calibration Error) ---
def compute_ece(probs, labels, n_bins=15):
    """
    Lower ECE = Better Calibration (Model knows when it doesn't know)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences, predictions = np.max(probs, axis=1), np.argmax(probs, axis=1)
    accuracies = predictions == labels

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# --- 2. Loss Functions ---
def kl_divergence(alpha, num_classes, device=None):
    if device is None: device = alpha.device
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

class EDLLoss(nn.Module):
    def __init__(self, num_classes=2, annealing_step=10, device='cuda'):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
        self.epoch = 0

    def forward(self, output, target):
        evidence = output
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        belief = alpha / S
        y = F.one_hot(target, self.num_classes).float()
        
        # MSE Loss on Beliefs
        loss_mse = torch.sum((y - belief) ** 2 + ((alpha * (S - alpha)) / (S * S * (S + 1))), dim=1)
        
        # KL Divergence Regularization
        annealing_coef = min(1, self.epoch / self.annealing_step)
        alpha_tilde = y + (1 - y) * alpha
        kl = kl_divergence(alpha_tilde, self.num_classes, device=self.device)
        return torch.mean(loss_mse + annealing_coef * kl.squeeze())

# --- 3. The Proposed Architecture ---

class GatedFusion(nn.Module):
    """
    NOVELTY: Attentional Gating to weigh agents based on their consensus/feature strength.
    """
    def __init__(self, num_agents, num_classes, embedding_dim=64):
        super(GatedFusion, self).__init__()
        # Input: Concat of 3 evidence vectors (3 * num_classes)
        self.gate_net = nn.Sequential(
            nn.Linear(num_agents * num_classes, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_agents), # Output 1 weight per agent
            nn.Softmax(dim=1) # Ensure weights sum to 1
        )

    def forward(self, evidences):
        # evidences: [Batch, Agents, Classes]
        batch_size, n_agents, n_classes = evidences.shape
        
        # Flatten evidences to feed into Gate Net
        flat_evidences = evidences.view(batch_size, -1)
        
        # Calculate dynamic weights: [Batch, Agents]
        weights = self.gate_net(flat_evidences)
        
        # Expand weights for broadcasting: [Batch, Agents, 1]
        weights_expanded = weights.unsqueeze(2)
        
        # Weighted Sum (Dempster-Shafer variant)
        # We weigh the EVIDENCE, not the probability
        weighted_evidence = evidences * weights_expanded
        fused_evidence = torch.sum(weighted_evidence, dim=1)
        
        return fused_evidence, weights

class DST_AgenticNet(nn.Module):
    def __init__(self, num_classes=2, mode='GATED_FUSION'):
        super(DST_AgenticNet, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        
        # --- Agent 1: RGB (ResNet18) ---
        base_model = models.resnet18(pretrained=True)
        self.rgb_extractor = nn.Sequential(*list(base_model.children())[:-1]) 
        self.rgb_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes), nn.Softplus()
        )
        
        # --- Agent 2: Edge (Structural) ---
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.edge_head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Softplus()
        )
        
        # --- Agent 3: Freq (Spectral) ---
        self.freq_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Softplus()
        )
        
        # Fusion Strategy
        self.gate = GatedFusion(num_agents=3, num_classes=num_classes)
        
        # Fixed Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_edges(self, x):
        gray = torch.mean(x, dim=1, keepdim=True)
        ex = F.conv2d(gray, self.sobel_x, padding=1)
        ey = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(ex**2 + ey**2 + 1e-6)

    def get_freq(self, x):
        gray = torch.mean(x, dim=1, keepdim=True)
        fft = torch.fft.fft2(gray, norm='ortho')
        fft_amp = torch.abs(fft)
        return F.adaptive_avg_pool2d(fft_amp, (1, 128)).flatten(1) 

    def forward(self, x):
        # 1. Get Individual Evidences
        # RGB
        rgb_feat = self.rgb_extractor(x).flatten(1)
        e_rgb = self.rgb_head(rgb_feat)
        
        # Edge
        edges = self.get_edges(x)
        edge_feat = self.edge_extractor(edges)
        e_edge = self.edge_head(edge_feat)
        
        # Freq
        freq_vec = self.get_freq(x)
        e_freq = self.freq_head(freq_vec)
        
        all_evidences = torch.stack([e_rgb, e_edge, e_freq], dim=1)
        
        # 2. Fusion Logic
        if self.mode == 'RGB_ONLY':
            return e_rgb, all_evidences, None
        
        elif self.mode == 'NAIVE_FUSION':
            # Simple Sum (What you did before)
            return torch.sum(all_evidences, dim=1), all_evidences, None
            
        elif self.mode == 'GATED_FUSION':
            # Proposed Method
            fused, weights = self.gate(all_evidences)
            return fused, all_evidences, weights
            
        return e_rgb, all_evidences, None

# --- 4. Training ---

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    # WARMUP: Train agents independently first
    if epoch < WARMUP_EPOCHS:
        curr_mode = 'WARMUP'
    else:
        curr_mode = 'FUSION'
        
    for inputs, labels in tqdm(dataloader, desc=f"Training ({curr_mode})", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 1. Forward Pass
        fused_evidence, individual_evidences, weights = model(inputs)
        
        # 2. AGENT DROPOUT (The Fix!)
        # With 30% probability, we zero out the RGB evidence entering the fusion layer
        # This forces the network to rely on Edge/Freq agents
        if curr_mode == 'FUSION' and np.random.rand() < 0.3:
            # Create a mask that zeros out the RGB agent (index 0)
            # evidence shape: [Batch, 3, Classes]
            mask = torch.ones_like(individual_evidences)
            mask[:, 0, :] = 0.0 # Kill RGB
            
            # Recalculate fusion with RGB blinded
            blind_evidences = individual_evidences * mask
            fused_evidence_blind, _ = model.gate(blind_evidences)
            
            loss = criterion(fused_evidence_blind, labels)
            
        else:
            # Normal training
            if curr_mode == 'WARMUP':
                loss_rgb = criterion(individual_evidences[:,0,:], labels)
                loss_edge = criterion(individual_evidences[:,1,:], labels)
                loss_freq = criterion(individual_evidences[:,2,:], labels)
                loss = loss_rgb + loss_edge + loss_freq
            else:
                # Gated Fusion Loss
                loss_main = criterion(fused_evidence, labels)
                # Keep auxiliary heads alive
                loss_aux = 0.2 * (criterion(individual_evidences[:,1,:], labels) + 
                                  criterion(individual_evidences[:,2,:], labels))
                loss = loss_main + loss_aux

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] # For ECE
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            fused, _, _ = model(inputs)
            
            # Belief -> Probability
            alpha = fused + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S
            
            _, preds = torch.max(fused, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    ece = compute_ece(np.array(all_probs), np.array(all_labels))
    
    return acc, f1, ece

def visualize_gating_behavior(model, dataloader, device):
        """
        Generates a plot showing how the model weights different agents
        for different types of images.
        """
        model.eval()
        all_weights = []
        all_preds = []
        all_labels = []
        
        # Collect weights from the test set
        print("Extracting Gating Weights...")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                # Forward pass to get weights (fused, evidences, weights)
                _, _, weights = model(inputs) 
                
                # weights shape: [Batch, 3, 1] -> squeeze to [Batch, 3]
                all_weights.extend(weights.squeeze().cpu().numpy())
                all_labels.extend(labels.numpy())
        
        weights_np = np.array(all_weights) # Shape: [N_samples, 3]
        labels_np = np.array(all_labels)
        
        # 3 Agents: RGB (idx 0), Edge (idx 1), Freq (idx 2)
        agent_names = ['RGB Agent', 'Edge Agent', 'Freq Agent']
        
        # --- PLOT 1: Average Contribution of Each Agent ---
        plt.figure(figsize=(10, 6))
        mean_weights = np.mean(weights_np, axis=0)
        sns.barplot(x=agent_names, y=mean_weights, palette="viridis")
        plt.title("Global Agent Importance (What does the model trust?)")
        plt.ylabel("Average Attention Weight")
        plt.ylim(0, 1.0)
        for i, v in enumerate(mean_weights):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
        plt.savefig("global_agent_importance.png", dpi=300)
        plt.show()
        
        # --- PLOT 2: Weight Distribution (Did it actually learn dynamic gating?) ---
        # If this plot shows wide distributions, it means the model is "thinking" 
        # differently for each image. If narrow, it just learned a static bias.
        plt.figure(figsize=(12, 6))
        df_weights = pd.DataFrame(weights_np, columns=agent_names)
        df_weights_melted = df_weights.melt(var_name="Agent", value_name="Weight")
        
        sns.kdeplot(data=df_weights_melted, x="Weight", hue="Agent", fill=True, clip=(0,1))
        plt.title("Dynamic Gating Distribution (Evidence of Adaptive Fusion)")
        plt.xlabel("Attention Weight Assigned")
        plt.savefig("dynamic_gating_dist.png", dpi=300)
        plt.show()


# --- 5. Main ---
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Data Prep ---
    print("Preparing Data...")
    all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)
    if not all_image_paths: return
    
    # Quick Label Logic
    labels = [int(os.path.basename(p).split('_class')[1].split('.')[0]) for p in all_image_paths]
    df = pd.DataFrame({'path': all_image_paths, 'label': labels})
    
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=RANDOM_SEED)
    
    trans = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(HistopathologyDataset(train_df, trans), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(HistopathologyDataset(val_df, trans), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(HistopathologyDataset(test_df, trans), batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model ---
    model = DST_AgenticNet(num_classes=2, mode=EXPERIMENT_MODE).to(DEVICE)
    criterion = EDLLoss(num_classes=2, annealing_step=EPOCHS, device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    

    # --- Train ---
    best_f1 = 0
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        criterion.epoch = epoch
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        acc, f1, ece = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | ECE: {ece:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"best_{EXPERIMENT_MODE}.pth")
            
    # --- Final Test ---
    model.load_state_dict(torch.load(f"best_{EXPERIMENT_MODE}.pth"))
    acc, f1, ece = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nFINAL TEST RESULTS ({EXPERIMENT_MODE}):")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ECE (Calibration Error): {ece:.4f} (Lower is better)")
    # --- RUN THIS using your loaded model and test_loader ---
    model.load_state_dict(torch.load("best_GATED_FUSION.pth"))
    visualize_gating_behavior(model, test_loader, DEVICE)      
    # Save for paper
    with open("final_results.txt", "a") as f:
        f.write(f"{EXPERIMENT_MODE},{acc},{f1},{ece}\n")

# Copy Dataset Class from previous code here
class HistopathologyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        try:
            image = Image.open(img_path).convert("RGB")
            label = int(self.dataframe.iloc[idx]['label'])
            if self.transform: image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except: return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), torch.tensor(0)

if __name__ == "__main__":
    main()