import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import time
from collections import OrderedDict
import copy
import warnings
import random
import torch.nn.functional as F
from tqdm import tqdm # For progress bar

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    # IMPORTANT: Set this to the path where your downloaded parquet files are.
    BASE_PATH = '/kaggle/input/fedalery-karye-ji/patchcamelyon_data'

    # NEW: Path for preprocessed, resized images. This will be created.
    PREPROCESSED_PATH = './preprocessed_patchcamelyon'

    # =================================================================
    # NEW: NON-IID SCENARIO SELECTION
    # =================================================================
    # Choose from 'A', 'B', or 'C'.
    # 'A': Low Non-IID (alpha=10.0) - Balanced data, approaching IID.
    # 'B': Moderate Non-IID (alpha=1.0) - Label skew, uneven proportions.
    # 'C': Pathological Non-IID (alpha=0.5) - Extreme skew, some clients might miss a class.
    NON_IID_SCENARIO = 'A' # Change this value to switch scenarios

    # Federated Learning Settings
    NUM_CLIENTS = 5
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # Learning Rate Scheduler Settings
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_PATIENCE = 2
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_MIN_LR = 1e-6

    # Loss Function Settings
    USE_FOCAL_LOSS = True
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    # REVISED: Reduced multiplier to be less aggressive on the positive class, aiming to reduce FPs.
    CLASS_WEIGHT_MULTIPLIER = 2.0

    # FedAlert Settings (REVISED FOR HIGHER PRECISION)
    USE_FEDALERT = True
    # Increased alpha (from 4.0 to 6.0) and decreased beta (from 3.0 to 2.0)
    # to heavily penalize false positives and aggressively aim for higher precision.
    FEDALERT_ALPHA = 6.0
    FEDALERT_BETA = 2.0
    FEDALERT_LAMBDA = 0.5

    # FedProx Settings
    USE_FEDPROX = True
    FEDPROX_MU = 0.01

    # NEW: Advanced Aggregation Settings
    USE_CONSENSUS_AGGREGATION = True
    CONSENSUS_LAMBDA = 0.5 # Blending factor: 0.0 = only FedAlert, 1.0 = only consensus

    # Transfer Learning Settings
    USE_TRANSFER_LEARNING = True
    PRETRAINED_MODEL = 'mobilenet_v2'
    FREEZE_LAYERS = True
    UNFREEZE_AFTER_ROUNDS = 5

    # Communication Efficiency
    COMPRESSION_RATIO = 0.3
    QUANTIZATION_BITS = 8
    ADAPTIVE_UPDATE = False
    UPDATE_THRESHOLD = 0.005

    # Privacy Settings
    USE_DIFFERENTIAL_PRIVACY = False
    DP_NOISE_MULTIPLIER = 0.01
    DP_CLIP_NORM = 5.0

    # Agentic Alert Settings
    # The threshold for the FedAlert loss and for the final validation of the alert system.
    ALERT_THRESHOLD = 0.75
    ALERT_DELAY_TARGET = 2.0

    # Model Settings
    IMG_SIZE = 96
    NUM_CLASSES = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluation
    VERBOSE = True
    SAVE_RESULTS = True

config = Config()

# =====================================================================
# DATA PREPROCESSING (RUNS ONLY ONCE)
# =====================================================================

def get_parquet_files(directory):
    """Get all parquet files from a directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    return sorted(files)

def preprocess_data(config):
    """
    Extracts, resizes, and saves images from parquet files to a standard
    image folder structure. This is a one-time operation to speed up training.
    """
    print("="*70)
    print("Checking for preprocessed data...")

    if os.path.exists(config.PREPROCESSED_PATH) and len(os.listdir(config.PREPROCESSED_PATH)) > 0:
        print(f"Preprocessed data found at '{config.PREPROCESSED_PATH}'. Skipping preprocessing.")
        return

    print("Preprocessing not found. Starting one-time image extraction and resizing...")
    print(f"Source Parquet Files: '{config.BASE_PATH}'")
    print(f"Destination for PNGs: '{config.PREPROCESSED_PATH}'")

    splits = {
        'train': os.path.join(config.BASE_PATH, 'train'),
        'test': os.path.join(config.BASE_PATH, 'test'),
        'validation': os.path.join(config.BASE_PATH, 'validation')
    }

    for split_name, source_path in splits.items():
        print(f"\nProcessing split: '{split_name}'...")
        parquet_files = get_parquet_files(source_path)
        if not parquet_files:
            print(f"  Warning: No parquet files found in {source_path}. Skipping.")
            continue

        dest_split_path = os.path.join(config.PREPROCESSED_PATH, split_name)
        os.makedirs(os.path.join(dest_split_path, '0'), exist_ok=True)
        os.makedirs(os.path.join(dest_split_path, '1'), exist_ok=True)

        for pf in tqdm(parquet_files, desc=f"  Converting {split_name} files"):
            df = pd.read_parquet(pf)
            for idx, row in df.iterrows():
                try:
                    img_bytes = row['image']['bytes']
                    label = int(row['label'])
                    parquet_filename = os.path.splitext(os.path.basename(pf))[0]
                    image_id = f"{parquet_filename}_row_{idx}"

                    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)

                    save_path = os.path.join(dest_split_path, str(label), f"{image_id}.png")
                    img.save(save_path)
                except Exception as e:
                    print(f"Could not process image {image_id if 'image_id' in locals() else 'N/A'}. Error: {e}")

    print("\nPreprocessing complete. Images are saved and ready for fast loading.")
    print("="*70)

# =====================================================================
# DATA LOADING & FEDERATED SPLITTING
# =====================================================================

def create_non_iid_splits_from_dataset(dataset, num_clients, config):
    """
    Creates non-IID data splits for clients based on a selected scenario.
    Uses Dirichlet distribution to partition indices per class.
    """
    # Select alpha based on the scenario
    scenario_alphas = {
        'A': 10.0, # Low Non-IID (Approximates IID)
        'B': 1.0,  # Moderate Non-IID (Label Skew)
        'C': 0.5   # Pathological Non-IID (Missing Classes)
    }
    # Default to 0.5 if not found
    alpha = scenario_alphas.get(config.NON_IID_SCENARIO.upper(), 0.5) 
    
    print(f"  Creating client splits for Scenario '{config.NON_IID_SCENARIO}' (Dirichlet alpha={alpha})")

    labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    
    # 2. Group all indices by their class label
    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Initialize list of indices for each client
    client_indices = [[] for _ in range(num_clients)]

    # 3. Distribute each class separately
    for c in range(num_classes):
        class_indices = idx_by_class[c]
        np.random.shuffle(class_indices)
        
        # --- CRITICAL FIX ---
        # Generate proportions for how to split THIS class 'c' across 'num_clients'.
        # This vector sums to 1.0.
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Calculate count of samples for each client
        # (proportions * total_class_samples)
        num_samples_per_client = (proportions * len(class_indices)).astype(int)
        
        # Fix rounding errors: Add remainder to the last client (or random client)
        # so we don't lose any data.
        remainder = len(class_indices) - sum(num_samples_per_client)
        num_samples_per_client[-1] += remainder

        # Assign indices to clients
        start_idx = 0
        for client_id, count in enumerate(num_samples_per_client):
            end_idx = start_idx + count
            if count > 0:
                client_indices[client_id].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx

    # 4. Final shuffle so classes are mixed within the client's dataset
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    # 5. Create Subset objects
    return [Subset(dataset, indices) for indices in client_indices]

# =====================================================================
# LOSS FUNCTIONS
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha
        if isinstance(self.alpha, (list, np.ndarray)): alpha_t = self.alpha[targets]
        loss = alpha_t * focal_term * ce_loss
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class FedAlertLoss(nn.Module):
    def __init__(self, base_criterion, alert_threshold=0.75, alpha=6.0, beta=2.0, temperature=10.0, reduction='mean'):
        super(FedAlertLoss, self).__init__()
        self.base_criterion = base_criterion; self.alert_threshold = alert_threshold
        self.alpha = alpha; self.beta = beta; self.temperature = temperature
        self.reduction = reduction
    def soft_threshold(self, probs, threshold):
        # Using a steep sigmoid (high temperature) to approximate a hard threshold
        return torch.sigmoid((probs - threshold) * self.temperature)
    def forward(self, inputs, targets):
        base_loss = self.base_criterion(inputs, targets)
        probs = F.softmax(inputs, dim=1)
        prob_malignant = probs[:, 1] # Probability of class 1 (Tumor)

        # Alert is triggered when P(Tumor) > threshold
        alert_triggered = self.soft_threshold(prob_malignant, self.alert_threshold)

        # Loss Term 1: False Alarm Penalty (True Label is Normal, Alert Triggered)
        false_alarm_mask = (targets == 0).float()
        false_alarm_loss = false_alarm_mask * alert_triggered # Penalizes when Normal is misclassified high

        # Loss Term 2: Missed Detection Penalty (True Label is Tumor, Alert NOT Triggered)
        missed_detection_mask = (targets == 1).float()
        missed_detection_loss = missed_detection_mask * (1.0 - alert_triggered) # Penalizes when Tumor is missed low

        if self.reduction == 'mean':
            false_alarm_loss = false_alarm_loss.mean()
            missed_detection_loss = missed_detection_loss.mean()
        elif self.reduction == 'sum':
            false_alarm_loss = false_alarm_loss.sum()
            missed_detection_loss = missed_detection_loss.sum()

        # Total Alert Loss: weighted sum of False Alarm (FP) and Missed Detection (FN) penalties
        alert_loss = self.alpha * false_alarm_loss + self.beta * missed_detection_loss

        total_loss = base_loss + alert_loss
        return total_loss

# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================
def create_transfer_learning_model(model_name='mobilenet_v2', num_classes=2, pretrained=True):
    model = models.mobilenet_v2(pretrained=pretrained)
    for param in model.features.parameters(): param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, num_classes))
    # print("Using pre-trained MobileNetV2 with a new classifier.")
    return model

def unfreeze_model_layers(model, model_name='mobilenet_v2'):
    if model_name == 'mobilenet_v2':
        # Unfreeze the last 3 feature blocks (feature blocks 14, 15, 16) for fine-tuning
        for param in model.features[-3:].parameters(): param.requires_grad = True
        # print("Fine-tuning: Unfroze last 3 feature blocks of MobileNetV2")

# =====================================================================
# COMMUNICATION & PRIVACY
# =====================================================================
class CommunicationOptimizer:
    @staticmethod
    def top_k_sparsification(params, k_ratio=0.3):
        sparse_params = {}
        for name, param in params.items():
            flat_param = param.flatten()
            k = max(1, int(flat_param.numel() * k_ratio))
            # Top-K based on magnitude
            topk_vals, topk_idx = torch.topk(torch.abs(flat_param.float()), k)
            # Only store the non-zero values and their indices
            sparse_params[name] = {'values': flat_param[topk_idx], 'indices': topk_idx, 'shape': param.shape, 'dtype': param.dtype}
        return sparse_params, 0
    @staticmethod
    def decompress_sparse_params(sparse_params):
        params = {}
        for name, sparam in sparse_params.items():
            # Create a zero tensor of the original size
            flat_param = torch.zeros(int(np.prod(sparam['shape'])), device=sparam['values'].device, dtype=sparam.get('dtype', torch.float32))
            # Place the received values back at their original indices
            flat_param[sparam['indices']] = sparam['values']
            params[name] = flat_param.reshape(sparam['shape'])
        return params

# =====================================================================
# FEDERATED LEARNING CLIENT
# =====================================================================
class FederatedClient:
    def __init__(self, client_id, client_dataset, config):
        self.client_id = client_id; self.config = config
        self.dataset = client_dataset
        self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

        # Calculate class weights for loss function
        labels = [self.dataset.dataset.targets[i] for i in self.dataset.indices]
        class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
        if 0 in class_counts: class_weights = torch.ones(config.NUM_CLASSES, dtype=torch.float)
        else:
            total_samples = len(labels)
            class_weights = torch.FloatTensor([total_samples / (config.NUM_CLASSES * count) for count in class_counts])
        class_weights[1] *= config.CLASS_WEIGHT_MULTIPLIER
        class_weights = class_weights.to(config.DEVICE)
        print(f"  Client {client_id} - Samples: {len(self.dataset)}, Class dist: {class_counts}, Weights: {class_weights.cpu().numpy()}")

        self.model = create_transfer_learning_model(model_name=config.PRETRAINED_MODEL, num_classes=config.NUM_CLASSES).to(config.DEVICE)

        # Initialize Loss Function (FedAlert or standard)
        if config.USE_FEDALERT:
            base_criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, weight=class_weights) if config.USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=class_weights)
            # Note: FedAlertLoss uses the revised alpha/beta from Config
            self.criterion = FedAlertLoss(base_criterion, alert_threshold=config.ALERT_THRESHOLD, alpha=config.FEDALERT_ALPHA, beta=config.FEDALERT_BETA)
        elif config.USE_FOCAL_LOSS: self.criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, weight=class_weights)
        else: self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE) if config.USE_LR_SCHEDULER else None

        self.global_model_params = None; self.prev_loss = float('inf')
        self.comm_optimizer = CommunicationOptimizer(); self.current_alert_metrics = {}

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)
        if self.config.USE_FEDPROX: self.global_model_params = copy.deepcopy(parameters)

    def unfreeze_layers_for_finetuning(self):
        if self.config.USE_TRANSFER_LEARNING: unfreeze_model_layers(self.model, self.config.PRETRAINED_MODEL)

    def get_parameters(self): return copy.deepcopy(self.model.state_dict())

    def train(self, epochs):
        self.model.train(); epoch_losses = []
        all_probs, all_labels = [], []

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in self.dataloader:
                images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss_val = self.criterion(outputs, labels)

                with torch.no_grad():
                    all_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                proximal_term = 0.0
                if self.config.USE_FEDPROX and self.global_model_params:
                    for name, param in self.model.named_parameters():
                        if name in self.global_model_params: proximal_term += torch.norm(param - self.global_model_params[name].to(self.config.DEVICE)) ** 2
                    proximal_term = (self.config.FEDPROX_MU / 2.0) * proximal_term

                loss = loss_val + proximal_term
                loss.backward(); self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.dataloader)
            epoch_losses.append(epoch_loss)

        avg_loss = np.mean(epoch_losses)
        if self.scheduler: self.scheduler.step(avg_loss)

        # Calculate Alert-specific metrics based on the local data at ALERT_THRESHOLD
        if self.config.USE_FEDALERT and len(all_probs) > 0:
            preds = (np.array(all_probs) >= self.config.ALERT_THRESHOLD).astype(int)
            tp = np.sum((preds == 1) & (np.array(all_labels) == 1))
            fp = np.sum((preds == 1) & (np.array(all_labels) == 0))
            fn = np.sum((preds == 0) & (np.array(all_labels) == 1))

            # Using Precision and Recall at the high alert threshold is more telling
            precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            self.current_alert_metrics = {
                'precision_at_threshold': precision_at_threshold,
                'recall_at_threshold': recall_at_threshold,
            }

        return avg_loss, self.current_alert_metrics

    def compute_update(self, global_params):
        local_params = self.get_parameters()
        return OrderedDict((name, local_params[name] - global_params[name]) for name in local_params)

    def should_send_update(self, current_loss):
        if not self.config.ADAPTIVE_UPDATE: return True
        if self.prev_loss == float('inf'): self.prev_loss = current_loss; return True
        improvement = (self.prev_loss - current_loss) / self.prev_loss
        self.prev_loss = current_loss
        return improvement > self.config.UPDATE_THRESHOLD

    def compress_update(self, update):
        return self.comm_optimizer.top_k_sparsification(update, self.config.COMPRESSION_RATIO)

# =====================================================================
# FEDERATED LEARNING SERVER
# =====================================================================
class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.global_model = create_transfer_learning_model(model_name=config.PRETRAINED_MODEL, num_classes=config.NUM_CLASSES).to(config.DEVICE)
        self.comm_optimizer = CommunicationOptimizer()
        self.round_metrics = []; self.total_bytes_transmitted = 0; self.layers_unfrozen = False

    def get_parameters(self): return copy.deepcopy(self.global_model.state_dict())

    def unfreeze_layers_for_finetuning(self, round_num):
        if self.config.USE_TRANSFER_LEARNING and self.config.FREEZE_LAYERS and not self.layers_unfrozen and round_num >= self.config.UNFREEZE_AFTER_ROUNDS:
            print(f"\nRound {round_num}: Starting Fine-Tuning Phase")
            unfreeze_model_layers(self.global_model, self.config.PRETRAINED_MODEL)
            self.layers_unfrozen = True

    # NEW: Helper function for consensus-based aggregation
    def _calculate_consensus_scores(self, client_updates):
        if len(client_updates) <= 1: return [1.0] * len(client_updates)
        flat_updates = []
        for update in client_updates:
            dense_update = self.comm_optimizer.decompress_sparse_params(update)
            # Concatenate all flattened tensors from the update
            tensors = [p.view(-1).float() for p in dense_update.values()]
            flat_updates.append(torch.cat(tensors))

        num_clients = len(flat_updates)
        cosine_matrix = torch.zeros((num_clients, num_clients), device=self.config.DEVICE)
        for i in range(num_clients):
            for j in range(i, num_clients):
                # Calculate cosine similarity between two update vectors
                sim = F.cosine_similarity(flat_updates[i], flat_updates[j], dim=0)
                cosine_matrix[i, j] = sim; cosine_matrix[j, i] = sim

        # Calculate average similarity to all other clients (consensus)
        # Sum of similarities - 1 (self-similarity) / (N-1)
        consensus_scores = (torch.sum(cosine_matrix, dim=1) - 1) / (num_clients - 1)

        # Normalize scores to [0, 1]
        if consensus_scores.max() - consensus_scores.min() > 1e-8:
             consensus_scores = (consensus_scores - consensus_scores.min()) / (consensus_scores.max() - consensus_scores.min())
        else: # Handle case where all similarities are the same
             consensus_scores = torch.ones_like(consensus_scores)

        return consensus_scores.cpu().tolist()

    # REVISED: Implemented more sophisticated, multi-stage aggregation
    def aggregate_updates(self, client_updates, client_weights, clients):
        global_params = self.get_parameters()
        effective_weights = np.array(client_weights, dtype=np.float32)

        # Stage 1: FedAlert-based re-weighting
        if self.config.USE_FEDALERT and clients:
            print("    Applying FedAlert-based re-weighting...")
            alert_scores = []
            for client in clients:
                metrics = client.current_alert_metrics
                precision = metrics.get('precision_at_threshold', 0.0)
                recall = metrics.get('recall_at_threshold', 0.0)
                # Calculate F1-score at the alert threshold locally
                f1_at_threshold = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                alert_quality_score = f1_at_threshold + 0.1 # Add a small epsilon to ensure non-zero weights
                alert_scores.append(alert_quality_score)

            alert_scores = np.array(alert_scores)
            effective_weights *= alert_scores
            print(f"    Alert F1 scores: {[f'{s-0.1:.2f}' for s in alert_scores]}")

        # Stage 2: Consensus-based re-weighting
        if self.config.USE_CONSENSUS_AGGREGATION and len(client_updates) > 1:
            print("    Applying Consensus-based Aggregation...")
            consensus_scores = self._calculate_consensus_scores(client_updates)
            consensus_scores = np.array(consensus_scores)
            lam = self.config.CONSENSUS_LAMBDA
            # Blend the original weights (scaled by alert score) with the consensus scores
            effective_weights = (1 - lam) * effective_weights + lam * (effective_weights * consensus_scores)
            print(f"    Consensus scores: {[f'{s:.2f}' for s in consensus_scores]}")

        # Final Aggregation (Weighted Average)
        aggregated_update = OrderedDict((name, torch.zeros_like(param)) for name, param in global_params.items())
        total_weight = sum(effective_weights)
        if total_weight == 0:
            print("    Warning: Total aggregation weight is zero. Skipping update.")
            return

        for update, weight in zip(client_updates, effective_weights):
            dense_update = self.comm_optimizer.decompress_sparse_params(update)
            for name in aggregated_update:
                if aggregated_update[name].is_floating_point():
                    update_tensor = dense_update[name].to(self.config.DEVICE).to(aggregated_update[name].dtype)
                    aggregated_update[name] += update_tensor * (weight / total_weight)

        # Apply the aggregated update to the global model
        for name in global_params:
            if global_params[name].is_floating_point(): global_params[name] += aggregated_update[name]
        self.global_model.load_state_dict(global_params)

    # REVISED: Added a threshold parameter for final validation
    def evaluate(self, test_loader, threshold=0.5):
        self.global_model.eval()
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.config.DEVICE)
                outputs = self.global_model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Prediction is now based on the specified threshold
        final_preds = (np.array(all_probs)[:, 1] > threshold).astype(int)

        accuracy = accuracy_score(all_labels, final_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, final_preds, average='binary', zero_division=0)
        # AUC is threshold-independent, calculated once from probabilities
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1]) if len(np.unique(all_labels)) > 1 else 0.5

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}, all_probs, all_labels

# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================
def main():
    print("="*70); print("Edge-aware Federated Histopathology (Optimized for Preprocessed Data)"); print("="*70)

    preprocess_data(config)

    print("\nLoading datasets from preprocessed image folders...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    full_train_dataset = ImageFolder(root=os.path.join(config.PREPROCESSED_PATH, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(config.PREPROCESSED_PATH, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nCreating non-IID splits for {config.NUM_CLIENTS} clients...")
    seed = 0
    while True:
        seed += 1; np.random.seed(seed); random.seed(seed)
        # MODIFIED: Pass the config object to the splitting function
        client_datasets = create_non_iid_splits_from_dataset(full_train_dataset, config.NUM_CLIENTS, config)
        if all(len(ds) > 0 for ds in client_datasets):
            print(f"  Successfully created a valid split where all clients have data (using seed {seed})."); break
        else:
            if seed == 1: print(f"  Attempting data split with random seed...")

    print("\nInitializing federated learning components...")
    clients = [FederatedClient(i, ds, config) for i, ds in enumerate(client_datasets)]
    server = FederatedServer(config)

    print("\n" + "="*70); print("Starting Federated Learning"); print("="*70 + "\n")
    all_metrics = []

    for round_num in range(config.NUM_ROUNDS):
        print(f"\n{'='*40} Round {round_num + 1}/{config.NUM_ROUNDS} {'='*40}")
        server.unfreeze_layers_for_finetuning(round_num + 1)

        global_params = server.get_parameters()
        client_updates, client_weights, participating_clients, round_alert_metrics = [], [], [], []

        for client in clients:
            client.set_parameters(global_params)
            if server.layers_unfrozen: client.unfreeze_layers_for_finetuning()

            print(f"\n--- Client {client.client_id} training...")
            loss, alert_metrics = client.train(config.LOCAL_EPOCHS)

            if client.should_send_update(loss):
                update = client.compute_update(global_params)
                compressed_update, _ = client.compress_update(update)
                client_updates.append(compressed_update)
                client_weights.append(len(client.dataset))
                participating_clients.append(client)
                if alert_metrics: round_alert_metrics.append(alert_metrics)
            else: print(f"  Update skipped (insufficient improvement)")

        if client_updates:
            print("\n--- Server aggregating client updates...")
            server.aggregate_updates(client_updates, client_weights, clients=participating_clients)

        # Log FedAlert specific metrics for the round
        if round_alert_metrics:
            avg_alert_prec = np.mean([m['precision_at_threshold'] for m in round_alert_metrics])
            avg_alert_rec = np.mean([m['recall_at_threshold'] for m in round_alert_metrics])
            print(f"\nRound {round_num + 1} FedAlert Local Metrics (Avg) at Thresh={config.ALERT_THRESHOLD:.2f}: "
                  f"Precision: {avg_alert_prec:.4f}, Recall: {avg_alert_rec:.4f}")

        print("\n--- Evaluating global model...")
        # Use 0.5 threshold for general round-by-round monitoring and plots
        metrics, _, _ = server.evaluate(test_loader, threshold=0.5)
        all_metrics.append(metrics)

        print(f"\nRound {round_num + 1} Results (Thresh=0.50): Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

    if not all_metrics: print("\nNo training rounds completed. Exiting."); return


    # =================================================================
    # FINAL EVALUATION & VISUALIZATION
    # =================================================================
    print("\n" + "="*70); print("FINAL RESULTS SUMMARY"); print("="*70 + "\n")

    # 1. Evaluate and capture raw probabilities
    metrics_05, final_probs, final_labels = server.evaluate(test_loader, threshold=0.5)

    # 2. Re-evaluate at the FedAlert threshold for validation
    metrics_alert, _, _ = server.evaluate(test_loader, threshold=config.ALERT_THRESHOLD)

    print(f"Classification Performance at Standard Threshold (0.50):")
    print(f"  Final Accuracy:  {metrics_05['accuracy']:.4f}"); print(f"  Final Precision: {metrics_05['precision']:.4f}")
    print(f"  Final Recall:    {metrics_05['recall']:.4f}"); print(f"  Final F1-Score:  {metrics_05['f1']:.4f}")
    print(f"  Final AUC:       {metrics_05['auc']:.4f}\n")

    print(f"Classification Performance at FedAlert Threshold ({config.ALERT_THRESHOLD:.2f}):")
    print(f"  Final Accuracy:  {metrics_alert['accuracy']:.4f}"); print(f"  Final Precision: {metrics_alert['precision']:.4f}")
    print(f"  Final Recall:    {metrics_alert['recall']:.4f}"); print(f"  Final F1-Score:  {metrics_alert['f1']:.4f}")
    print(f"  Final AUC:       {metrics_alert['auc']:.4f}") # AUC is always the same

    print("\nGenerating visualizations...")

    # Plotting Federated Training Performance (uses 0.5 threshold metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rounds = range(1, len(all_metrics) + 1)

    axes[0, 0].plot(rounds, [m['accuracy'] for m in all_metrics], 'b-o', label='Accuracy')
    axes[0, 1].plot(rounds, [m['auc'] for m in all_metrics], 'r-o', label='AUC')
    axes[1, 0].plot(rounds, [m['f1'] for m in all_metrics], 'g-o', label='F1-Score')
    axes[1, 1].plot(rounds, [m['recall'] for m in all_metrics], 'm-o', label='Recall')

    for ax in axes.flat: ax.set_xlabel('Round'); ax.grid(True); ax.legend()
    plt.tight_layout(pad=3.0); plt.suptitle('Federated Training Performance (Thresh=0.50)', fontsize=16); plt.savefig('federated_training_metrics.png'); plt.show()

    # Plotting Confusion Matrix (uses FedAlert threshold)
    final_preds_alert = (np.array(final_probs)[:, 1] > config.ALERT_THRESHOLD).astype(int)
    cm = confusion_matrix(final_labels, final_preds_alert)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])
    plt.title(f'Confusion Matrix - Final Model (Threshold={config.ALERT_THRESHOLD:.2f})'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.savefig('confusion_matrix.png'); plt.show()

if __name__ == "__main__":
    # Hardcoded configuration settings
    SCENARIO = 'A'
    MODEL_MODE = 'full'  # Options: 'fedavg', 'fedprox', 'fedalert_only', 'consensus_only', 'full'

    # OVERRIDE CONFIG BASED ON HARDCODED SETTINGS
    config.NON_IID_SCENARIO = SCENARIO
    
    print(f"--- RUNNING CONFIGURATION: Model={MODEL_MODE}, Scenario={SCENARIO} ---")

    if MODEL_MODE == 'fedavg':
        config.USE_FEDALERT = False
        config.USE_CONSENSUS_AGGREGATION = False
        config.USE_FEDPROX = False
        config.USE_FOCAL_LOSS = False
    
    elif MODEL_MODE == 'fedprox':
        config.USE_FEDALERT = False
        config.USE_CONSENSUS_AGGREGATION = False
        config.USE_FEDPROX = True
        config.USE_FOCAL_LOSS = False
        
    elif MODEL_MODE == 'fedalert_only':
        config.USE_FEDALERT = True
        config.USE_CONSENSUS_AGGREGATION = False
        config.USE_FEDPROX = False 
        config.USE_FOCAL_LOSS = True
        
    elif MODEL_MODE == 'consensus_only':
        config.USE_FEDALERT = False
        config.USE_CONSENSUS_AGGREGATION = True
        config.USE_FEDPROX = False
        config.USE_FOCAL_LOSS = True
        
    elif MODEL_MODE == 'full':
        config.USE_FEDALERT = True
        config.USE_CONSENSUS_AGGREGATION = True
        config.USE_FEDPROX = True
        config.USE_FOCAL_LOSS = True

    # Run the main training loop
    main()