"""
Edge-aware Federated Histopathology: Communication-Efficient, Privacy-Preserving 
Breast Cancer Patch Detection with Agentic Alerts

Complete implementation for PatchCamelyon dataset on Kaggle, modified to
download the dataset directly from Hugging Face.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import time
from collections import OrderedDict
import copy
import warnings
from datasets import load_dataset # Added for Hugging Face integration
import random # Add this import at the top of your file if not already there
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    # Paths (data will be downloaded here)
    BASE_PATH = './patchcamelyon_data'
    TRAIN_PATH = os.path.join(BASE_PATH, 'train')
    TEST_PATH = os.path.join(BASE_PATH, 'test')
    VAL_PATH = os.path.join(BASE_PATH, 'validation')
    
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
    FOCAL_LOSS_ALPHA = 0.25  # Weight for positive class
    FOCAL_LOSS_GAMMA = 2.0   # Focusing parameter
    CLASS_WEIGHT_MULTIPLIER = 2.5  # Increased from 1.5 to 2.5 for aggressive weighting
    
    # FedAlert Settings (Novel Alert-Optimized Federated Learning)
    USE_FEDALERT = True  # Enable FedAlert algorithm
    FEDALERT_ALPHA = 2.0  # Weight for false alarm penalty
    FEDALERT_BETA = 3.0   # Weight for missed detection penalty
    FEDALERT_LAMBDA = 0.5  # Mixing coefficient (0=pure FedAlert, 1=pure base loss)
    
    # FedProx Settings
    USE_FEDPROX = True
    FEDPROX_MU = 0.01  # Proximal term coefficient
    
    # Transfer Learning Settings
    USE_TRANSFER_LEARNING = True
    PRETRAINED_MODEL = 'mobilenet_v2'  # Options: 'mobilenet_v2', 'efficientnet_b0'
    FREEZE_LAYERS = True
    UNFREEZE_AFTER_ROUNDS = 5  # Start fine-tuning after N rounds
    
    # Communication Efficiency
    COMPRESSION_RATIO = 0.3  # Top-k sparsification (keep 30% of gradients)
    QUANTIZATION_BITS = 8
    ADAPTIVE_UPDATE = False  # Disabled - all clients send updates every round
    UPDATE_THRESHOLD = 0.005  # 0.5% improvement threshold
    
    # Privacy Settings
    USE_DIFFERENTIAL_PRIVACY = False  # Temporarily disabled to diagnose issue
    DP_NOISE_MULTIPLIER = 0.01  # Reduced noise to prevent gradient corruption
    DP_CLIP_NORM = 5.0  # Increased clip norm to preserve gradient information
    
    # Agentic Alert Settings
    ALERT_THRESHOLD = 0.75  # Probability threshold for malignant detection
    ALERT_DELAY_TARGET = 2.0  # Target delay in seconds
    
    # Model Settings
    IMG_SIZE = 96
    NUM_CLASSES = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluation
    VERBOSE = True
    SAVE_RESULTS = True

config = Config()

# =====================================================================
# DATASET DOWNLOAD AND PREPARATION (NEW)
# =====================================================================

def download_and_prepare_dataset(config):
    """Downloads the dataset from Hugging Face and saves it in the expected format."""
    print("Downloading and preparing dataset from Hugging Face...")
    
    # Create directories if they don't exist
    os.makedirs(config.TRAIN_PATH, exist_ok=True)
    os.makedirs(config.TEST_PATH, exist_ok=True)
    os.makedirs(config.VAL_PATH, exist_ok=True)

    # Load the dataset from Hugging Face
    try:
        dataset = load_dataset("1aurent/PatchCamelyon", cache_dir="./hf_cache")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please ensure you have an internet connection and the `datasets` library is installed (`pip install datasets`).")
        exit()

    # Define sharding to mimic the original multi-file structure
    shards_config = {
        'train': (config.TRAIN_PATH, 13),
        'test': (config.TEST_PATH, 2),
        'valid': (config.VAL_PATH, 2)  # <-- CORRECTED THIS LINE

    }

    for split, (path, num_shards) in shards_config.items():
        print(f"Processing and sharding '{split}' split...")
        
        # Check if files already exist to avoid reprocessing
        if len(os.listdir(path)) >= num_shards:
            print(f"  Files for '{split}' split already exist. Skipping.")
            continue

        split_dataset = dataset[split]
        df = split_dataset.to_pandas()
        
        # Ensure the image column format is correct for the existing Dataset class
        df['image'] = df['image'].apply(lambda x: {'bytes': x['bytes']})

        # Shard the dataframe and save to multiple parquet files
        shard_size = (len(df) + num_shards - 1) // num_shards # Ceiling division
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(df))
            shard_df = df.iloc[start_idx:end_idx]
            
            if not shard_df.empty:
                file_path = os.path.join(path, f'{split}-{i:05d}-of-{num_shards:05d}.parquet')
                shard_df.to_parquet(file_path)
                print(f"  Saved: {file_path}")
            
    print("Dataset preparation complete.")


# =====================================================================
# DATASET AND DATA LOADING
# =====================================================================

class PatchCamelyonDataset(Dataset):
    """Dataset for PatchCamelyon parquet files"""
    
    def __init__(self, parquet_files, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        # Load all parquet files
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for idx, row in df.iterrows():
                self.data.append(row['image'])
                self.labels.append(int(row['label']))
        
        print(f"Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image from bytes
        img_bytes = self.data[idx]['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def get_parquet_files(directory):
    """Get all parquet files from directory"""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    return sorted(files)

def create_non_iid_splits(parquet_files, num_clients):
    """
    Create non-IID data splits for federated clients
    Simulates heterogeneous data distribution across hospitals
    """
    np.random.shuffle(parquet_files)
    
    # Dirichlet distribution for non-IID partitioning
    alpha = 0.5  # Lower alpha = more non-IID
    num_files = len(parquet_files)
    
    # Generate partition sizes using Dirichlet
    proportions = np.random.dirichlet([alpha] * num_clients)
    proportions = (proportions * num_files).astype(int)
    
    # Adjust to ensure sum equals num_files
    proportions[-1] = num_files - proportions[:-1].sum()
    
    client_files = []
    start_idx = 0
    for prop in proportions:
        end_idx = start_idx + prop
        client_files.append(parquet_files[start_idx:end_idx])
        start_idx = end_idx
    
    return client_files

# =====================================================================
# LOSS FUNCTIONS
# =====================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    This loss focuses training on hard examples by down-weighting easy examples.
    Perfect for medical imaging where there are many "easy" negative examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N,) where each value is 0 <= targets[i] <= C-1
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha
        if isinstance(self.alpha, (list, np.ndarray)):
            alpha_t = self.alpha[targets]
        
        loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FedAlertLoss(nn.Module):
    """
    FedAlert Loss: Novel loss function for alert-optimized federated learning
    
    This loss directly optimizes for high-precision, low-false-alarm detection by:
    1. Penalizing false alarms (FP) - samples predicted as malignant when benign
    2. Penalizing missed detections (FN) - samples predicted as benign when malignant
    3. Using a differentiable approximation of alert precision at a specific threshold
    
    Mathematical Formulation:
    L_FedAlert = L_base + alpha * L_false_alarm + beta * L_missed_detection
    
    Where:
    - L_base: Standard cross-entropy or focal loss
    - L_false_alarm: Penalty for false positives at alert threshold
    - L_missed_detection: Penalty for false negatives at alert threshold
    - alpha, beta: Hyperparameters controlling trade-off
    
    Key Innovation: Unlike standard FL that optimizes global accuracy, FedAlert
    optimizes the specific operating point (threshold) used for clinical alerts.
    """
    
    def __init__(self, base_criterion, alert_threshold=0.75, alpha=2.0, beta=3.0, 
                 temperature=10.0, reduction='mean'):
        """
        Args:
            base_criterion: Base loss function (e.g., FocalLoss or CrossEntropyLoss)
            alert_threshold: Probability threshold for triggering alerts (default: 0.75)
            alpha: Weight for false alarm penalty (default: 2.0)
            beta: Weight for missed detection penalty (default: 3.0) - usually higher!
            temperature: Temperature for soft threshold approximation (default: 10.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FedAlertLoss, self).__init__()
        self.base_criterion = base_criterion
        self.alert_threshold = alert_threshold
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.reduction = reduction
        
    def soft_threshold(self, probs, threshold):
        """
        Differentiable soft approximation of threshold function using sigmoid
        
        Args:
            probs: Predicted probabilities [N]
            threshold: Alert threshold
            
        Returns:
            Soft indicator: ~0 if prob << threshold, ~1 if prob >> threshold
        """
        # sigmoid((prob - threshold) * temperature)
        # When temperature is high, this approximates a hard threshold
        return torch.sigmoid((probs - threshold) * self.temperature)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits [N, C] where C=2 (benign, malignant)
            targets: True labels [N] where 0=benign, 1=malignant
            
        Returns:
            Combined loss optimized for alert performance
        """
        # Base loss (cross-entropy or focal loss)
        base_loss = self.base_criterion(inputs, targets)
        
        # Get probabilities for malignant class (class 1)
        probs = F.softmax(inputs, dim=1)
        prob_malignant = probs[:, 1]  # P(malignant)
        
        # Soft threshold: probability that alert would be triggered
        alert_triggered = self.soft_threshold(prob_malignant, self.alert_threshold)
        
        # False Alarm Loss (False Positive)
        # Penalize when benign samples (target=0) trigger alerts (high prob)
        # Loss is high when: target=0 AND prob_malignant > threshold
        false_alarm_mask = (targets == 0).float()
        false_alarm_loss = false_alarm_mask * alert_triggered
        
        # Missed Detection Loss (False Negative)
        # Penalize when malignant samples (target=1) don't trigger alerts (low prob)
        # Loss is high when: target=1 AND prob_malignant < threshold
        missed_detection_mask = (targets == 1).float()
        missed_detection_loss = missed_detection_mask * (1.0 - alert_triggered)
        
        # Aggregate losses
        if self.reduction == 'mean':
            false_alarm_loss = false_alarm_loss.mean()
            missed_detection_loss = missed_detection_loss.mean()
        elif self.reduction == 'sum':
            false_alarm_loss = false_alarm_loss.sum()
            missed_detection_loss = missed_detection_loss.sum()
        
        # Total FedAlert loss
        # Beta (missed detection weight) is typically higher than alpha
        # because missing cancer is more critical than false alarms in medical context
        alert_loss = self.alpha * false_alarm_loss + self.beta * missed_detection_loss
        
        # Combined loss
        total_loss = base_loss + alert_loss
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'false_alarm_loss': false_alarm_loss.item(),
            'missed_detection_loss': missed_detection_loss.item(),
            'alert_loss': alert_loss.item(),
            'total_loss': total_loss.item()
        }

# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================

class LightweightCNN(nn.Module):
    """Lightweight CNN for edge deployment - MobileNet-inspired"""
    
    def __init__(self, num_classes=2):
        super(LightweightCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Depthwise separable conv blocks
            self._make_depthwise_block(32, 64, 1),
            self._make_depthwise_block(64, 128, 2),
            self._make_depthwise_block(128, 128, 1),
            self._make_depthwise_block(128, 256, 2),
            self._make_depthwise_block(256, 256, 1),
            self._make_depthwise_block(256, 512, 2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_depthwise_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize weights properly to prevent bias"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_transfer_learning_model(model_name='mobilenet_v2', num_classes=2, pretrained=True):
    """
    Create a transfer learning model using pre-trained weights
    
    Args:
        model_name: 'mobilenet_v2' or 'efficientnet_b0'
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
    
    Returns:
        model: Modified model with custom classifier
    """
    if model_name == 'mobilenet_v2':
        # Load pre-trained MobileNetV2
        model = models.mobilenet_v2(pretrained=pretrained)
        
        # Freeze feature extraction layers
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        print(f"Transfer Learning: Using pre-trained MobileNetV2")
        print(f"  Frozen layers: features")
        print(f"  Trainable layers: classifier")
        
    elif model_name == 'efficientnet_b0':
        try:
            # Try to load EfficientNet (requires timm library)
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
            
            # Freeze early layers
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'fc' not in name:
                    param.requires_grad = False
            
            print(f"Transfer Learning: Using pre-trained EfficientNet-B0")
        except ImportError:
            print("Warning: timm library not found. Falling back to MobileNetV2")
            return create_transfer_learning_model('mobilenet_v2', num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def unfreeze_model_layers(model, model_name='mobilenet_v2'):
    """
    Gradually unfreeze layers for fine-tuning
    
    Args:
        model: The model to unfreeze
        model_name: Type of model architecture
    """
    if model_name == 'mobilenet_v2':
        # Unfreeze the last few blocks of features
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        print("Fine-tuning: Unfroze last 3 feature blocks of MobileNetV2")
    elif model_name == 'efficientnet_b0':
        # Unfreeze last few blocks
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if 'blocks.6' in name or 'blocks.5' in name or 'bn2' in name:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"Fine-tuning: Unfroze {unfrozen_count} parameters in EfficientNet-B0")

# =====================================================================
# COMMUNICATION OPTIMIZATION
# =====================================================================

# =====================================================================
# COMMUNICATION OPTIMIZATION
# =====================================================================

class CommunicationOptimizer:
    """Handles compression, quantization, and adaptive updates"""
    
    @staticmethod
    def top_k_sparsification(params, k_ratio=0.1):
        """Keep only top-k% of parameters by magnitude"""
        sparse_params = {}
        total_bytes_original = 0
        total_bytes_compressed = 0
        
        for name, param in params.items():
            flat_param = param.flatten()
            total_bytes_original += flat_param.numel() * 4  # float32
            
            k = max(1, int(flat_param.numel() * k_ratio))
            # NOTE: torch.topk works on integer types as well
            topk_vals, topk_idx = torch.topk(torch.abs(flat_param.float()), k) # Use .float() for abs value comparison safely
            
            # Store only non-zero values and indices
            sparse_params[name] = {
                'values': flat_param[topk_idx],
                'indices': topk_idx,
                'shape': param.shape,
                'dtype': param.dtype  # <--- FIX: Store the original dtype
            }
            total_bytes_compressed += k * 4 + k * 4  # values + indices
        
        compression_ratio = total_bytes_compressed / total_bytes_original if total_bytes_original > 0 else 0
        return sparse_params, compression_ratio
    
    @staticmethod
    def quantize_params(params, num_bits=8):
        """Quantize parameters to reduce precision"""
        quantized_params = {}
        
        for name, param in params.items():
            min_val = param.min()
            max_val = param.max()
            
            # Quantize to num_bits
            scale = (max_val - min_val) / (2**num_bits - 1)
            quantized = ((param - min_val) / scale).round().to(torch.uint8)
            
            quantized_params[name] = {
                'quantized': quantized,
                'min': min_val,
                'scale': scale,
                'shape': param.shape
            }
        
        return quantized_params
    
    @staticmethod
    def dequantize_params(quantized_params):
        """Dequantize parameters"""
        params = {}
        
        for name, qparam in quantized_params.items():
            dequantized = qparam['quantized'].float() * qparam['scale'] + qparam['min']
            params[name] = dequantized.reshape(qparam['shape'])
        
        return params
    
    @staticmethod
    def decompress_sparse_params(sparse_params):
        """Decompress sparse parameters"""
        params = {}
        
        for name, sparam in sparse_params.items():
            target_device = sparam['values'].device

            # <--- FIX: Use the stored dtype when creating the zero tensor
            # Default to float32 for safety/backward compatibility
            original_dtype = sparam.get('dtype', torch.float32)
            
            flat_param = torch.zeros(
                int(np.prod(sparam['shape'])), 
                device=target_device, 
                dtype=original_dtype  # Use the correct dtype here
            )
            
            # Now the dtypes of flat_param (destination) and sparam['values'] (source) will match
            flat_param[sparam['indices']] = sparam['values']
            params[name] = flat_param.reshape(sparam['shape'])
        
        return params
# =====================================================================
# PRIVACY MECHANISMS
# =====================================================================

class PrivacyPreserver:
    """Differential privacy and secure aggregation"""
    
    
    @staticmethod
    def add_gaussian_noise(params, noise_multiplier, clip_norm):
        """Add Gaussian noise for differential privacy"""
        noisy_params = {}
        
        for name, param in params.items():
            # FIX: Skip non-floating point tensors like 'num_batches_tracked' from BatchNorm
            if not param.is_floating_point():
                noisy_params[name] = param  # Keep the buffer as is
                continue
                
            # Clip gradients
            param_norm = torch.norm(param)
            if param_norm > clip_norm:
                param = param * (clip_norm / param_norm)
            
            # Add Gaussian noise
            noise = torch.randn_like(param) * noise_multiplier * clip_norm
            noisy_params[name] = param + noise
        
        return noisy_params

# =====================================================================
# FEDERATED LEARNING CLIENT
# =====================================================================

class FederatedClient:
    """Individual client in federated learning"""
    
    def __init__(self, client_id, parquet_files, config):
        self.client_id = client_id
        self.config = config
        
        # Data transforms with AGGRESSIVE augmentation
        augmentation_list = [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),  # Increased from 10 to 20 degrees
            transforms.ColorJitter(
                brightness=0.2,    # Vary brightness
                contrast=0.2,      # Vary contrast
                saturation=0.2,    # Vary saturation
                hue=0.1           # Vary hue
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Random translation
                scale=(0.9, 1.1),      # Random scaling
                shear=10               # Random shearing
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.transform = transforms.Compose(augmentation_list)
        
        # Load dataset
        self.dataset = PatchCamelyonDataset(parquet_files, transform=self.transform)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=2
        )
        
        # Calculate class weights for handling imbalance
        labels = [self.dataset.labels[i] for i in range(len(self.dataset))]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        # Compute balanced class weights
        class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
        
        # AGGRESSIVE: Increased multiplier from 1.5 to 2.5 for even more focus on tumor class
        # This heavily penalizes false negatives (missed tumors)
        class_weights[1] = class_weights[1] * config.CLASS_WEIGHT_MULTIPLIER
        
        class_weights = class_weights.to(config.DEVICE)
        
        print(f"  Client {client_id} - Class distribution: {class_counts}, Weights: {class_weights.cpu().numpy()}")
        
        # Model selection: Transfer Learning vs Custom CNN
        if config.USE_TRANSFER_LEARNING:
            self.model = create_transfer_learning_model(
                model_name=config.PRETRAINED_MODEL,
                num_classes=config.NUM_CLASSES,
                pretrained=True
            ).to(config.DEVICE)
        else:
            self.model = LightweightCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        
        # Loss function selection: FedAlert vs Focal Loss vs Weighted Cross Entropy
        if config.USE_FEDALERT:
            # First create base criterion
            if config.USE_FOCAL_LOSS:
                base_criterion = FocalLoss(
                    alpha=config.FOCAL_LOSS_ALPHA,
                    gamma=config.FOCAL_LOSS_GAMMA,
                    weight=class_weights
                )
            else:
                base_criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Wrap with FedAlertLoss
            self.criterion = FedAlertLoss(
                base_criterion=base_criterion,
                alert_threshold=config.ALERT_THRESHOLD,
                alpha=config.FEDALERT_ALPHA,
                beta=config.FEDALERT_BETA
            )
            self.use_fedalert = True
            print(f"  Using FedAlert Loss (threshold={config.ALERT_THRESHOLD}, α={config.FEDALERT_ALPHA}, β={config.FEDALERT_BETA})")
        elif config.USE_FOCAL_LOSS:
            self.criterion = FocalLoss(
                alpha=config.FOCAL_LOSS_ALPHA,
                gamma=config.FOCAL_LOSS_GAMMA,
                weight=class_weights
            )
            self.use_fedalert = False
            print(f"  Using Focal Loss (alpha={config.FOCAL_LOSS_ALPHA}, gamma={config.FOCAL_LOSS_GAMMA})")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.use_fedalert = False
            print(f"  Using Weighted Cross Entropy Loss")
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        
        # Learning Rate Scheduler
        if config.USE_LR_SCHEDULER:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_SCHEDULER_MIN_LR,
                verbose=True
            )
            print(f"  Using ReduceLROnPlateau scheduler (patience={config.LR_SCHEDULER_PATIENCE})")
        else:
            self.scheduler = None
        
        # Store global model for FedProx
        self.global_model_params = None
        
        # Metrics tracking
        self.prev_loss = float('inf')
        self.comm_optimizer = CommunicationOptimizer()
        self.privacy_preserver = PrivacyPreserver()
        
        # FedAlert metrics tracking
        self.alert_performance_history = []
        self.current_alert_metrics = {
            'precision_at_threshold': 0.0,
            'recall_at_threshold': 0.0,
            'false_alarm_rate': 0.0
        }
        
    def set_parameters(self, parameters):
        """Set model parameters from server"""
        self.model.load_state_dict(parameters)
        
        # Store global model parameters for FedProx
        if self.config.USE_FEDPROX:
            self.global_model_params = copy.deepcopy(parameters)
    
    def unfreeze_layers_for_finetuning(self):
        """Unfreeze model layers for fine-tuning"""
        if self.config.USE_TRANSFER_LEARNING:
            unfreeze_model_layers(self.model, self.config.PRETRAINED_MODEL)
    
    def get_parameters(self):
        """Get model parameters"""
        return copy.deepcopy(self.model.state_dict())
    
    def train(self, epochs):
        """Local training with FedProx proximal term and FedAlert optimization"""
        self.model.train()
        epoch_losses = []
        
        # For FedAlert: track predictions vs labels
        all_probs = []
        all_labels = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            running_base_loss = 0.0
            running_proximal_loss = 0.0
            running_alert_loss = 0.0
            running_false_alarm_loss = 0.0
            running_missed_detection_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Loss computation (handles both FedAlert and standard losses)
                if self.use_fedalert:
                    # FedAlertLoss returns (loss, loss_dict)
                    criterion_loss, loss_dict = self.criterion(outputs, labels)
                    base_loss = criterion_loss
                    
                    # Track alert-specific losses
                    running_alert_loss += loss_dict['alert_loss']
                    running_false_alarm_loss += loss_dict['false_alarm_loss']
                    running_missed_detection_loss += loss_dict['missed_detection_loss']
                    running_base_loss += loss_dict['base_loss']
                else:
                    # Standard loss (Focal Loss or Cross Entropy)
                    base_loss = self.criterion(outputs, labels)
                    running_base_loss += base_loss.item()
                
                # Collect predictions for alert metrics computation
                if self.use_fedalert:
                    with torch.no_grad():
                        probs = F.softmax(outputs, dim=1)
                        all_probs.extend(probs[:, 1].cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                # FedProx proximal term: penalize deviation from global model
                proximal_term = 0.0
                if self.config.USE_FEDPROX and self.global_model_params is not None:
                    for name, param in self.model.named_parameters():
                        if name in self.global_model_params:
                            proximal_term += torch.norm(param - self.global_model_params[name].to(self.config.DEVICE)) ** 2
                    
                    proximal_term = (self.config.FEDPROX_MU / 2.0) * proximal_term
                
                # Total loss
                loss = base_loss + proximal_term
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if isinstance(proximal_term, torch.Tensor):
                    running_proximal_loss += proximal_term.item()
            
            epoch_loss = running_loss / len(self.dataloader)
            epoch_losses.append(epoch_loss)
            
            if self.config.VERBOSE:
                avg_base = running_base_loss / len(self.dataloader)
                avg_prox = running_proximal_loss / len(self.dataloader)
                
                if self.use_fedalert:
                    avg_alert = running_alert_loss / len(self.dataloader)
                    avg_fa = running_false_alarm_loss / len(self.dataloader)
                    avg_md = running_missed_detection_loss / len(self.dataloader)
                    print(f"  Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
                    print(f"    Base: {avg_base:.4f}, Alert: {avg_alert:.4f} (FA: {avg_fa:.4f}, MD: {avg_md:.4f}), Prox: {avg_prox:.6f}")
                else:
                    print(f"  Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f} "
                          f"(Base: {avg_base:.4f}, Prox: {avg_prox:.6f})")
        
        avg_loss = np.mean(epoch_losses)
        
        # Compute alert performance metrics for FedAlert
        if self.use_fedalert and len(all_probs) > 0:
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            
            # Predictions at alert threshold
            alert_preds = (all_probs >= self.config.ALERT_THRESHOLD).astype(int)
            
            # True positives, false positives, false negatives
            tp = np.sum((alert_preds == 1) & (all_labels == 1))
            fp = np.sum((alert_preds == 1) & (all_labels == 0))
            fn = np.sum((alert_preds == 0) & (all_labels == 1))
            tn = np.sum((alert_preds == 0) & (all_labels == 0))
            
            # Metrics
            precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            self.current_alert_metrics = {
                'precision_at_threshold': precision_at_threshold,
                'recall_at_threshold': recall_at_threshold,
                'false_alarm_rate': false_alarm_rate,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }
            
            self.alert_performance_history.append(self.current_alert_metrics.copy())
            
            if self.config.VERBOSE:
                print(f"  Client {self.client_id} - Alert Performance:")
                print(f"    Precision@{self.config.ALERT_THRESHOLD}: {precision_at_threshold:.4f}")
                print(f"    Recall@{self.config.ALERT_THRESHOLD}: {recall_at_threshold:.4f}")
                print(f"    False Alarm Rate: {false_alarm_rate:.4f}")
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.config.VERBOSE:
                print(f"  Client {self.client_id} - Current LR: {current_lr:.6f}")
        
        return avg_loss
    
    def compute_update(self, global_params):
        """Compute parameter update (delta)"""
        local_params = self.get_parameters()
        update = OrderedDict()
        
        for name in local_params.keys():
            update[name] = local_params[name] - global_params[name]
        
        return update
        
    def should_send_update(self, current_loss):
        """Adaptive update decision"""
        if not self.config.ADAPTIVE_UPDATE:
            return True
        
        # FIX: Handle the very first update where prev_loss is infinity
        if self.prev_loss == float('inf'):
            self.prev_loss = current_loss
            return True
        
        improvement = (self.prev_loss - current_loss) / self.prev_loss if self.prev_loss != 0 else float('inf')
        self.prev_loss = current_loss
        
        return improvement > self.config.UPDATE_THRESHOLD
    
    def compress_update(self, update):
        """Compress update for communication efficiency"""
        # Apply top-k sparsification
        sparse_update, comp_ratio = self.comm_optimizer.top_k_sparsification(
            update, self.config.COMPRESSION_RATIO
        )
        
        return sparse_update, comp_ratio

# =====================================================================
# FEDERATED LEARNING SERVER
# =====================================================================

class FederatedServer:
    """Central server for federated learning"""
    
    def __init__(self, config):
        self.config = config
        
        # Model selection: Transfer Learning vs Custom CNN
        if config.USE_TRANSFER_LEARNING:
            self.global_model = create_transfer_learning_model(
                model_name=config.PRETRAINED_MODEL,
                num_classes=config.NUM_CLASSES,
                pretrained=True
            ).to(config.DEVICE)
            print(f"\nServer initialized with {config.PRETRAINED_MODEL} (Transfer Learning)")
        else:
            self.global_model = LightweightCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
            print(f"\nServer initialized with LightweightCNN")
        
        self.comm_optimizer = CommunicationOptimizer()
        
        # Metrics tracking
        self.round_metrics = []
        self.total_bytes_transmitted = 0
        self.layers_unfrozen = False
        
    def get_parameters(self):
        """Get global model parameters"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def unfreeze_layers_for_finetuning(self, round_num):
        """
        Gradually unfreeze model layers for fine-tuning after initial rounds
        
        Args:
            round_num: Current training round
        """
        if (self.config.USE_TRANSFER_LEARNING and 
            self.config.FREEZE_LAYERS and 
            not self.layers_unfrozen and 
            round_num >= self.config.UNFREEZE_AFTER_ROUNDS):
            
            print(f"\n{'='*70}")
            print(f"Round {round_num}: Starting Fine-Tuning Phase")
            print(f"{'='*70}")
            unfreeze_model_layers(self.global_model, self.config.PRETRAINED_MODEL)
            self.layers_unfrozen = True
            print(f"Model will now fine-tune deeper layers on histopathology data\n")
    
    def compute_alert_based_weights(self, clients, client_data_sizes):
        """
        FedAlert Aggregation: Compute client weights based on alert performance
        
        This is a key innovation: instead of weighting by data size alone,
        we weight clients based on their contribution to alert quality.
        
        Weight = (data_size) * (alert_quality_score)
        
        Where alert_quality_score balances precision and recall at threshold:
        - High precision → fewer false alarms
        - High recall → fewer missed detections
        - F1-score at threshold captures this trade-off
        
        Args:
            clients: List of FederatedClient objects
            client_data_sizes: List of data sizes for each client
            
        Returns:
            List of adjusted weights for aggregation
        """
        if not self.config.USE_FEDALERT:
            # Standard FedAvg: weight by data size only
            return client_data_sizes
        
        alert_weights = []
        
        for client, data_size in zip(clients, client_data_sizes):
            # Get alert performance metrics
            metrics = client.current_alert_metrics
            
            precision = metrics.get('precision_at_threshold', 0.0)
            recall = metrics.get('recall_at_threshold', 0.0)
            
            # F1-score at threshold as quality measure
            # This balances precision (low false alarms) and recall (low missed detections)
            if precision + recall > 0:
                f1_at_threshold = 2 * (precision * recall) / (precision + recall)
            else:
                f1_at_threshold = 0.0
            
            # Combined weight: data size * alert quality
            # Add small epsilon to avoid zero weights
            alert_quality_score = f1_at_threshold + 0.1  # Epsilon = 0.1
            combined_weight = data_size * alert_quality_score
            
            alert_weights.append(combined_weight)
        
        # Normalize to sum to total data size (maintains scale compatibility)
        total_data = sum(client_data_sizes)
        total_weight = sum(alert_weights)
        
        if total_weight > 0:
            normalized_weights = [w * (total_data / total_weight) for w in alert_weights]
        else:
            # Fallback to data size if all weights are zero
            normalized_weights = client_data_sizes
        
        return normalized_weights
    
    def aggregate_updates(self, client_updates, client_weights, clients=None):
        """
        FedAlert Aggregation with alert-performance-based weighting
        
        Args:
            client_updates: List of compressed client updates
            client_weights: List of client data sizes
            clients: List of FederatedClient objects (needed for FedAlert)
        """
        global_params = self.get_parameters()
        
        # Compute alert-based weights if FedAlert is enabled
        if self.config.USE_FEDALERT and clients is not None:
            effective_weights = self.compute_alert_based_weights(clients, client_weights)
            
            if self.config.VERBOSE:
                print("\n  FedAlert Aggregation Weights:")
                for i, (original, adjusted) in enumerate(zip(client_weights, effective_weights)):
                    ratio = adjusted / original if original > 0 else 0
                    print(f"    Client {i}: {original:.0f} → {adjusted:.2f} (×{ratio:.2f})")
        else:
            effective_weights = client_weights
        
        # Initialize aggregated update
        aggregated_update = OrderedDict()
        for name in global_params.keys():
            aggregated_update[name] = torch.zeros_like(global_params[name])
        
        # Weighted averaging
        total_weight = sum(effective_weights)
        
        for update, weight in zip(client_updates, effective_weights):
            # Decompress sparse updates
            dense_update = self.comm_optimizer.decompress_sparse_params(update)
            
            for name in aggregated_update.keys():
                # FIX: Skip non-floating point tensors (like num_batches_tracked)
                if not aggregated_update[name].is_floating_point():
                    continue
                    
                # Handle dtype mismatch by converting to the same dtype as aggregated_update
                update_tensor = dense_update[name].to(self.config.DEVICE)
                
                # Convert to the same dtype as the aggregated_update tensor
                if update_tensor.dtype != aggregated_update[name].dtype:
                    update_tensor = update_tensor.to(aggregated_update[name].dtype)
                
                aggregated_update[name] += update_tensor * (weight / total_weight)
        
        # Update global model
        for name in global_params.keys():
            # FIX: Only update floating-point parameters
            if global_params[name].is_floating_point():
                global_params[name] += aggregated_update[name]
        
        self.global_model.load_state_dict(global_params)
        
        return global_params
    
    def evaluate(self, test_loader, verbose=False):
        """Evaluate global model"""
        self.global_model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        total_time = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                start_time = time.time()
                
                images = images.to(self.config.DEVICE)
                outputs = self.global_model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                total_time += (time.time() - start_time)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Diagnostic information
        if verbose:
            unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
            unique_labels, label_counts = np.unique(all_labels, return_counts=True)
            print(f"  [Diagnostic] Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
            print(f"  [Diagnostic] True label distribution: {dict(zip(unique_labels, label_counts))}")
            print(f"  [Diagnostic] Avg prob for class 0: {all_probs[:, 0].mean():.4f}")
            print(f"  [Diagnostic] Avg prob for class 1: {all_probs[:, 1].mean():.4f}")
            print(f"  [Diagnostic] Max prob for class 1: {all_probs[:, 1].max():.4f}")
            print(f"  [Diagnostic] Min prob for class 1: {all_probs[:, 1].min():.4f}")
            
            # Count how many samples have prob > 0.5 for class 1
            prob_class1_above_threshold = (all_probs[:, 1] > 0.5).sum()
            print(f"  [Diagnostic] Samples with P(tumor) > 0.5: {prob_class1_above_threshold} / {len(all_labels)}")
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auc = 0.5 # Default AUC for single-class predictions
        
        avg_latency = total_time / len(all_labels) if len(all_labels) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'latency': avg_latency
        }
        
        return metrics, all_probs, all_labels

# =====================================================================
# AGENTIC ALERT SYSTEM
# =====================================================================

class AgenticAlertSystem:
    """Real-time monitoring and alert system"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.alert_log = []
        
    def monitor_and_alert(self, image, true_label=None):
        """Monitor single image and trigger alert if malignant"""
        self.model.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.config.DEVICE)
            output = self.model(image)
            prob = torch.softmax(output, dim=1)[0, 1].item()  # Probability of malignant
        
        inference_time = time.time() - start_time
        
        # Check if alert should be triggered
        should_alert = prob > self.config.ALERT_THRESHOLD
        
        alert_info = {
            'timestamp': time.time(),
            'malignant_probability': prob,
            'alert_triggered': should_alert,
            'inference_time': inference_time,
            'true_label': true_label
        }
        
        self.alert_log.append(alert_info)
        
        if should_alert and self.config.VERBOSE:
            print(f"⚠️  ALERT: Malignant tissue detected! (Confidence: {prob:.2%})")
        
        return alert_info
    
    def get_alert_metrics(self):
        """Calculate alert system performance metrics"""
        if not self.alert_log:
            return {}
        
        alerts = [log for log in self.alert_log if log['alert_triggered']]
        
        # Calculate false positive rate
        true_positives = sum(1 for alert in alerts if alert['true_label'] == 1)
        false_positives = sum(1 for alert in alerts if alert['true_label'] == 0)
        
        false_alarm_rate = false_positives / len(alerts) if alerts else 0
        alert_precision = true_positives / len(alerts) if alerts else 0
        avg_delay = np.mean([log['inference_time'] for log in self.alert_log])
        
        return {
            'total_alerts': len(alerts),
            'false_alarm_rate': false_alarm_rate,
            'alert_precision': alert_precision,
            'avg_inference_time': avg_delay
        }

# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================

def main():
    print("="*70)
    print("Edge-aware Federated Histopathology")
    print("Communication-Efficient, Privacy-Preserving Cancer Detection")
    print("="*70)
    print(f"\nDevice: {config.DEVICE}")
    print(f"Clients: {config.NUM_CLIENTS}")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    
    print("\n" + "="*70)
    print("ENHANCEMENTS ACTIVE")
    print("="*70)
    
    print("\n[Tier 1] Training Strategy Improvements:")
    if config.USE_LR_SCHEDULER:
        print(f"  ✓ Learning Rate Scheduler: ReduceLROnPlateau")
        print(f"    - Patience: {config.LR_SCHEDULER_PATIENCE}")
        print(f"    - Factor: {config.LR_SCHEDULER_FACTOR}")
        print(f"    - Min LR: {config.LR_SCHEDULER_MIN_LR}")
    
    print(f"  ✓ Aggressive Class Weighting: {config.CLASS_WEIGHT_MULTIPLIER}x for tumor class")
    
    if config.USE_FEDALERT:
        print(f"  ✓✓✓ FedAlert Algorithm (NOVEL CONTRIBUTION)")
        print(f"    - Alert Threshold: {config.ALERT_THRESHOLD}")
        print(f"    - False Alarm Weight (α): {config.FEDALERT_ALPHA}")
        print(f"    - Missed Detection Weight (β): {config.FEDALERT_BETA}")
        print(f"    - Direct optimization for alert performance")
        print(f"    - Alert-quality-based client aggregation")
    elif config.USE_FOCAL_LOSS:
        print(f"  ✓ Focal Loss Implementation")
        print(f"    - Alpha: {config.FOCAL_LOSS_ALPHA}")
        print(f"    - Gamma: {config.FOCAL_LOSS_GAMMA}")
    
    print("\n[Tier 2] Federated Learning Enhancements:")
    if config.USE_FEDPROX:
        print(f"  ✓ FedProx Algorithm (µ={config.FEDPROX_MU})")
        print(f"    - Stabilizes training with non-IID data")
        print(f"    - Reduces client drift")
    
    print("\n[Tier 3] Architecture and Data Improvements:")
    if config.USE_TRANSFER_LEARNING:
        print(f"  ✓ Transfer Learning: {config.PRETRAINED_MODEL}")
        if config.FREEZE_LAYERS:
            print(f"    - Initial frozen training on classifier only")
            print(f"    - Fine-tuning starts at round {config.UNFREEZE_AFTER_ROUNDS}")
    
    print(f"  ✓ Aggressive Data Augmentation:")
    print(f"    - Random flips (horizontal/vertical)")
    print(f"    - Rotation (±20°)")
    print(f"    - Color jittering (brightness, contrast, saturation, hue)")
    print(f"    - Affine transforms (translation, scaling, shearing)")
    
    print("\n" + "="*70 + "\n")
    
    # ========================
    # Download and Prepare Data
    # ========================
    download_and_prepare_dataset(config)
    
    # ========================
    # Load Data
    # ========================
    print("\nLoading datasets from local parquet files...")
    train_files = get_parquet_files(config.TRAIN_PATH)
    test_files = get_parquet_files(config.TEST_PATH)
    val_files = get_parquet_files(config.VAL_PATH)
    
    # =====================================================================
    # NEW: Find a workable random seed that allocates files to all clients
    # =====================================================================
    print("\nSearching for a random seed that allocates data to all clients...")
    workable_seed = -1
    max_attempts = 10000 # Safety break to prevent an infinite loop

    for seed in range(max_attempts):
        np.random.seed(seed)
        # Create a temporary copy for the simulation, as shuffle works in-place
        temp_files = list(train_files) 
        client_splits_simulation = create_non_iid_splits(temp_files, config.NUM_CLIENTS)

        # Check if any client was allocated zero files
        if all(len(files) > 0 for files in client_splits_simulation):
            workable_seed = seed
            print(f"  SUCCESS: Found workable seed: {workable_seed}. Using this for the experiment.")
            break

    if workable_seed == -1:
        raise RuntimeError(f"Could not find a workable seed in {max_attempts} attempts. "
                           "Consider reducing client count, increasing file count, or changing the Dirichlet alpha.")

    # =====================================================================
    # SET GLOBAL SEED FOR FULL REPRODUCIBILITY using the found seed
    # =====================================================================
    random.seed(workable_seed)
    np.random.seed(workable_seed)
    torch.manual_seed(workable_seed)

    # ========================
    # Create the definitive non-IID splits with the workable seed
    # ========================
    print(f"\nCreating non-IID splits for {config.NUM_CLIENTS} clients using seed {workable_seed}...")
    # This call will now be deterministic and produce the desired split
    client_file_splits = create_non_iid_splits(train_files, config.NUM_CLIENTS)
    
    for i, files in enumerate(client_file_splits):
        print(f"  Client {i}: {len(files)} files")    
    # ========================
    # Initialize Components
    # ========================
    print("\nInitializing federated learning components...")
    
    # Create clients
    clients = []
    for i, files in enumerate(client_file_splits):
        client = FederatedClient(i, files, config)
        clients.append(client)
    
    # Create server
    server = FederatedServer(config)
    
    # Test dataset
    test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = PatchCamelyonDataset(test_files, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Agentic alert system
    alert_system = AgenticAlertSystem(server.global_model, config)
    
    # ========================
    # Federated Training
    # ========================
    print("\n" + "="*70)
    print("Starting Federated Learning")
    print("="*70 + "\n")
    
    all_metrics = []
    communication_stats = []
    
    for round_num in range(config.NUM_ROUNDS):
        print(f"\n{'='*70}")
        print(f"Round {round_num + 1}/{config.NUM_ROUNDS}")
        print(f"{'='*70}")
        
        round_start = time.time()
        
        # Check if we should unfreeze layers for fine-tuning
        server.unfreeze_layers_for_finetuning(round_num + 1)
        
        # Get global parameters
        global_params = server.get_parameters()
        
        # Client updates
        client_updates = []
        client_weights = []
        participating_clients = []  # Track which clients participated
        round_bytes = 0
        clients_participated = 0
        
        for client in clients:
            # Set global parameters
            client.set_parameters(global_params)
            
            # Sync layer freezing status with server
            if server.layers_unfrozen:
                client.unfreeze_layers_for_finetuning()
            
            # Local training
            print(f"\nClient {client.client_id} training...")
            loss = client.train(config.LOCAL_EPOCHS)
            
            # Decide if update should be sent
            if client.should_send_update(loss):
                # Compute update
                update = client.compute_update(global_params)
                
                # Apply privacy (differential privacy)
                if config.USE_DIFFERENTIAL_PRIVACY:
                    update = client.privacy_preserver.add_gaussian_noise(
                        update, 
                        config.DP_NOISE_MULTIPLIER, 
                        config.DP_CLIP_NORM
                    )
                
                # Compress update
                compressed_update, comp_ratio = client.compress_update(update)
                
                # Track communication
                update_size = sum([u['values'].numel() * 4 + u['indices'].numel() * 4 
                                 for u in compressed_update.values()])
                round_bytes += update_size
                
                client_updates.append(compressed_update)
                client_weights.append(len(client.dataset))
                participating_clients.append(client)  # Track participating client
                clients_participated += 1
                
                print(f"  Update sent (compression ratio: {comp_ratio:.2%})")
            else:
                print(f"  Update skipped (insufficient improvement)")
        
        # Server aggregation
        if client_updates:
            print(f"\nAggregating {clients_participated} client updates...")
            server.aggregate_updates(client_updates, client_weights, clients=participating_clients)
        
        # Evaluation
        print("\nEvaluating global model...")
        metrics, probs, labels = server.evaluate(test_loader, verbose=True)
        
        round_time = time.time() - round_start
        
        # Communication statistics
        comm_stats = {
            'round': round_num + 1,
            'bytes_transmitted': round_bytes,
            'clients_participated': clients_participated,
            'round_time': round_time
        }
        communication_stats.append(comm_stats)
        server.total_bytes_transmitted += round_bytes
        
        # Store metrics
        metrics['round'] = round_num + 1
        metrics['bytes_transmitted'] = round_bytes
        metrics['round_time'] = round_time
        all_metrics.append(metrics)
        
        # Print results
        print(f"\nRound {round_num + 1} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Latency:   {metrics['latency']:.4f}s")
        print(f"  Bytes Tx:  {round_bytes/1e6:.2f} MB")
        print(f"  Round Time: {round_time:.2f}s")
    
    # ========================
    # Agentic Alert Testing
    # ========================
    print("\n" + "="*70)
    print("Testing Agentic Alert System")
    print("="*70 + "\n")
    
    alert_system.model = server.global_model
    
    # Test on subset of test data
    print("Running inference with alert monitoring...")
    for i, (images, labels) in enumerate(test_loader):
        if i >= 10:  # Test on 10 batches
            break
        
        for j in range(images.size(0)):
            alert_system.monitor_and_alert(images[j], labels[j].item())
    
    alert_metrics = alert_system.get_alert_metrics()
    
    print("\nAlert System Performance:")
    print(f"  Total Alerts:      {alert_metrics.get('total_alerts', 0)}")
    print(f"  False Alarm Rate:  {alert_metrics.get('false_alarm_rate', 0):.2%}")
    print(f"  Alert Precision:   {alert_metrics.get('alert_precision', 0):.2%}")
    print(f"  Avg Inference:     {alert_metrics.get('avg_inference_time', 0):.4f}s")
    
    # ========================
    # Final Results Summary
    # ========================
    if not all_metrics:
        print("\nNo training rounds completed. Exiting.")
        return
        
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70 + "\n")
    
    final_metrics = all_metrics[-1]
    
    print("Classification Performance:")
    print(f"  Final Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Final Precision: {final_metrics['precision']:.4f}")
    print(f"  Final Recall:    {final_metrics['recall']:.4f}")
    print(f"  Final F1-Score:  {final_metrics['f1']:.4f}")
    print(f"  Final AUC:       {final_metrics['auc']:.4f}")
    
    print("\nCommunication Efficiency:")
    total_bytes_mb = server.total_bytes_transmitted / 1e6
    avg_bytes_per_round = total_bytes_mb / config.NUM_ROUNDS if config.NUM_ROUNDS > 0 else 0
    print(f"  Total Data Transmitted: {total_bytes_mb:.2f} MB")
    print(f"  Avg per Round: {avg_bytes_per_round:.2f} MB")
    
    print("\nLatency:")
    avg_round_time = np.mean([m['round_time'] for m in all_metrics])
    print(f"  Avg Round Time: {avg_round_time:.2f}s")
    print(f"  Avg Inference Latency: {final_metrics['latency']:.4f}s")
    
    # ========================
    # Visualizations
    # ========================
    print("\nGenerating visualizations...")
    
    # Plot 1: Training metrics over rounds
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    rounds = [m['round'] for m in all_metrics]
    
    axes[0, 0].plot(rounds, [m['accuracy'] for m in all_metrics], 'b-o')
    axes[0, 0].set_title('Accuracy over Rounds')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(rounds, [m['auc'] for m in all_metrics], 'r-o')
    axes[0, 1].set_title('AUC over Rounds')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(rounds, [m['f1'] for m in all_metrics], 'g-o')
    axes[1, 0].set_title('F1-Score over Rounds')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(rounds, [m['bytes_transmitted']/1e6 for m in all_metrics], 'm-o')
    axes[1, 1].set_title('Communication Cost per Round')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('MB Transmitted')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('federated_training_metrics.png', dpi=300, bbox_inches='tight')
    print("  Saved: federated_training_metrics.png")
    plt.show()
    
    # Plot 2: Confusion Matrix
    _, final_probs, final_labels = server.evaluate(test_loader)
    final_preds = (np.array(final_probs)[:, 1] > 0.5).astype(int)
    cm = confusion_matrix(final_labels, final_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.title('Confusion Matrix - Final Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  Saved: confusion_matrix.png")
    plt.show()
    
    # Plot 3: ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(final_labels, np.array(final_probs)[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {final_metrics["auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("  Saved: roc_curve.png")
    plt.show()
    
    # ========================
    # Save Results
    # ========================
    if config.SAVE_RESULTS:
        print("\nSaving results...")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv('training_metrics.csv', index=False)
        print("  Saved: training_metrics.csv")
        
        # Save communication stats
        comm_df = pd.DataFrame(communication_stats)
        comm_df.to_csv('communication_stats.csv', index=False)
        print("  Saved: communication_stats.csv")
        
        # Save alert logs
        if alert_system.alert_log:
            alert_df = pd.DataFrame(alert_system.alert_log)
            alert_df.to_csv('alert_logs.csv', index=False)
            print("  Saved: alert_logs.csv")
        
        # Save model
        torch.save(server.global_model.state_dict(), 'final_global_model.pth')
        print("  Saved: final_global_model.pth")

    # Final summary reports...
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()