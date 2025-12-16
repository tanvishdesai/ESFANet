import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import zipfile
import glob
import shutil

# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    # Path to your preprocessed test data (Same as your main script)
    DATA_PATH = './preprocessed_patchcamelyon/test'
    
    # -----------------------------------------------------------------
    # MODEL PATHS
    # Dictionary mapping 'Legend Name' -> 'Path to .pth file OR .zip file'
    # ideally, compare your Baseline (FedAvg) vs your Best Model (Full)
    # -----------------------------------------------------------------
    MODELS_TO_COMPARE = {
        'Baseline (FedAvg)': './path/to/fedavg_output.zip', 
        'Proposed (FedAlert+Consensus)': './path/to/full_model_output.zip' 
    }
    
    # If you don't have .pth files yet, set this to True to generate 
    # dummy plots just to see the code work (uses random weights)
    USE_DUMMY_MODE = False 

    # Visualization Settings
    NUM_SAMPLES = 2000       # t-SNE is slow; 2000 samples is usually enough for a clean plot
    PERPLEXITY = 30          # Balance between local/global geometry (5-50 is standard)
    IMG_SIZE = 96
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output filename
    OUTPUT_FILE = 'tsne_feature_comparison.png'

config = Config()

# =====================================================================
# MODEL LOADER (Same architecture as your main script)
# =====================================================================
def load_model(model_path, device):
    print(f"Loading model from: {model_path}")

    final_model_path = model_path

    # Check if it is a zip file
    if model_path.endswith('.zip'):
        print(f"  Detected zip file. Extracting...")
        extraction_root = './extracted_models'
        model_filename = os.path.basename(model_path).replace('.zip', '')
        extract_to = os.path.join(extraction_root, model_filename)
        
        # Clean previous extraction if exists to ensure fresh start
        if os.path.exists(extract_to):
            shutil.rmtree(extract_to)
        os.makedirs(extract_to, exist_ok=True)
        
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            # Search for .pth files recursively
            pth_files = glob.glob(os.path.join(extract_to, '**', '*.pth'), recursive=True)
            
            if not pth_files:
                print(f"  Error: No .pth file found in zip {model_path}")
                return None
            
            # Pick the first one (or add logic to be more specific)
            final_model_path = pth_files[0]
            print(f"  Found model .pth file: {final_model_path}")
            
        except Exception as e:
            print(f"  Error extracting zip: {e}")
            return None
    
    # Recreate the architecture used in fdl-2.py
    model = models.mobilenet_v2(pretrained=False) # Pretrained doesn't matter, we load weights
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), 
        nn.Linear(num_features, 2)
    )
    
    if not config.USE_DUMMY_MODE:
        try:
            state_dict = torch.load(final_model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading {final_model_path}: {e}")
            print("Make sure the path is correct and points to a state_dict.")
            return None
            
    model = model.to(device)
    model.eval()
    
    # We need to hook into the features BEFORE the classifier
    # MobileNetV2 features end at model.features
    feature_extractor = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    
    return feature_extractor

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def extract_features(model, dataloader, device, limit=1000):
    features_list = []
    labels_list = []
    count = 0
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            # Get embeddings (latent space)
            embeddings = model(images)
            
            features_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())
            
            count += images.size(0)
            if count >= limit:
                break
                
    features = np.concatenate(features_list, axis=0)[:limit]
    labels = np.concatenate(labels_list, axis=0)[:limit]
    
    return features, labels

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("="*60)
    print("Generating t-SNE Visualization for Research Paper")
    print("="*60)

    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = ImageFolder(root=config.DATA_PATH, transform=transform)
        # Shuffle to get a random mix of classes
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        print(f"Data loaded. Total available: {len(dataset)}. Using subset of: {config.NUM_SAMPLES}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Iterate Models and Compute t-SNE
    results = {}
    
    for name, path in config.MODELS_TO_COMPARE.items():
        print(f"\nProcessing: {name}...")
        
        # Load Model
        feature_model = load_model(path, config.DEVICE)
        if feature_model is None: continue
        
        # Extract Features
        feats, labs = extract_features(feature_model, dataloader, config.DEVICE, limit=config.NUM_SAMPLES)
        
        # Run t-SNE
        print(f"Running t-SNE for {name} (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=config.PERPLEXITY, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(feats)
        
        results[name] = {
            'tsne': tsne_results,
            'labels': labs
        }

    # 3. Plotting
    print("\nGenerating Plot...")
    num_plots = len(results)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    if num_plots == 1: axes = [axes] # Handle single plot case

    for ax, (name, data) in zip(axes, results.items()):
        X = data['tsne']
        y = data['labels']
        
        # Create scatter plot
        # Class 0: Normal, Class 1: Tumor
        scatter = ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Normal', alpha=0.5, s=15, edgecolors='none')
        scatter = ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Tumor', alpha=0.5, s=15, edgecolors='none')
        
        ax.set_title(f"{name}\nLatent Space", fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_FILE, dpi=300)
    print(f"\nSuccess! Plot saved to: {config.OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()