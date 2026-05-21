# -*- coding: utf-8 -*-
"""
Computational Complexity Evaluation Script for DST-AgenticNet

This script calculates:
- Parameter count (total, trainable, per-component)
- Inference latency (single image, batch)
- Memory requirements (GPU peak, model file size)
- Training time (from logs)

And documents:
- Dataset version and source
- Label parsing convention
- Split reproduction instructions
- Dataset size and class distribution

Usage:
    python compute_complexity.py --model_path "best_GATED_FUSION.pth" --data_dir "/kaggle/input/breast-histopathology-images"
"""

import os
import sys
import json
import time
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
#                           DATASET DOCUMENTATION
# ==============================================================================
DATASET_INFO = """
================================================================================
                        DATASET DOCUMENTATION
================================================================================

DATASET NAME:
    Breast Histopathology Images (Invasive Ductal Carcinoma - IDC)

SOURCE:
    Kaggle: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    Original: https://doi.org/10.17605/OSF.IO/W98RK

DESCRIPTION:
    The original dataset consisted of 162 whole mount slide images of 
    Breast Cancer (BCa) specimens scanned at 40x. From these, 277,524 
    patches of size 50x50 were extracted (198,738 IDC negative and 
    78,786 IDC positive).

LABEL PARSING CONVENTION:
    Filename format: {patient_id}_idx5_x{x}_y{y}_class{label}.png
    Example: 8975_idx5_x101_y1101_class0.png
    
    Parser code:
        labels = [int(os.path.basename(p).split('_class')[1].split('.')[0]) 
                  for p in all_image_paths]
    
    Breakdown:
        1. os.path.basename(p) → "8975_idx5_x101_y1101_class0.png"
        2. .split('_class') → ["8975_idx5_x101_y1101", "0.png"]
        3. [1] → "0.png"
        4. .split('.')[0] → "0"
        5. int(...) → 0
    
    Classes:
        0 = Non-IDC (Negative for invasive ductal carcinoma)
        1 = IDC (Positive for invasive ductal carcinoma)

SPLIT REPRODUCTION INSTRUCTIONS:
    Random Seed: 42
    Stratification: By label (class-balanced splits)
    
    Code to reproduce exact splits:
    ```python
    import os
    import glob
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    RANDOM_SEED = 42
    DATA_DIR = '/kaggle/input/breast-histopathology-images'
    
    # Load all image paths
    all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)
    
    # Parse labels from filenames
    labels = [int(os.path.basename(p).split('_class')[1].split('.')[0]) 
              for p in all_image_paths]
    
    # Create dataframe
    df = pd.DataFrame({'path': all_image_paths, 'label': labels})
    
    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['label'], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=RANDOM_SEED
    )
    ```
    
    Expected split ratios:
        - Training:   70% of total
        - Validation: 15% of total
        - Testing:    15% of total

================================================================================
"""

# ==============================================================================
#                           MODEL ARCHITECTURE
# ==============================================================================

class GatedFusion(nn.Module):
    """Attentional Gating to weigh agents based on their consensus/feature strength."""
    def __init__(self, num_agents, num_classes, embedding_dim=64):
        super(GatedFusion, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_agents * num_classes, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_agents),
            nn.Softmax(dim=1)
        )

    def forward(self, evidences):
        batch_size, n_agents, n_classes = evidences.shape
        flat_evidences = evidences.view(batch_size, -1)
        weights = self.gate_net(flat_evidences)
        weights_expanded = weights.unsqueeze(2)
        weighted_evidence = evidences * weights_expanded
        fused_evidence = torch.sum(weighted_evidence, dim=1)
        return fused_evidence, weights


class DST_AgenticNet(nn.Module):
    """Dynamic Uncertainty-Gated Fusion Network with multi-agent evidence combination."""
    def __init__(self, num_classes=2, mode='GATED_FUSION'):
        super(DST_AgenticNet, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        
        # Agent 1: RGB (ResNet18)
        base_model = models.resnet18(pretrained=True)
        self.rgb_extractor = nn.Sequential(*list(base_model.children())[:-1]) 
        self.rgb_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes), nn.Softplus()
        )
        
        # Agent 2: Edge (Structural)
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.edge_head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Softplus()
        )
        
        # Agent 3: Freq (Spectral)
        self.freq_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes), nn.Softplus()
        )
        
        # Fusion Strategy
        self.gate = GatedFusion(num_agents=3, num_classes=num_classes)
        
        # Fixed Sobel filters
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
        
        if self.mode == 'RGB_ONLY':
            return e_rgb, all_evidences, None
        elif self.mode == 'NAIVE_FUSION':
            return torch.sum(all_evidences, dim=1), all_evidences, None
        elif self.mode == 'GATED_FUSION':
            fused, weights = self.gate(all_evidences)
            return fused, all_evidences, weights
            
        return e_rgb, all_evidences, None


# ==============================================================================
#                           COMPLEXITY METRICS
# ==============================================================================

def count_parameters(model):
    """Count total and trainable parameters, broken down by component."""
    
    # Total counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Component breakdown
    components = {
        'RGB Agent (ResNet18 Backbone)': model.rgb_extractor,
        'RGB Head': model.rgb_head,
        'Edge Extractor': model.edge_extractor,
        'Edge Head': model.edge_head,
        'Frequency Head': model.freq_head,
        'Gating Network': model.gate,
    }
    
    component_params = {}
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        component_params[name] = {'total': params, 'trainable': trainable}
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'components': component_params
    }


def measure_inference_latency(model, device, image_size=96, num_warmup=10, num_runs=100):
    """Measure inference latency for single image and various batch sizes."""
    
    model.eval()
    batch_sizes = [1, 8, 32, 64, 128]
    latency_results = {}
    
    for batch_size in batch_sizes:
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
        
        latency_results[f'batch_{batch_size}'] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'per_image_ms': np.mean(times) / batch_size
        }
    
    return latency_results


def measure_memory(model, device, image_size=96, batch_size=32):
    """Measure GPU memory usage during inference."""
    
    memory_results = {}
    
    if device.type == 'cuda':
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Get baseline memory
        baseline_memory = torch.cuda.memory_allocated()
        
        # Model memory (already on GPU)
        model_memory = torch.cuda.memory_allocated() - baseline_memory
        
        # Run inference
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Peak memory during inference
        peak_memory = torch.cuda.max_memory_allocated()
        
        memory_results = {
            'model_memory_mb': model_memory / (1024 ** 2),
            'peak_memory_mb': peak_memory / (1024 ** 2),
            'inference_memory_mb': (peak_memory - model_memory) / (1024 ** 2),
            'batch_size_measured': batch_size
        }
    else:
        memory_results = {
            'note': 'Memory measurement only available on CUDA devices'
        }
    
    return memory_results


def get_model_file_size(model_path):
    """Get the size of the model checkpoint file."""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return {
            'file_size_bytes': size_bytes,
            'file_size_mb': size_bytes / (1024 ** 2)
        }
    return {'error': f'Model file not found: {model_path}'}


def get_dataset_statistics(data_dir):
    """Calculate dataset size and class distribution."""
    
    all_image_paths = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    
    if not all_image_paths:
        return {'error': f'No images found in {data_dir}'}
    
    # Parse labels
    labels = []
    for p in all_image_paths:
        try:
            label = int(os.path.basename(p).split('_class')[1].split('.')[0])
            labels.append(label)
        except (IndexError, ValueError):
            continue
    
    df = pd.DataFrame({'path': all_image_paths[:len(labels)], 'label': labels})
    
    # Calculate splits (for documentation)
    RANDOM_SEED = 42
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=RANDOM_SEED)
    
    class_distribution = df['label'].value_counts().to_dict()
    
    return {
        'total_images': len(df),
        'training_set_size': len(train_df),
        'validation_set_size': len(val_df),
        'test_set_size': len(test_df),
        'class_distribution': {
            'class_0_non_idc': class_distribution.get(0, 0),
            'class_1_idc': class_distribution.get(1, 0)
        },
        'class_imbalance_ratio': max(class_distribution.values()) / min(class_distribution.values()) if class_distribution else 0,
        'random_seed': RANDOM_SEED,
        'stratification': 'By label (class-balanced)',
        'split_ratios': '70% train / 15% val / 15% test'
    }


def calculate_flops(model, device, image_size=96):
    """Estimate FLOPs using a forward pass counter (optional, requires thop)."""
    try:
        from thop import profile, clever_format
        dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        return {
            'flops': flops,
            'flops_formatted': flops_formatted,
            'params_formatted': params_formatted
        }
    except ImportError:
        return {'note': 'Install thop for FLOPs calculation: pip install thop'}


def generate_report(metrics, output_path='complexity_report.md'):
    """Generate a markdown report of all metrics."""
    
    report = f"""# DST-AgenticNet Computational Complexity Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

{DATASET_INFO}

## Parameter Count

| Metric | Value |
|--------|-------|
| **Total Parameters** | {metrics['parameters']['total_parameters']:,} |
| **Trainable Parameters** | {metrics['parameters']['trainable_parameters']:,} |
| **Non-trainable Parameters** | {metrics['parameters']['non_trainable_parameters']:,} |

### Component Breakdown

| Component | Total Params | Trainable Params |
|-----------|-------------|------------------|
"""
    
    for comp, counts in metrics['parameters']['components'].items():
        report += f"| {comp} | {counts['total']:,} | {counts['trainable']:,} |\n"
    
    report += f"""
## Inference Latency

Device: {metrics.get('device', 'Unknown')}

| Batch Size | Mean (ms) | Std (ms) | Per-Image (ms) |
|------------|-----------|----------|----------------|
"""
    
    for batch_key, lat in metrics['latency'].items():
        batch_size = batch_key.replace('batch_', '')
        report += f"| {batch_size} | {lat['mean_ms']:.2f} | {lat['std_ms']:.2f} | {lat['per_image_ms']:.3f} |\n"
    
    report += f"""
## Memory Requirements

"""
    
    if 'peak_memory_mb' in metrics['memory']:
        report += f"""| Metric | Value |
|--------|-------|
| Model Memory | {metrics['memory']['model_memory_mb']:.2f} MB |
| Peak Memory (batch={metrics['memory']['batch_size_measured']}) | {metrics['memory']['peak_memory_mb']:.2f} MB |
| Inference Memory | {metrics['memory']['inference_memory_mb']:.2f} MB |
"""
    else:
        report += f"Note: {metrics['memory'].get('note', 'Memory metrics not available')}\n"
    
    if 'file_size_mb' in metrics['model_file']:
        report += f"""
### Model File Size

| Metric | Value |
|--------|-------|
| File Size | {metrics['model_file']['file_size_mb']:.2f} MB |
"""
    
    report += f"""
## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | {metrics['dataset'].get('total_images', 'N/A'):,} |
| Training Set | {metrics['dataset'].get('training_set_size', 'N/A'):,} |
| Validation Set | {metrics['dataset'].get('validation_set_size', 'N/A'):,} |
| Test Set | {metrics['dataset'].get('test_set_size', 'N/A'):,} |
| Class 0 (Non-IDC) | {metrics['dataset'].get('class_distribution', {}).get('class_0_non_idc', 'N/A'):,} |
| Class 1 (IDC) | {metrics['dataset'].get('class_distribution', {}).get('class_1_idc', 'N/A'):,} |
| Random Seed | {metrics['dataset'].get('random_seed', 42)} |
| Stratification | {metrics['dataset'].get('stratification', 'By label')} |
| Split Ratios | {metrics['dataset'].get('split_ratios', '70/15/15')} |

## FLOPs (if available)

"""
    
    if 'flops_formatted' in metrics.get('flops', {}):
        report += f"Total FLOPs: {metrics['flops']['flops_formatted']}\n"
    else:
        report += f"Note: {metrics.get('flops', {}).get('note', 'FLOPs not calculated')}\n"
    
    report += """
## Reproducibility

To reproduce these metrics, run:

```bash
python compute_complexity.py \\
    --model_path "path/to/best_GATED_FUSION.pth" \\
    --data_dir "/kaggle/input/breast-histopathology-images"
```

## Citation

If you use this model or dataset, please cite:

```bibtex
@misc{dst_agenticnet,
    title={DST-AgenticNet: Dynamic Uncertainty-Gated Multi-Agent Fusion for Medical Image Classification},
    author={Your Name},
    year={2024}
}
```
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Compute complexity metrics for DST-AgenticNet')
    parser.add_argument('--model_path', type=str, default='best_GATED_FUSION.pth',
                        help='Path to the saved model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/breast-histopathology-images',
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=96,
                        help='Input image size (default: 96)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output files')
    parser.add_argument('--mode', type=str, default='GATED_FUSION',
                        choices=['RGB_ONLY', 'NAIVE_FUSION', 'GATED_FUSION'],
                        help='Model mode (default: GATED_FUSION)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("\n[1/6] Initializing model...")
    model = DST_AgenticNet(num_classes=2, mode=args.mode).to(device)
    
    # Load checkpoint if exists
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}")
        print("Using randomly initialized model for complexity calculation")
    
    model.eval()
    
    # Collect all metrics
    all_metrics = {'device': str(device)}
    
    # Parameter count
    print("\n[2/6] Counting parameters...")
    all_metrics['parameters'] = count_parameters(model)
    print(f"  Total parameters: {all_metrics['parameters']['total_parameters']:,}")
    
    # Inference latency
    print("\n[3/6] Measuring inference latency...")
    all_metrics['latency'] = measure_inference_latency(model, device, args.image_size)
    print(f"  Single image latency: {all_metrics['latency']['batch_1']['mean_ms']:.2f} ms")
    
    # Memory
    print("\n[4/6] Measuring memory usage...")
    all_metrics['memory'] = measure_memory(model, device, args.image_size)
    if 'peak_memory_mb' in all_metrics['memory']:
        print(f"  Peak GPU memory: {all_metrics['memory']['peak_memory_mb']:.2f} MB")
    
    # Model file size
    print("\n[5/6] Getting model file size...")
    all_metrics['model_file'] = get_model_file_size(args.model_path)
    if 'file_size_mb' in all_metrics['model_file']:
        print(f"  Model file size: {all_metrics['model_file']['file_size_mb']:.2f} MB")
    
    # Dataset statistics
    print("\n[6/6] Calculating dataset statistics...")
    all_metrics['dataset'] = get_dataset_statistics(args.data_dir)
    if 'total_images' in all_metrics['dataset']:
        print(f"  Total images: {all_metrics['dataset']['total_images']:,}")
        print(f"  Class 0 (Non-IDC): {all_metrics['dataset']['class_distribution']['class_0_non_idc']:,}")
        print(f"  Class 1 (IDC): {all_metrics['dataset']['class_distribution']['class_1_idc']:,}")
    
    # FLOPs (optional)
    all_metrics['flops'] = calculate_flops(model, device, args.image_size)
    
    # Generate outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save JSON metrics
    json_path = os.path.join(args.output_dir, 'complexity_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {json_path}")
    
    # Generate markdown report
    report_path = os.path.join(args.output_dir, 'complexity_report.md')
    generate_report(all_metrics, report_path)
    
    # Print summary
    print("\n" + "="*60)
    print("                    SUMMARY")
    print("="*60)
    print(f"Total Parameters:     {all_metrics['parameters']['total_parameters']:,}")
    print(f"Trainable Parameters: {all_metrics['parameters']['trainable_parameters']:,}")
    print(f"Single Image Latency: {all_metrics['latency']['batch_1']['mean_ms']:.2f} ms")
    if 'peak_memory_mb' in all_metrics['memory']:
        print(f"Peak GPU Memory:      {all_metrics['memory']['peak_memory_mb']:.2f} MB")
    if 'file_size_mb' in all_metrics['model_file']:
        print(f"Model File Size:      {all_metrics['model_file']['file_size_mb']:.2f} MB")
    print("="*60)


if __name__ == "__main__":
    main()
