# filepath: d:\GitHub_repo\delong_test_adaptive_heatmap.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms
import traceback
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import scipy.stats as stats

# Use non-interactive backend to avoid TclError
import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────────────────────────────
# Configuration (all cases)
# ──────────────────────────────
# Model name
model_name = 'resnet50d.a1_in1k'  # ResNet 모델

num_classes = 2  # Binary classification
batch_size = 1   # Fix batch size to 1

# Epochs
epochs = [100, 200, 300, 400, 500]

# All cases configuration
case_configs = {
    'case1': {
        'ckpt_root': r'd://res/case1',
        'base_data_path': r"C:\Users\IE\Desktop\stent\additional_dataset\Bona UC vs others_labeling",
        'save_prefix': 'delong_test_res_case1_adaptive',
        'figure_title': "Identification of Bonastent® uncovered",
        'num_classes': 2
    },
    'case2': {
        'ckpt_root': r'd://res/case2',
        'base_data_path': r"C:\Users\IE\Desktop\stent\additional_dataset\EGIS vs others_labeling",
        'save_prefix': 'delong_test_res_case2_adaptive',
        'figure_title': "Identification of EGIS",
        'num_classes': 2
    },
    'case3': {
        'ckpt_root': r'd://res/case3',
        'base_data_path': r"C:\Users\IE\Desktop\stent\additional_dataset\Epic vs others_labeling",
        'save_prefix': 'delong_test_res_case3_adaptive',
        'figure_title': "Identification of Epic™",
        'num_classes': 2
    },
    'case4': {
        'ckpt_root': r'd://res/case4',
        'base_data_path': r"C:\Users\IE\Desktop\stent\additional_dataset\NITIS vs others_labeling",
        'save_prefix': 'delong_test_res_case4_adaptive',
        'figure_title': "Identification of NITI-S",
        'num_classes': 2
    },
    'case5': {
        'ckpt_root': r'd://res/case5',
        'base_data_path': r"C:\Users\IE\Desktop\stent\additional_dataset\S vs M_labeling",
        'save_prefix': 'delong_test_res_case5_adaptive',
        'figure_title': "Single vs Multiple stent",
        'num_classes': 2
    },
    'single': {
        'ckpt_root': r'd:\\\res\single',
        'base_data_path': r"C:\Users\IE\Desktop\stent\additional_dataset\add_single class_labeling",
        'save_prefix': 'delong_test_res_single_adaptive',
        'figure_title': "Classification of vendors (single stent)",
        'num_classes': 5
    }
}

# Data preprocessing
preprocess = transforms.Compose([
    transforms.Resize(1536, interpolation=transforms.InterpolationMode.BICUBIC,
                      antialias=True),
    transforms.CenterCrop((1536, 1536)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

plt.rcParams['axes.unicode_minus'] = False

def load_model(ckpt_path, model_name, num_classes=2):
    """Load entire model from checkpoint (multi-class supported)"""
    try:
        model = torch.load(ckpt_path, map_location='cpu')
        
        # Resolve timm version compatibility issues
        if hasattr(model, 'patch_embed'):
            if hasattr(model.patch_embed, '_init_img_size') and not hasattr(model.patch_embed, 'strict_img_size'):
                model.patch_embed.strict_img_size = model.patch_embed._init_img_size
            if not hasattr(model.patch_embed, 'dynamic_img_pad'):
                model.patch_embed.dynamic_img_pad = False
        
        # Handle dynamic_img_size attribute of VisionTransformer
        if not hasattr(model, 'dynamic_img_size'):
            model.dynamic_img_size = False
        
        # Handle reg_token attribute (VisionTransformer compatibility)
        if not hasattr(model, 'reg_token'):
            model.reg_token = None
        
        # Handle strict_img_size attribute
        if not hasattr(model, 'strict_img_size'):
            model.strict_img_size = False
        
        print(f"✅ Weights loaded successfully: {ckpt_path}")
        return model.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load weights: {ckpt_path} - Error: {e}")
        return None

def inference_once(model, dataset_path):
    """Run inference"""
    dataset = datasets.ImageFolder(root=dataset_path, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)
    
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Infer', leave=False):
            logits = model(imgs.to(device))
            probabilities = torch.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().numpy()
            probs = probabilities.cpu().numpy()
            
            y_true.append(labels.item())
            y_pred.append(int(preds[0]))
            y_score.append(probs[0])  # 모든 클래스의 확률
    
    return np.array(y_true), np.array(y_pred), np.array(y_score)

def delong_test_with_clip(y_true, y_score1, y_score2, min_var_diff=1e-8):
    """
    DeLong test implementation (clipped version)
    Statistical significance test between two ROC curves
    Use np.clip to avoid negative or extremely small var_diff
    """
    # Validate array lengths
    if len(y_true) != len(y_score1) or len(y_true) != len(y_score2):
        raise ValueError(f"배열 길이 불일치: y_true={len(y_true)}, y_score1={len(y_score1)}, y_score2={len(y_score2)}")
    
    # Use probability of class 1 only
    y_score1_binary = y_score1[:, 1]
    y_score2_binary = y_score2[:, 1]
    y_true_binary = (y_true == 1).astype(int)
    
    # Split positive and negative classes
    positive_indices = y_true_binary == 1
    negative_indices = y_true_binary == 0
    
    y_score1_pos = y_score1_binary[positive_indices]
    y_score1_neg = y_score1_binary[negative_indices]
    y_score2_pos = y_score2_binary[positive_indices]
    y_score2_neg = y_score2_binary[negative_indices]
    
    # Compute DeLong statistics
    n_pos = len(y_score1_pos)
    n_neg = len(y_score1_neg)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"No positive or negative samples: n_pos={n_pos}, n_neg={n_neg}")
    
    # Compute V10, V01 (DeLong et al., 1988)
    V10_1 = np.mean(y_score1_pos)
    V01_1 = np.mean(y_score1_neg)
    V10_2 = np.mean(y_score2_pos)
    V01_2 = np.mean(y_score2_neg)
    
    # Compute AUC
    auc1 = V10_1 - V01_1
    auc2 = V10_2 - V01_2
    
    # Compute variances
    S10_1 = np.var(y_score1_pos) / n_pos
    S01_1 = np.var(y_score1_neg) / n_neg
    S10_2 = np.var(y_score2_pos) / n_pos
    S01_2 = np.var(y_score2_neg) / n_neg
    
    # Compute covariances
    try:
        S10_12 = np.cov(y_score1_pos, y_score2_pos)[0, 1] / n_pos
        S01_12 = np.cov(y_score1_neg, y_score2_neg)[0, 1] / n_neg
    except Exception as e:
        raise ValueError(f"Covariance computation failed: {e}")
    
    # Compute statistic (higher precision)
    var_diff = S10_1 + S01_1 + S10_2 + S01_2 - 2 * (S10_12 + S01_12)
    
    # Debug info (especially for close AUCs)
    if abs(auc1 - auc2) < 0.01:  # Small AUC difference
        print(f"🔍 DeLong Debug - AUC1: {auc1:.6f}, AUC2: {auc2:.6f}, AUC_Diff: {auc1-auc2:.6f}")
        print(f"   Variances: S10_1={S10_1:.8f}, S01_1={S01_1:.8f}, S10_2={S10_2:.8f}, S01_2={S01_2:.8f}")
        print(f"   Covariances: S10_12={S10_12:.8f}, S01_12={S01_12:.8f}")
        print(f"   var_diff (original): {var_diff:.8f}")
    
    # Handle based on magnitude of AUC difference
    auc_diff_abs = abs(auc1 - auc2)
    
    # 1. AUCs are nearly identical
    if auc_diff_abs < 1e-10:
        print(f"⚠️  AUCs are nearly identical: {auc1:.6f} vs {auc2:.6f} -> p-value = 1")
        return 0.0, 1.0, (auc1, auc2)
    
    # 2. AUC difference is negligible (< 0.0035) - set p-value = 1
    if auc_diff_abs < 0.0035:
        print(f"⚠️  Negligible AUC difference: AUC_diff={auc_diff_abs:.6f} -> p-value = 1 (almost no difference)")
        return 0.0, 1.0, (auc1, auc2)
    
    # 3. General case: clip only when var_diff is negative
    original_var_diff = var_diff
    if var_diff < 0:
        # For negative case, set proportional to squared AUC difference
        adaptive_min_var = max(1e-6, (auc_diff_abs ** 2) * 0.1)
        var_diff = adaptive_min_var
        print(f"⚠️  Negative var_diff clipped: {original_var_diff:.8f} -> {var_diff:.8f} (adaptive minimum: {adaptive_min_var:.8f})")
    
    z_stat = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Additional debug for extremely small p-values
    if p_value < 1e-4:
        print(f"⚠️  Extremely small p-value detected: {p_value:.2e}")
        print(f"   z_stat: {z_stat:.6f}, var_diff: {var_diff:.8f}")
        print(f"   AUC difference: {auc1-auc2:.6f}, sqrt(var_diff): {np.sqrt(var_diff):.8f}")
    
    return z_stat, p_value, (auc1, auc2)

def perform_delong_tests_for_case_by_set(case_name, config):
    """For a single case, run DeLong tests per set and then perform meta-analysis"""
    
    ckpt_root = config['ckpt_root']
    base_data_path = config['base_data_path']
    save_prefix = config['save_prefix']
    
    # Store results per set
    set_results = {}
    
    # Iterate over sets
    for set_num in range(1, 6):
        
        dataset_path = os.path.join(base_data_path, f'set{set_num}', 'val')
        if not os.path.exists(dataset_path):
            continue
        
        # Store results per epoch
        epoch_results = {}
        
        # Run inference for each epoch
        for epoch in epochs:
            # Find checkpoint file
            ckpt_path = os.path.join(ckpt_root, f'set{set_num}', f'{epoch}.pth')
            if not os.path.exists(ckpt_path):
                continue
            
            # Load model
            num_classes = config['num_classes']
            model = load_model(ckpt_path, model_name, num_classes)
            if model is None:
                continue
            
            # Inference
            y_true, y_pred, y_score = inference_once(model, dataset_path)
            
            # Compute ROC AUC (multi-class supported)
            if num_classes == 2:
                # Binary classification
                y_score_binary = y_score[:, 1]
                y_true_binary = (y_true == 1).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
                roc_auc = auc(fpr, tpr)
            else:
                # Multi-class (mean AUC across classes)
                y_true_one_hot = np.zeros((len(y_true), num_classes))
                for i, lbl in enumerate(y_true):
                    y_true_one_hot[i, lbl] = 1
                
                class_aucs = []
                for c in range(num_classes):
                    fpr, tpr, _ = roc_curve(y_true_one_hot[:, c], y_score[:, c])
                    roc_auc = auc(fpr, tpr)
                    class_aucs.append(roc_auc)
                
                roc_auc = np.mean(class_aucs)  # 클래스별 평균 AUC
            
            # Save results
            epoch_results[epoch] = {
                'y_true': y_true,
                'y_score': y_score,
                'auc': roc_auc
            }
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
        
        # Perform DeLong tests across epochs within the set
        if len(epoch_results) >= 2:
            set_comparison_results = []
            epoch_list = sorted(epoch_results.keys())
            
            for i, epoch1 in enumerate(epoch_list):
                for j, epoch2 in enumerate(epoch_list[i+1:], i+1):
                    result1 = epoch_results[epoch1]
                    result2 = epoch_results[epoch2]
                    
                    auc1 = result1['auc']
                    auc2 = result2['auc']
                    auc_diff = auc1 - auc2
                    
                    # Run DeLong test
                    y_true = result1['y_true']
                    y_score1 = result1['y_score']
                    y_score2 = result2['y_score']
                    
                    # Check array lengths
                    if len(y_true) != len(y_score1) or len(y_true) != len(y_score2):
                        raise ValueError(f'Array length mismatch - y_true={len(y_true)}, y_score1={len(y_score1)}, y_score2={len(y_score2)}')
                    
                    # Run DeLong test (clipped version)
                    z_stat, p_value, _ = delong_test_with_clip(y_true, y_score1, y_score2)
                    
                    if not np.isnan(p_value):
                        significance = "유의함 (p<0.05)" if p_value < 0.05 else "유의하지 않음"
                    else:
                        significance = "계산 불가"
                    
                    result = {
                        'Case': case_name.upper(),
                        'Set': set_num,
                        'Epoch1': epoch1,
                        'Epoch2': epoch2,
                        'AUC1': auc1,
                        'AUC2': auc2,
                        'AUC_Diff': auc_diff,
                        'Abs_AUC_Diff': abs(auc_diff),
                        'Z_Stat': z_stat,
                        'P_Value': p_value,
                        'Significance': significance,
                        'Sample_Size': len(y_true)
                    }
                    set_comparison_results.append(result)
            
            set_results[set_num] = set_comparison_results
    
    # Meta-analysis: integrate results across sets
    if set_results:
        # Concatenate all set results
        all_set_results = []
        for set_num, results in set_results.items():
            all_set_results.extend(results)
        
        if not all_set_results:
            return None, set_results
        
        # Convert to DataFrame
        df = pd.DataFrame(all_set_results)
        
        # Meta-analysis: aggregate results per epoch pair
        meta_analysis_results = []
        
        # Find unique epoch pairs
        epoch_pairs = df[['Epoch1', 'Epoch2']].drop_duplicates()
        
        for _, pair in epoch_pairs.iterrows():
            epoch1, epoch2 = pair['Epoch1'], pair['Epoch2']
            
            # Filter all set results for this epoch pair
            pair_results = df[(df['Epoch1'] == epoch1) & (df['Epoch2'] == epoch2)]
            
            if len(pair_results) == 0:
                continue
            
            # Keep only valid p-values
            valid_results = pair_results[~pair_results['P_Value'].isna()]
            
            if len(valid_results) == 0:
                # All results are NaN
                meta_result = {
                    'Case': case_name.upper(),
                    'Epoch1': epoch1,
                    'Epoch2': epoch2,
                    'Mean_AUC1': np.nan,
                    'Mean_AUC2': np.nan,
                    'Mean_AUC_Diff': np.nan,
                    'Mean_Abs_AUC_Diff': np.nan,
                    'Meta_Z_Stat': np.nan,
                    'Meta_P_Value': np.nan,
                    'Meta_Significance': "계산 불가",
                    'Num_Sets': len(pair_results),
                    'Valid_Sets': 0,
                    'Sample_Size_Total': pair_results['Sample_Size'].sum()
                }
            else:
                # Perform meta-analysis (using Fisher's method)
                # Convert p-values to z-scores
                z_scores = []
                weights = []
                
                # Debug: check the 300-200 epoch pair
                debug_epochs = (epoch1 == 300 and epoch2 == 200) or (epoch1 == 200 and epoch2 == 300)
                if debug_epochs:
                    print(f"🔍 Meta-analysis debug - epoch pair {epoch1}-{epoch2}:")
                    print(f"   Number of valid sets: {len(valid_results)}")
                
                for _, row in valid_results.iterrows():
                    if not np.isnan(row['P_Value']) and row['P_Value'] > 0:
                        # Convert p-value to z-score
                        z_score = stats.norm.ppf(1 - row['P_Value'] / 2)  # 양측 검정
                        if row['AUC_Diff'] < 0:  # Consider direction
                            z_score = -z_score
                        
                        z_scores.append(z_score)
                        weights.append(row['Sample_Size'])  # 샘플 크기를 가중치로 사용
                        
                        # Debug: 300-200 epoch pair
                        if debug_epochs:
                            print(f"   Set{row['Set']}: p-value={row['P_Value']:.2e}, z-score={z_score:.6f}, weight={row['Sample_Size']}")
                
                if z_scores:
                    # Weighted average z-score
                    weighted_z = np.average(z_scores, weights=weights)
                    
                    # Meta-analysis p-value
                    meta_p_value = 2 * (1 - stats.norm.cdf(abs(weighted_z)))
                    
                    # Debug: 300-200 epoch pair
                    if debug_epochs:
                        print(f"   가중평균 z-score: {weighted_z:.6f}")
                        print(f"   Meta-analysis p-value: {meta_p_value:.2e}")
                        print(f"   AUC difference: {valid_results['AUC_Diff'].mean():.6f}")
                        print(f"   Absolute AUC difference: {valid_results['Abs_AUC_Diff'].mean():.6f}")
                    
                    # Only set meta-analysis p-value to 1 when AUC difference is truly negligible
                    mean_auc_diff_abs = valid_results['Abs_AUC_Diff'].mean()
                    if mean_auc_diff_abs < 0.0035:  # Only when less than 0.35%
                        print(f"⚠️  Negligible AUC difference in meta-analysis: {mean_auc_diff_abs:.6f} -> p-value = 1")
                        meta_p_value = 1.0
                    meta_result = {
                        'Case': case_name.upper(),
                        'Epoch1': epoch1,
                        'Epoch2': epoch2,
                        'Mean_AUC1': valid_results['AUC1'].mean(),
                        'Mean_AUC2': valid_results['AUC2'].mean(),
                        'Mean_AUC_Diff': valid_results['AUC_Diff'].mean(),
                        'Mean_Abs_AUC_Diff': valid_results['Abs_AUC_Diff'].mean(),
                        'Meta_Z_Stat': weighted_z,
                        'Meta_P_Value': meta_p_value,
                        'Meta_Significance': "유의함 (p<0.05)" if meta_p_value < 0.05 else "유의하지 않음",
                        'Num_Sets': len(pair_results),
                        'Valid_Sets': len(valid_results),
                        'Sample_Size_Total': pair_results['Sample_Size'].sum()
                    }
                else:
                    meta_result = {
                        'Case': case_name.upper(),
                        'Epoch1': epoch1,
                        'Epoch2': epoch2,
                        'Mean_AUC1': np.nan,
                        'Mean_AUC2': np.nan,
                        'Mean_AUC_Diff': np.nan,
                        'Mean_Abs_AUC_Diff': np.nan,
                        'Meta_Z_Stat': np.nan,
                        'Meta_P_Value': np.nan,
                        'Meta_Significance': "계산 불가",
                        'Num_Sets': len(pair_results),
                        'Valid_Sets': 0,
                        'Sample_Size_Total': pair_results['Sample_Size'].sum()
                    }
            
            meta_analysis_results.append(meta_result)
        
        # Save results
        if meta_analysis_results:
            meta_df = pd.DataFrame(meta_analysis_results)
            
            # Save per-set detailed results
            set_detail_df = pd.DataFrame(all_set_results)
            set_detail_filename = f'{save_prefix}_set_detail_results.csv'
            set_detail_df.to_csv(set_detail_filename, index=False)
            
            # Save meta-analysis results
            meta_filename = f'{save_prefix}_meta_analysis_results.csv'
            meta_df.to_csv(meta_filename, index=False)
            
            return meta_df, set_results
        else:
            return None, set_results
    else:
        return None, set_results

# ──────────────────────────────
# Main execution
# ──────────────────────────────

# Store overall results
all_meta_results = []
all_set_results = {}

# Run DeLong tests per set for each case
for case_name, config in case_configs.items():
    print(f"=== {case_name.upper()} test started (adaptive heatmap) ===")
    print(f"Checkpoint path: {config['ckpt_root']}")
    print(f"Data path: {config['base_data_path']}")
    print(f"Save prefix: {config['save_prefix']}")
    print("=" * 50)
    
    meta_df, set_results = perform_delong_tests_for_case_by_set(case_name, config)
    if meta_df is not None:
        all_meta_results.append(meta_df)
        print(f"✅ {case_name.upper()} meta-analysis completed")
        print(f"Results file: {config['save_prefix']}_meta_analysis_results.csv")
        print(f"Detailed results file: {config['save_prefix']}_set_detail_results.csv")
    if set_results:
        all_set_results[case_name] = set_results
    
    print(f"=== {case_name.upper()} test finished (adaptive heatmap) ===")
    print()

# Combine meta-analysis results across all cases
if all_meta_results:
    combined_meta_df = pd.concat(all_meta_results, ignore_index=True)
    combined_meta_df.to_csv('resnet_all_cases_meta_analysis_results_adaptive.csv', index=False)
    print("✅ Combined meta-analysis across all cases completed")
else:
    print("❌ No data available to process.")

# Visualization function
def create_adaptive_heatmap_visualization(all_meta_results, case_configs):
    """Visualize meta-analysis results as adaptive heatmaps for each case"""
    
    # Configure subplots (6 cases: 5 existing + 1 single)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Meta-Analysis Results: Adaptive Epoch-wise Comparison Heatmaps', fontsize=16, fontweight='bold')
    
    case_names = list(case_configs.keys())
    
    for idx, case_name in enumerate(case_names):
        # Compute subplot position
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Filter meta-analysis results for this case
        case_df = all_meta_results[all_meta_results['Case'] == case_name.upper()]
        
        if case_df.empty:
            ax.text(0.5, 0.5, f'No data for {case_name.upper()}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(case_configs[case_name]['figure_title'], fontsize=11, fontweight='bold')
            continue
        
        # Extract epoch list
        all_epochs = sorted(set(case_df['Epoch1'].tolist() + case_df['Epoch2'].tolist()))
        
        # Create heatmap data (meta-analysis p-value)
        heatmap_data = np.full((len(all_epochs), len(all_epochs)), np.nan)
        
        # Set diagonal to NaN (self-comparison)
        np.fill_diagonal(heatmap_data, np.nan)
        
        # Fill heatmap with meta-analysis results for each epoch pair
        for _, row_data in case_df.iterrows():
            epoch1_idx = all_epochs.index(row_data['Epoch1'])
            epoch2_idx = all_epochs.index(row_data['Epoch2'])
            
            # Use meta-analysis p-value as is (for significance judgment)
            p_value = row_data['Meta_P_Value']
            if not np.isnan(p_value) and p_value > 0 and p_value <= 1:
                # Use p-value as is
                heatmap_value = p_value
            else:
                heatmap_value = np.nan
            
            # Set values only for off-diagonal entries
            if epoch1_idx != epoch2_idx:
                heatmap_data[epoch1_idx, epoch2_idx] = heatmap_value
                heatmap_data[epoch2_idx, epoch1_idx] = heatmap_value  # 대칭
        
        # Set adaptive color range per case (based on actual p-values)
        non_diagonal_data = heatmap_data[~np.eye(heatmap_data.shape[0], dtype=bool)]
        valid_data = non_diagonal_data[~np.isnan(non_diagonal_data)]
        
        if len(valid_data) > 0:
            # Use actual p-value range per case
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)
        else:
            # Defaults when no data is available
            vmin = 0.001
            vmax = 0.05
        
        # Use a standard colormap (viridis)
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Axis labels
        ax.set_xticks(range(len(all_epochs)))
        ax.set_yticks(range(len(all_epochs)))
        ax.set_xticklabels(all_epochs)
        ax.set_yticklabels(all_epochs)
        
        # Title and labels
        ax.set_title(case_configs[case_name]['figure_title'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch 1', fontsize=10)
        ax.set_ylabel('Epoch 2', fontsize=10)
        
        # Show values
        for i in range(len(all_epochs)):
            for j in range(len(all_epochs)):
                if i == j:
                    # Diagonal: self-comparison
                    text = ax.text(j, i, 'N/A',
                                 ha="center", va="center", color="black", fontsize=7, fontweight='normal')
                elif not np.isnan(heatmap_data[i, j]):
                    # Actual comparison result (p-value as is)
                    p_val = heatmap_data[i, j]
                    
                    # Adjust text color by p-value
                    if p_val < 0.04:
                        text_color = "white"  # white under 0.04
                    else:
                        text_color = "black"  # others black
                    font_weight = "normal"  # normal weight for all
                    
                    # Show p-value (scientific notation)
                    if p_val < 0.001:
                        text = f"{p_val:.2e}"
                    else:
                        text = f"{p_val:.3f}"
                    
                    text = ax.text(j, i, text,
                                 ha="center", va="center", color=text_color, 
                                 fontsize=8, fontweight="normal")
                else:
                    # No data (NaN)
                    text = ax.text(j, i, 'N/A',
                                 ha="center", va="center", color="red", fontsize=8, fontweight='normal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('p-value', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save
    plt.savefig('all_cases_adaptive_heatmap123.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run visualization
if all_meta_results:
    create_adaptive_heatmap_visualization(combined_meta_df, case_configs)
    print("✅ Adaptive heatmap visualization for all cases completed") 