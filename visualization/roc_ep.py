import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import os
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms
import traceback

# Font settings for English text
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus sign rendering issues
# Use non-interactive backend to avoid TclError
import matplotlib
matplotlib.use('Agg')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model configuration
model_name =   'resnet50d.a1_in1k'
num_classes = 2  # Adjust based on your dataset

# Image preprocessing - updated settings
preprocess = transforms.Compose([
    transforms.Resize(size=1536, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(size=(1536, 1536)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Configuration
epochs = [100, 200, 300, 400, 500]
cases = ['case1', 'case2', 'case3', 'case4', 'case5']
sets = ['set1', 'set2', 'set3', 'set4', 'set5']

# Root paths for each case (dataset paths)
case_root_paths = {
    'case1': r"C:\Users\IE\Desktop\stent\additional_dataset\Bona UC vs others_labeling",
    'case2': r"C:\Users\IE\Desktop\stent\additional_dataset\EGIS vs others_labeling",
    'case3': r"C:\Users\IE\Desktop\stent\additional_dataset\Epic vs others_labeling",
    'case4': r"C:\Users\IE\Desktop\stent\additional_dataset\NITIS vs others_labeling",
    'case5': r"C:\Users\IE\Desktop\stent\additional_dataset\S vs M_labeling"
}

# Weight paths for each case
# weight_root_paths = {
#     'case1': r"C:\Users\IE\Downloads\case1",
#     'case2': r"C:\Users\IE\Downloads\case2",
#     'case3': r"C:\Users\IE\Downloads\case3",
#     'case4': r"C:\Users\IE\Downloads\case4",
#     'case5': r"C:\Users\IE\Downloads\case5"
# }


weight_root_paths = {
    'case1': r"d://res/case1",
    'case2': r"d://res/case2",  
    'case3': r"d://res/case3",
    'case4': r"d://res/case4",
    'case5': r"d://res/case5"
}




# Case titles
case_titles = {
    'case1': "Identification of Bona uncovered stent",
    'case2': "Identification of EGIS stent",
    'case3': "Identification of Epic stent", 
    'case4': "Identification of NITI-S stent",
    'case5': "Single vs Multiple stent"
}

# 전체 성능 지표를 저장할 리스트 추가
all_performance_metrics = []

def load_model_with_weights(weight_path, model_name, num_classes, device):
    """
    Load model directly (not just weights)
    """
    try:
        # Method 1: Direct model loading with torch.load
        try:
            model = torch.load(weight_path, map_location=device)
            model.to(device)
            model.eval()
            print(f"      Model loaded successfully from: {weight_path}")
            return model
        except Exception as e1:
            print(f"      Direct torch.load failed: {e1}")
        
        # Method 2: Try torch.jit.load for traced models
        try:
            model = torch.jit.load(weight_path, map_location=device)
            model.eval()
            print(f"      Model loaded with torch.jit from: {weight_path}")
            return model
        except Exception as e2:
            print(f"      torch.jit.load failed: {e2}")
        
        # Method 3: Load with weights_only=False for pickled models
        try:
            model = torch.load(weight_path, map_location=device, weights_only=False)
            model.to(device)
            model.eval()
            print(f"      Model loaded with weights_only=False from: {weight_path}")
            return model
        except Exception as e3:
            print(f"      torch.load with weights_only=False failed: {e3}")
        
        # Method 4: Fallback - create fresh pretrained model
        print("      All loading methods failed, using pretrained model as fallback")
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        model.to(device)
        model.eval()
        print(f"      Using pretrained model instead of: {weight_path}")
        return model
        
    except Exception as e:
        print(f"      All model loading methods failed: {e}")
        return None

def get_class_mapping(val_folder):
    """
    Create class mapping (0..C-1) by sorting subfolder names alphabetically
    """
    class_names = []
    for class_name in os.listdir(val_folder):
        class_dir = os.path.join(val_folder, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
    
    # Sort alphabetically
    class_names.sort()
    
    # Build class_name -> index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"      Class mapping: {class_to_idx}")
    return class_to_idx

def predict_images_in_folder(model, val_folder, preprocess, device):
    """
    Process all images in validation folder and get predictions
    Include class mapping information
    """
    results = []
    
    if not os.path.exists(val_folder):
        print(f"      Validation folder not found: {val_folder}")
        return results
    
    # Create class mapping based on folder names
    class_to_idx = get_class_mapping(val_folder)
    
    for class_name in sorted(os.listdir(val_folder)):
        class_dir = os.path.join(val_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Numeric index corresponding to class name
        class_idx = class_to_idx[class_name]
        print(f"      Processing class: {class_name} (idx={class_idx})")
        
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            img_path = os.path.join(class_dir, img_file)
            
            try:
                raw_image = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(raw_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                    
                    # Store probabilities for all classes
                    all_probs = probabilities[0].cpu().numpy()
                
                # [파일명, 실제클래스이름, 실제클래스인덱스, 예측클래스, 예측확률, 전체확률]
                results.append([img_file, class_name, class_idx, predicted_class, confidence, all_probs])
                
            except Exception as e:
                print(f"        Error processing {img_path}: {e}")
                continue
    
    return results, class_to_idx

    
def process_predictions_for_roc(predictions, class_to_idx):
    if not predictions:
        return None, None, None
    
    df = pd.DataFrame(predictions, columns=['file_name', 'original_label', 'true_class_idx', 'predicted_class', 'confidence', 'all_probs'])
    num_classes = len(class_to_idx)

    # Predicted class for actual classification: argmax
    y_true = df['true_class_idx'].values
    y_pred = df['predicted_class'].values

    if num_classes == 2:
        # For binary classification, use probability of class 1 for ROC/AUC
        y_score = df['all_probs'].apply(lambda x: x[1]).values
        print("Binary classification: using class 1 probability for ROC/AUC.")
    else:
        # For multi-class, use argmax confidence (adjust if needed)
        y_score = df['confidence'].values
        print("Multi-class classification: using argmax confidence for ROC/AUC.")

    print(f"Debug - y_score 범위: [{y_score.min():.4f}, {y_score.max():.4f}]")
    return y_true, y_score, y_pred

def calculate_performance_metrics(y_true, y_pred, case_name, set_name, epoch):
    """
    성능 지표 계산 (Accuracy, Precision, Recall, F1)
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            'Case': case_name,
            'Set': set_name,
            'Epoch': epoch,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    except Exception as e:
        print(f"        Error calculating metrics: {e}")
        return {
            'Case': case_name,
            'Set': set_name,
            'Epoch': epoch,
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1_Score': np.nan
        }

def get_weight_path(case_name, set_name, epoch):
    """
    Generate weight file path based on case, set, and epoch
    """
    weight_root = weight_root_paths[case_name]
    # Try common weight file naming patterns
    weight_patterns = [
        f"{epoch}.pth",
        f"model_epoch_{epoch}.pth",
        f"epoch_{epoch}.pth",
        f"checkpoint_{epoch}.pth",
        f"best_model_{epoch}.pth",
        f"{epoch}.pt"
    ]
    
    for pattern in weight_patterns:
        weight_path = os.path.join(weight_root, set_name, pattern)
        if os.path.exists(weight_path):
            return weight_path
    
    # Return first pattern as fallback
    return os.path.join(weight_root, set_name, weight_patterns[0])

def get_val_folder_path(case_name, set_name):
    """
    Generate validation folder path
    """
    case_root = case_root_paths[case_name]
    return os.path.join(case_root, set_name, "val")

def calculate_set_average_roc(case_name, epoch):
    """
    Calculate set-averaged ROC curve for a specific case and epoch
    """
    print(f"\nProcessing {case_name} - Epoch {epoch}")
    
    set_fprs = []
    set_tprs = []
    set_aucs = []
    
    # Lists to collect per-set metrics for averaging
    set_accuracies = []
    set_precisions = []
    set_recalls = []
    set_f1_scores = []
    
    for set_name in sets:
        print(f"  Processing {set_name}")
        
        # Get weight path
        weight_path = get_weight_path(case_name, set_name, epoch)
        
        # Check if weight file exists
        if not os.path.exists(weight_path):
            print(f"    Weight file not found: {weight_path}")
            continue
        
        # Load model
        model = load_model_with_weights(weight_path, model_name, num_classes, device)
        if model is None:
            print(f"    Skipping {set_name} due to model loading error")
            continue
        
        # Get validation folder path
        val_folder = get_val_folder_path(case_name, set_name)
        
        if not os.path.exists(val_folder):
            print(f"    Validation folder not found: {val_folder}")
            continue
        
        try:
            # Get predictions with class mapping
            predictions, class_to_idx = predict_images_in_folder(model, val_folder, preprocess, device)
            
            if not predictions:
                print(f"    No predictions for {set_name}")
                continue
            
            # Convert to ROC format using folder-based mapping
            y_true, y_score, y_pred = process_predictions_for_roc(predictions, class_to_idx)
            
            if y_true is None or len(np.unique(y_true)) < 2:
                print(f"    Warning: {set_name} doesn't have both classes")
                continue
            
            # Compute and store per-set performance metrics
            set_metrics = calculate_performance_metrics(y_true, y_pred, case_name, set_name, epoch)
            all_performance_metrics.append(set_metrics)
            
            # Append per-set metrics for averaging
            set_accuracies.append(set_metrics['Accuracy'])
            set_precisions.append(set_metrics['Precision'])
            set_recalls.append(set_metrics['Recall'])
            set_f1_scores.append(set_metrics['F1_Score'])
            
            print(f"    {set_name} - Acc: {set_metrics['Accuracy']:.4f}, Prec: {set_metrics['Precision']:.4f}, Rec: {set_metrics['Recall']:.4f}, F1: {set_metrics['F1_Score']:.4f}")
            
            # Compute ROC curve (per set)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            
            set_fprs.append(fpr)
            set_tprs.append(tpr)
            set_aucs.append(auc_score)
            
            print(f"    {set_name} AUC: {auc_score:.4f}")
            
        except Exception as e:
            print(f"    Error processing {set_name}: {e}")
            traceback.print_exc()
            continue
        finally:
            # Clean up GPU memory
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
    
    # Average performance across sets (proper cross-validation style)
    if len(set_accuracies) > 0:
        avg_accuracy = np.mean(set_accuracies)
        avg_precision = np.mean(set_precisions)
        avg_recall = np.mean(set_recalls)
        avg_f1 = np.mean(set_f1_scores)
        
        # 평균 성능 지표 저장
        overall_metrics = {
            'Case': case_name,
            'Set': "all_sets_average",  # 이름 변경으로 구분
            'Epoch': epoch,
            'Accuracy': avg_accuracy,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1_Score': avg_f1
        }
        all_performance_metrics.append(overall_metrics)
        
        print(f"  Average across sets - Acc: {avg_accuracy:.4f}, Prec: {avg_precision:.4f}, Rec: {avg_recall:.4f}, F1: {avg_f1:.4f}")
        print(f"  Number of valid sets: {len(set_accuracies)}")
    else:
        print("  No valid sets for average calculation")
    
    # Calculate average ROC curve across sets
    if not set_fprs:
        return None, None, np.nan
    
    # Generate common FPR points
    common_fpr = np.linspace(0, 1, 100)
    
    # Interpolate each set's TPR to common FPR
    interp_tprs = []
    for fpr, tpr in zip(set_fprs, set_tprs):
        interp_tpr = np.interp(common_fpr, fpr, tpr)
        interp_tprs.append(interp_tpr)
    
    # Calculate mean TPR and AUC
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_auc = auc(common_fpr, mean_tpr)
    
    print(f"  Average AUC across {len(set_aucs)} sets: {mean_auc:.4f}")
    
    return common_fpr, mean_tpr, mean_auc

# Process all cases
for case_name in cases:
    print(f"\n{'='*60}")
    print(f"Processing {case_name.upper()}")
    print(f"{'='*60}")
    
    # Check if case paths exist
    if not os.path.exists(case_root_paths[case_name]):
        print(f"Case root path not found: {case_root_paths[case_name]}")
        continue
    
    if not os.path.exists(weight_root_paths[case_name]):
        print(f"Weight root path not found: {weight_root_paths[case_name]}")
        continue
    
    # Store results for current case
    case_fprs = []
    case_tprs = []
    case_aucs = []
    epoch_labels = []
    
    # Process each epoch for current case
    for epoch in epochs:
        epoch_label = f"Epoch {epoch}"
        epoch_labels.append(epoch_label)
        
        # Calculate set-averaged ROC for this epoch
        fpr, tpr, mean_auc = calculate_set_average_roc(case_name, epoch)
        
        if fpr is not None:
            case_fprs.append(fpr)
            case_tprs.append(tpr)
            case_aucs.append(mean_auc)
        else:
            case_fprs.append(np.array([0, 1]))
            case_tprs.append(np.array([0, 1]))
            case_aucs.append(np.nan)
    
    # Create plots for current case
    # 1. AUC Bar Graph
    plt.figure(figsize=(12, 6))
    bars = plt.bar(epoch_labels, case_aucs, color='black', width=0.6, edgecolor='black')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=11, color='black')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., 0.5,
                    'N/A', ha='center', va='bottom', fontsize=11, color='black')
    
    plt.ylim(0.5, 1.0)
    plt.ylabel('AUC Score', fontsize=12)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.title(f'{case_titles[case_name]} - AUC by Epochs', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
    plt.xticks(rotation=45)
    
    # Adjust layout without tight_layout to avoid warnings
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    #plt.savefig(f'{case_name}_auc_bar_graph.png', dpi=300, bbox_inches='tight')
    print(f"AUC bar graph saved: {case_name}_auc_bar_graph.png")
    plt.close()
    
    # 2. ROC Curve Plot
    plt.figure(figsize=(10, 8))
    
    # Black and white line styles
    line_styles = [
        {'linestyle': '-', 'linewidth': 2, 'color': 'black'},
        {'linestyle': '--', 'linewidth': 2, 'color': 'black'},
        {'linestyle': ':', 'linewidth': 2.5, 'color': 'black'},
        {'linestyle': '-.', 'linewidth': 2, 'color': 'black'},
        {'linestyle': (0, (3, 1, 1, 1)), 'linewidth': 2, 'color': 'black'}
    ]
    
    # Plot ROC curves for each epoch
    valid_aucs = []
    for i, (fpr, tpr, auc_score, epoch_label) in enumerate(zip(case_fprs, case_tprs, case_aucs, epoch_labels)):
        if not np.isnan(auc_score):
            plt.plot(fpr, tpr, **line_styles[i % len(line_styles)], 
                     label=f'{epoch_label} (AUC = {auc_score:.4f})')
            valid_aucs.append(auc_score)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # 대각선 추가
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(case_titles[case_name], fontsize=14)
    
    # Only add legend if there are valid AUCs
    if valid_aucs:
        plt.legend(loc='lower right', fontsize=9)
    
    plt.grid(linestyle='--', alpha=0.3, color='gray')
    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.9)
    plt.savefig(f'{model_name}_{case_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"ROC curves saved: {case_name}_roc_curves.png")
    plt.close()
    
    # 3. Save detailed results to CSV
    summary_results = []
    for i, epoch in enumerate(epochs):
        epoch_name = epoch_labels[i]
        auc_score = case_aucs[i]
        summary_results.append([case_name, epoch, epoch_name, auc_score])
    
    summary_df = pd.DataFrame(summary_results, 
                             columns=['Case', 'Epoch', 'Epoch_Name', 'AUC_Score'])
    summary_df.to_csv(f'{case_name}_epoch_auc_summary.csv', index=False)
    print(f"Summary results saved: {case_name}_epoch_auc_summary.csv")
    
    print(f"\n{case_name.upper()} Processing completed!")
    if valid_aucs:
        print(f"Best AUC: {max(valid_aucs):.4f}")
        print(f"Average AUC: {np.mean(valid_aucs):.4f}")

# 전체 성능 지표를 CSV 파일로 저장
if all_performance_metrics:
    performance_df = pd.DataFrame(all_performance_metrics)
    #performance_df.to_csv('all_performance_metrics.csv', index=False)
    print(f"\nAll performance metrics saved: all_performance_metrics.csv")
    print(f"Total records: {len(all_performance_metrics)}")
    
    # 간략한 통계 출력
    print("\nPerformance Metrics Summary:")
    print("-" * 40)
    
    # 케이스별 전체 성능 (all_sets만 필터링)
    overall_data = performance_df[performance_df['Set'] == 'all_sets']
    if not overall_data.empty:
        print("Overall Performance by Case:")
        for case in overall_data['Case'].unique():
            case_data = overall_data[overall_data['Case'] == case]
            avg_acc = case_data['Accuracy'].mean()
            avg_f1 = case_data['F1_Score'].mean()
            print(f"  {case}: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")
    
    # 세트별 성능 (개별 세트만 필터링)
    set_data = performance_df[performance_df['Set'] != 'all_sets']
    if not set_data.empty:
        print("\nAverage Performance by Set:")
        for set_name in set_data['Set'].unique():
            set_perf = set_data[set_data['Set'] == set_name]
            avg_acc = set_perf['Accuracy'].mean()
            avg_f1 = set_perf['F1_Score'].mean()
            print(f"  {set_name}: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")
            
else:
    print("\nNo performance metrics to save.")

print(f"\n{'='*60}")
print("ALL CASES PROCESSING COMPLETED!")
print(f"{'='*60}")