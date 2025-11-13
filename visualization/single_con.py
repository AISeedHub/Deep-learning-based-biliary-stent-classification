import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Font settings for English text
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus sign rendering issues

# Path configuration
base_path = r"C:\Users\IE\Desktop\sten_exp\sci_exp\eff_sci\primary\6. Single Classes"

# Multi-class labels (index order)
CLASSES = ["Bona UC","Boston", "EGIS", "NITI-S"]  # 0, 1, 2, 3

# ---------------------------------
# Confusion matrix computation and visualization
# ---------------------------------
# Dict to store confusion matrices for all sets
conf_matrices = {}
all_y_true = []
all_y_pred = []

# Compute confusion matrix for each set
for i in range(1, 6):
    set_name = f"set{i}"
    file_path = os.path.join(base_path, f"02.16.{set_name}_gradcam_results.csv")
    
    print(f"\nProcessing: {set_name}")
    
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print(f"Data loaded: {len(df)} rows")
        print("Columns:", df.columns.tolist())
        
        # Extract ground-truth and predicted labels
        y_true_col = 'original_label'
        y_pred_col = 'predicted_class'
        
        # Validate columns
        if y_pred_col not in df.columns:
            print(f"Column '{y_pred_col}' not found. Available columns: {df.columns.tolist()}")
            continue
        
        # Ground truth (class name)
        y_true = df[y_true_col].values
        
        # Predicted labels (convert index to class name)
        y_pred_indices = df[y_pred_col].values
        y_pred = [CLASSES[int(idx)] if int(idx) < len(CLASSES) else f"Unknown_{idx}" for idx in y_pred_indices]
        
        print(f"Using columns — True: '{y_true_col}', Predicted: '{y_pred_col}'")
        print(f"Unique classes in true labels: {np.unique(y_true)}")
        print(f"Unique predicted indices: {np.unique(y_pred_indices)}")
        print(f"Unique classes in predicted labels: {np.unique(y_pred)}")
        
        # Inspect samples
        print(f"\nFirst 10 samples of {set_name}:")
        for j in range(min(10, len(y_true))):
            print(f"  True: '{y_true[j]}' | Predicted: '{y_pred[j]}' (idx: {y_pred_indices[j]}) | Match: {y_true[j] == y_pred[j]}")
        
        # Count correct predictions
        matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        print(f"Total matches: {matches} out of {len(y_true)} ({matches/len(y_true)*100:.2f}%)")
        
        # Per-class analysis
        print(f"\nClass-wise analysis for {set_name}:")
        for cls in CLASSES:
            true_count = sum(1 for t in y_true if t == cls)
            pred_count = sum(1 for p in y_pred if p == cls)
            correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            print(f"  {cls}: True={true_count}, Predicted={pred_count}, Correct={correct_count}")
        
        # Accumulate for average confusion matrix
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        conf_matrices[set_name] = cm
        
        print(f"Confusion matrix shape: {cm.shape}")
        print(f"Confusion matrix:\n{cm}")
        
        # Visualize (raw count) confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', 
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'Confusion Matrix - {set_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{set_name}.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix for {set_name} saved: confusion_matrix_{set_name}.png")
        
        # Accuracy for the set
        if np.sum(cm) > 0:
            accuracy = np.trace(cm) / np.sum(cm)
            print(f"Accuracy for {set_name}: {accuracy:.4f}")
        else:
            print(f"Accuracy for {set_name}: undefined (no data)")
        
    except Exception as e:
        print(f"Error calculating confusion matrix for {set_name}: {e}")
        import traceback
        traceback.print_exc()

# Average confusion matrix across all sets
if all_y_true and all_y_pred:
    print(f"\nCalculating average confusion matrix from {len(all_y_true)} total samples")
    print(f"Unique classes in combined true labels: {np.unique(all_y_true)}")
    print(f"Unique classes in combined predicted labels: {np.unique(all_y_pred)}")
    
    # Overall match rate
    total_matches = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p)
    print(f"Total overall matches: {total_matches} out of {len(all_y_true)} ({total_matches/len(all_y_true)*100:.2f}%)")
    
    # Compute overall confusion matrix using all data
    avg_cm = confusion_matrix(all_y_true, all_y_pred, labels=CLASSES)
    
    print(f"Average confusion matrix shape: {avg_cm.shape}")
    print(f"Average confusion matrix:\n{avg_cm}")
    
    # Visualize average (raw count) confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Greys', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Average Confusion Matrix - All Sets', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix_average.png', dpi=300, bbox_inches='tight')
    print("Average confusion matrix saved: confusion_matrix_average.png")
    
    # Overall accuracy
    if np.sum(avg_cm) > 0:
        overall_accuracy = np.trace(avg_cm) / np.sum(avg_cm)
        print(f"Overall accuracy: {overall_accuracy:.4f}")
    else:
        print("Overall accuracy: undefined (no data)")
    
else:
    print("No data found for average confusion matrix calculation.")

print("\nAll confusion matrix visualizations completed!")
plt.show()