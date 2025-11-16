import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Font settings for English text
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus sign rendering issues

# Path configuration
base_path = r"C:\Users\IE\Desktop\sten_exp\sci_exp\eff_sci\primary\6. Single Classes"

# 다중 클래스 설정 (인덱스 순서대로)
CLASSES = ["Bona UC","Boston", "EGIS", "Taewoong"]  # 0, 1, 2, 3

# ---------------------------------
# 데이터 수집 및 처리
# ---------------------------------
all_y_true = []
all_y_pred = []
all_confidences = []

# 세트별 ROC 데이터 저장
set_roc_data = {}

# 각 세트별 데이터 수집
for i in range(1, 6):
    set_name = f"set{i}"
    file_path = os.path.join(base_path, f"02.16.{set_name}_gradcam_results.csv")
    
    print(f"\nProcessing: {set_name}")
    
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        print(f"Data loaded: {len(df)} rows")
        
        # 실제 레이블과 예측 레이블 추출
        y_true_col = 'original_label'
        y_pred_col = 'predicted_class'
        conf_col = 'confidence'
        
        # 컬럼이 존재하는지 확인
        if y_pred_col not in df.columns:
            print(f"Column '{y_pred_col}' not found. Available columns: {df.columns.tolist()}")
            continue
        
        # 실제 레이블 (클래스명)
        y_true = df[y_true_col].values
        
        # 예측 레이블 (인덱스를 클래스명으로 변환)
        y_pred_indices = df[y_pred_col].values
        y_pred = [CLASSES[int(idx)] if int(idx) < len(CLASSES) else f"Unknown_{idx}" for idx in y_pred_indices]
        
        # 신뢰도 점수
        confidences = df[conf_col].values
        
        print(f"Unique classes in true labels: {np.unique(y_true)}")
        print(f"Unique classes in predicted labels: {np.unique(y_pred)}")
        
        # 세트별 ROC 계산
        y_true_bin = label_binarize(y_true, classes=CLASSES)
        n_classes = len(CLASSES)
        
        # 예측 점수 생성
        y_score = np.zeros((len(y_pred), n_classes))
        
        for j, (pred_class, conf) in enumerate(zip(y_pred, confidences)):
            if pred_class in CLASSES:
                pred_idx = CLASSES.index(pred_class)
                y_score[j, pred_idx] = conf
                # 다른 클래스들에 대해서는 (1-conf)/(n_classes-1)로 분배
                remaining_conf = (1 - conf) / (n_classes - 1)
                for k in range(n_classes):
                    if k != pred_idx:
                        y_score[j, k] = remaining_conf
        
        # 세트별 ROC curve 계산
        set_fpr = {}
        set_tpr = {}
        set_auc = {}
        
        for j in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, j], y_score[:, j])
            set_fpr[j] = fpr
            set_tpr[j] = tpr
            set_auc[j] = auc(fpr, tpr)
        
        set_roc_data[set_name] = {
            'fpr': set_fpr,
            'tpr': set_tpr,
            'auc': set_auc
        }
        
        # 전체 데이터 누적 (전체 평균 계산용)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_confidences.extend(confidences)
        
        # 정확도 계산
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        if np.sum(cm) > 0:
            accuracy = np.trace(cm) / np.sum(cm)
            print(f"Accuracy for {set_name}: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error processing {set_name}: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------------
# ROC Curve 계산 및 시각화
# ---------------------------------
if all_y_true and all_y_pred:
    print(f"\nCalculating ROC curves for {len(all_y_true)} total samples")
    
    # 전체 정확도 계산
    total_matches = sum(1 for t, p in zip(all_y_true, all_y_pred) if t == p)
    overall_accuracy = total_matches / len(all_y_true)
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    try:
        # 전체 데이터에 대한 ROC curve 계산
        y_true_bin = label_binarize(all_y_true, classes=CLASSES)
        n_classes = len(CLASSES)
        
        # 예측 점수 생성 (전체 데이터)
        y_score = np.zeros((len(all_y_pred), n_classes))
        
        for i, (pred_class, conf) in enumerate(zip(all_y_pred, all_confidences)):
            if pred_class in CLASSES:
                pred_idx = CLASSES.index(pred_class)
                y_score[i, pred_idx] = conf
                # 다른 클래스들에 대해서는 (1-conf)/(n_classes-1)로 분배
                remaining_conf = (1 - conf) / (n_classes - 1)
                for j in range(n_classes):
                    if j != pred_idx:
                        y_score[i, j] = remaining_conf
        
        # 각 클래스별 ROC curve와 AUC 계산 (전체 데이터)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 4개 클래스에 대한 평균 ROC curve 계산 (전체 데이터)
        # 공통 FPR 포인트 생성
        common_fpr = np.linspace(0, 1, 100)
        
        # 각 클래스의 TPR을 공통 FPR에 대해 보간
        interp_tprs = []
        for i in range(n_classes):
            interp_tpr = np.interp(common_fpr, fpr[i], tpr[i])
            interp_tprs.append(interp_tpr)
        
        # 평균 TPR 계산
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = auc(common_fpr, mean_tpr)
        
        # ROC curve 시각화 (흑백)
        plt.figure(figsize=(10, 8))
        
        # 흑백 선 스타일 설정 (각 클래스)
        line_styles = [
            {'linestyle': '-', 'linewidth': 2, 'color': 'black'},
            {'linestyle': '--', 'linewidth': 2, 'color': 'black'},
            {'linestyle': ':', 'linewidth': 2.5, 'color': 'black'},
            {'linestyle': '-.', 'linewidth': 2, 'color': 'black'}
        ]
        
        # 각 클래스별 ROC curve (전체 데이터)
        for i, cls in enumerate(CLASSES):
            plt.plot(fpr[i], tpr[i], **line_styles[i % len(line_styles)],
                     label=f'{cls} (AUC = {roc_auc[i]:.3f})')
        
        # 4개 클래스 평균 ROC curve 추가
        plt.plot(common_fpr, mean_tpr, 
                 linestyle='-', linewidth=3, color='gray',
                 label=f'Average (AUC = {mean_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Classification of vendors', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(linestyle='--', alpha=0.3, color='gray')
        plt.tight_layout()
        plt.savefig('roc_curve_classwise.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved: roc_curve_classwise.png")
        
        # AUC 점수 출력
        print(f"\nClass-wise AUC Scores:")
        for i, cls in enumerate(CLASSES):
            print(f"  {cls}: {roc_auc[i]:.4f}")
        
        print(f"\nMean AUC (4 classes): {mean_auc:.4f}")
        
    except Exception as e:
        print(f"Error calculating ROC curve: {e}")
        import traceback
        traceback.print_exc()
    
else:
    print("No data found for ROC curve calculation.")

print("\nROC curve visualization completed!")
plt.show()