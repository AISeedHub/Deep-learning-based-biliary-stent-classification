# filepath: d:\GitHub_repo\roc_pr.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score
)
import os
import torch
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms
import traceback
from torch.utils.data import DataLoader
from torchvision import datasets

# Use non-interactive backend to avoid TclError
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ──────────────────────────────
# 설정
# ──────────────────────────────
# 모델명 설정
model_name = 'resnet50.a1_in1k'

num_classes = 2  # 이진분류
batch_size = 1                             # ★ 배치 1로 고정

# 각 케이스별 설정
case_configs = {
    'case1': {
        'ckpt_root': r'D:\ckpts\res\res_case1',
        'base_data_path': r'd://BonaUC_vs_others_labeling',
        'save_prefix': 'roc_pr_res_case1',
        'figure_title': "Identification of Bonastent® uncovered",
        'num_classes': 2
    },
    'case2': {
        'ckpt_root': r'D:\ckpts\res\res_case2',
        'base_data_path': r'd://EGIS_vs_others_labeling',
        'save_prefix': 'roc_pr_res_case2',
        'figure_title': "Identification of EGIS",
        'num_classes': 2
    },
    'case3': {
        'ckpt_root': r'D:\ckpts\res\res_case3',
        'base_data_path': r'd://Epic_vs_others_labeling',
        'save_prefix': 'roc_pr_res_case3',
        'figure_title': "Identification of Epic™",
        'num_classes': 2
    },
    'case4': {
        'ckpt_root': r'D:\ckpts\res\res_case4',
        'base_data_path': r'd://NITIS_vs_others_labeling',
        'save_prefix': 'roc_pr_res_case4',
        'figure_title': "Identification of NITI-S",
        'num_classes': 2
    },
    'case5': {
        'ckpt_root': r'D:\ckpts\res\res_case5',
        'base_data_path': r'd://S_vs_M_labeling',
        'save_prefix': 'roc_pr_res_case5',
        'figure_title': "Single vs Multiple stent",
        'num_classes': 2
    },
    'case6': {
        'ckpt_root': r'D:\ckpts\res\res_single',
        'base_data_path': r"D:\new_single_class_labeling",
        'save_prefix': 'roc_pr_res_case6',
        'figure_title': "Classification of vendors (single stent)",
        'num_classes': 4
    }
}

# 클래스명 리스트 설정 (이진분류)
CLASS_NAMES = ['Class_0', 'Class_1']

# 데이터 전처리
preprocess = transforms.Compose([
    transforms.Resize(1536, interpolation=transforms.InterpolationMode.BICUBIC,
                      antialias=True),
    transforms.CenterCrop((1536, 1536)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

plt.rcParams['axes.unicode_minus'] = False

all_performance_metrics = []

def load_model(ckpt_path, model_name):
    """single_con2.py와 같은 방식으로 모델 로드"""
    try:
        # 먼저 모델 전체를 로드해보기
        model = torch.load(ckpt_path, map_location='cpu')
        print(f"  모델 전체 로드 성공")
    except:
        # 모델 전체 로드가 실패하면 state_dict로 시도
        try:
            model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
            state = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state, dict):
                state = state.get('model_state_dict', state)
                model.load_state_dict(state, strict=False)
                print(f"  state_dict 로드 성공")
            else:
                print(f"  알 수 없는 체크포인트 형식")
                return None
        except Exception as e:
            print(f"  모델 로드 실패: {e}")
            return None
    
    return model.to(device).eval()

def inference_once(model, dataset_path):
    """single_con2.py와 같은 방식으로 추론"""
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

def calculate_performance_metrics(y_true, y_pred, set_name):
    """성능 메트릭 계산"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return {
            'Set': set_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
    except Exception as e:
        print(f"    Error calculating metrics: {e}")
        return {
            'Set': set_name,
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1_Score': np.nan
        }

def plot_all_sets_roc(all_y_true_list, all_y_score_list, label_text, num_classes, save_name, figure_title):
    """모든 세트의 ROC 곡선과 평균 ROC 곡선을 하나의 figure에 그리기"""
    plt.figure(figsize=(10, 8))
    
    # 선 스타일 리스트 (실선과 점선 번갈아가며)
    line_styles = ['-', '--', '-', '--', '-']
    
    # 각 세트별 ROC 커브 데이터 저장
    all_fprs = []
    all_tprs = []
    all_aucs = []
    
    # 각 세트별 ROC 커브 그리기
    for set_idx, (y_true, y_score) in enumerate(zip(all_y_true_list, all_y_score_list)):
        if num_classes == 2:
            # 이진분류에서는 클래스 1의 확률을 사용
            y_score_binary = y_score[:, 1]  # 클래스 1의 확률
            
            # y_true를 이진분류 형식으로 변환 (클래스 1이면 1, 아니면 0)
            y_true_binary = (y_true == 1).astype(int)
            
            # ROC 커브 계산
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            roc_auc = auc(fpr, tpr)
            
            # 데이터 저장 (평균 계산용)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_aucs.append(roc_auc)
            
            # 각 세트별 ROC 커브 그리기 (검정색 선과 점선 번갈아가며)
            line_style = line_styles[set_idx % len(line_styles)]
            plt.plot(fpr, tpr, label=f"Set{set_idx+1} (AUC={roc_auc:.4f})", 
                    linewidth=1.5, color='black', linestyle=line_style, alpha=0.8)
        else:
            # 멀티클래스: 클래스별 평균 AUC 계산
            y_true_one_hot = np.zeros((len(y_true), num_classes))
            for i, lbl in enumerate(y_true):
                y_true_one_hot[i, lbl] = 1
            
            class_aucs = []
            class_fprs = []
            class_tprs = []
            
            for c in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_one_hot[:, c], y_score[:, c])
                roc_auc = auc(fpr, tpr)
                class_aucs.append(roc_auc)
                class_fprs.append(fpr)
                class_tprs.append(tpr)
            
            # 클래스별 평균 AUC
            avg_auc = np.mean(class_aucs)
            
            # 클래스별 ROC 커브의 평균 계산
            min_length = min(len(fpr) for fpr in class_fprs)
            aligned_fprs = []
            aligned_tprs = []
            
            for fpr, tpr in zip(class_fprs, class_tprs):
                if len(fpr) > min_length:
                    indices = np.linspace(0, len(fpr)-1, min_length, dtype=int)
                    aligned_fprs.append(fpr[indices])
                    aligned_tprs.append(tpr[indices])
                else:
                    aligned_fprs.append(fpr)
                    aligned_tprs.append(tpr)
            
            # 평균 FPR, TPR 계산
            avg_fpr = np.mean(aligned_fprs, axis=0)
            avg_tpr = np.mean(aligned_tprs, axis=0)
            
            # 데이터 저장 (평균 계산용)
            all_fprs.append(avg_fpr)
            all_tprs.append(avg_tpr)
            all_aucs.append(avg_auc)
            
            # 각 세트별 ROC 커브 그리기 (검정색 선과 점선 번갈아가며)
            line_style = line_styles[set_idx % len(line_styles)]
            plt.plot(avg_fpr, avg_tpr, label=f"Set{set_idx+1} (AUC={avg_auc:.4f})", 
                    linewidth=1.5, color='black', linestyle=line_style, alpha=0.8)
    
    # 평균 ROC 커브 계산 및 그리기
    if len(all_fprs) > 0:
        # 공통 FPR 포인트 생성
        common_fpr = np.linspace(0, 1, 100)
        
        # 각 세트의 TPR을 공통 FPR에 맞춰 보간
        interp_tprs = []
        for fpr, tpr in zip(all_fprs, all_tprs):
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interp_tprs.append(interp_tpr)
        
        # 평균 TPR 계산
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = auc(common_fpr, mean_tpr)
        
        # AUC의 평균과 분산 계산 (세트별 AUC 리스트 기반)
        if len(all_aucs) > 0:
            auc_mean = float(np.mean(all_aucs))
            auc_var = float(np.var(all_aucs))
            avg_label = f"Average (AUC={auc_mean:.4f} ± {auc_var:.4f})"
            # 콘솔에도 출력
            print(f"  Average AUC across sets: {auc_mean:.4f} ± {auc_var:.4f}")
        else:
            avg_label = f"Average (AUC={mean_auc:.4f})"
        
        # 평균 ROC 커브 그리기 (얇은 검은색 선)
        plt.plot(common_fpr, mean_tpr, label=avg_label, 
                linewidth=1.5, color='black', linestyle='-', alpha=0.9)
    
    # 점선만 그리기 (설명 없음)
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{figure_title}")
    plt.legend(loc='lower right')
    plt.grid(linestyle='--', alpha=0.3)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()



# ──────────────────────────────
# 메인 처리
# ──────────────────────────────
from tqdm import tqdm

# 각 케이스별로 처리
for case_name, config in case_configs.items():
    print(f"\n{'='*60}")
    print(f"Processing {case_name.upper()}")
    print(f"{'='*60}")
    
    ckpt_root = config['ckpt_root']
    base_data_path = config['base_data_path']
    save_prefix = config['save_prefix']
    figure_title = config['figure_title']
    
    # 현재 케이스의 성능 메트릭 초기화
    all_performance_metrics = []
    all_y_true_list = []
    all_y_score_list = []
    curr_num_classes = config['num_classes']
    # case6(멀티클래스)용 AUC 누적 리스트
    if case_name == 'case6' and curr_num_classes > 2:
        per_class_auc_lists = [[] for _ in range(curr_num_classes)]
        per_set_macro_aucs = []
    
    for i in range(1, 6):
        # 체크포인트 경로 (case6는 다른 파일명 패턴 사용)
        if case_name == 'case6':
            ckpt = os.path.join(ckpt_root, f'res_set{i}.pth')
        else:
            ckpt = os.path.join(ckpt_root, f'res_{case_name}_set{i}.pth')
        if not os.path.exists(ckpt):
            print(f'[경고] {ckpt} 없음 → 건너뜀')
            continue
        
        # 데이터셋 경로
        dataset_path = os.path.join(base_data_path, f'set{i}', 'val')
        if not os.path.exists(dataset_path):
            print(f'[경고] {dataset_path} 없음 → 건너뜀')
            continue

        print(f'\n■ {case_name.upper()} Set{i} ROC 분석 시작')
        print(f'  데이터: {dataset_path}')
        print(f'  모델: {ckpt}')
        print(f'  모델 타입: {model_name}')
        
        # 모델 로드
        model = load_model(ckpt, model_name)
        if model is None:
            print(f'  [경고] 모델 로드 실패 → 건너뜀')
            continue
        
        # 추론
        y_true, y_pred, y_score = inference_once(model, dataset_path)
        
        # 성능 메트릭 계산
        metrics = calculate_performance_metrics(y_true, y_pred, f'{case_name}_set{i}')
        all_performance_metrics.append(metrics)
        
        # ROC AUC 계산 (개별 저장 없이)
        if curr_num_classes == 2:
            y_score_binary = y_score[:, 1]  # 클래스 1의 확률
            y_true_binary = (y_true == 1).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            roc_auc = auc(fpr, tpr)
            print(f'  AUC={roc_auc:.4f}')
        else:
            # 멀티클래스: 클래스별 AUC 및 폴드별 매크로 평균 AUC 출력
            y_true_one_hot = np.zeros((len(y_true), curr_num_classes))
            for idx, lbl in enumerate(y_true):
                y_true_one_hot[idx, lbl] = 1
            class_aucs = []
            for c in range(curr_num_classes):
                fpr, tpr, _ = roc_curve(y_true_one_hot[:, c], y_score[:, c])
                class_aucs.append(auc(fpr, tpr))
            macro_auc = float(np.mean(class_aucs))
            print('  Macro AUC={:.4f} | per-class: {}'.format(
                macro_auc, ', '.join([f'c{ci}={a:.4f}' for ci, a in enumerate(class_aucs)])
            ))
            if case_name == 'case6':
                for c in range(curr_num_classes):
                    per_class_auc_lists[c].append(class_aucs[c])
                per_set_macro_aucs.append(macro_auc)

        all_y_true_list.append(y_true)
        all_y_score_list.append(y_score)

    # ──────────────────────────────
    # 현재 케이스 결과 저장
    # ──────────────────────────────
    if all_performance_metrics:
        df = pd.DataFrame(all_performance_metrics)
        #df.to_csv(f"{save_prefix}_all_performance_metrics.csv", index=False)
        print(f"\n성능 메트릭 저장 → {save_prefix}_all_performance_metrics.csv")
        
        # case6: 클래스별 AUC 평균±표준편차 및 폴드별 매크로 평균 출력
        if case_name == 'case6' and curr_num_classes > 2:
            print("\n[case6] 클래스별 AUC 평균 ± 표준편차:")
            for c in range(curr_num_classes):
                if len(per_class_auc_lists[c]) > 0:
                    auc_mean = float(np.mean(per_class_auc_lists[c]))
                    auc_std = float(np.std(per_class_auc_lists[c]))
                    print(f"  Class {c}: {auc_mean:.4f} ± {auc_std:.4f}")
            if len(per_set_macro_aucs) > 0:
                print("\n[case6] 각 폴드별 Macro AUC:")
                for si, v in enumerate(per_set_macro_aucs, start=1):
                    print(f"  Set{si}: {v:.4f}")
        
        # 각 세트별 성능 출력
        print(f'\n각 세트별 성능:')
        for metrics in all_performance_metrics:
            print(f"  {metrics['Set']}: ACC={metrics['Accuracy']:.4f}, "
                  f"Precision={metrics['Precision']:.4f}, "
                  f"Recall={metrics['Recall']:.4f}, "
                  f"F1={metrics['F1_Score']:.4f}")

        # 모든 세트의 ROC 곡선 그리기
        save_name = f'{save_prefix}_all_sets_roc.png'
        num_classes = config['num_classes']
        plot_all_sets_roc(all_y_true_list, all_y_score_list, "All Sets", num_classes, save_name, figure_title)
        print(f"\n모든 세트의 ROC 곡선 저장 → {save_name}")

    else:
        print(f'{case_name}: 실행할 체크포인트를 찾지 못했습니다.')
    
    print(f"\n{case_name.upper()} Processing completed!")

print("\nAll cases completed!")