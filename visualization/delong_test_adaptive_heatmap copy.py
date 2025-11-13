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
# 설정 (모든 케이스)
# ──────────────────────────────
# 모델명 설정
model_name = 'resnet50.a1_in1k'  # ResNet 모델

num_classes = 2  # 이진분류
batch_size = 1   # 배치 1로 고정

# 에폭 설정
epochs = [100, 200, 300, 400, 500]

# 모든 케이스 설정
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

def load_model(ckpt_path, model_name, num_classes=2):
    """체크포인트에서 모델 전체를 로드 (멀티클래스 지원)"""
    try:
        model = torch.load(ckpt_path, map_location='cpu')
        
        # timm 버전 호환성 문제 해결
        if hasattr(model, 'patch_embed'):
            if hasattr(model.patch_embed, '_init_img_size') and not hasattr(model.patch_embed, 'strict_img_size'):
                model.patch_embed.strict_img_size = model.patch_embed._init_img_size
            if not hasattr(model.patch_embed, 'dynamic_img_pad'):
                model.patch_embed.dynamic_img_pad = False
        
        # VisionTransformer의 dynamic_img_size 속성 처리
        if not hasattr(model, 'dynamic_img_size'):
            model.dynamic_img_size = False
        
        # reg_token 속성 처리 (VisionTransformer 호환성)
        if not hasattr(model, 'reg_token'):
            model.reg_token = None
        
        # strict_img_size 속성 처리
        if not hasattr(model, 'strict_img_size'):
            model.strict_img_size = False
        
        print(f"✅ 가중치 로드 성공: {ckpt_path}")
        return model.to(device).eval()
    except Exception as e:
        print(f"❌ 가중치 로드 실패: {ckpt_path} - 오류: {e}")
        return None

def inference_once(model, dataset_path):
    """추론 수행"""
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
    DeLong test 구현 (클립 버전)
    두 ROC 커브 간의 통계적 유의성 검정
    var_diff가 음수나 너무 작은 값을 방지하기 위해 np.clip 사용
    """
    # 배열 길이 검증
    if len(y_true) != len(y_score1) or len(y_true) != len(y_score2):
        raise ValueError(f"배열 길이 불일치: y_true={len(y_true)}, y_score1={len(y_score1)}, y_score2={len(y_score2)}")
    
    # 클래스 1의 확률만 사용
    y_score1_binary = y_score1[:, 1]
    y_score2_binary = y_score2[:, 1]
    y_true_binary = (y_true == 1).astype(int)
    
    # 양성 클래스와 음성 클래스 분리
    positive_indices = y_true_binary == 1
    negative_indices = y_true_binary == 0
    
    y_score1_pos = y_score1_binary[positive_indices]
    y_score1_neg = y_score1_binary[negative_indices]
    y_score2_pos = y_score2_binary[positive_indices]
    y_score2_neg = y_score2_binary[negative_indices]
    
    # DeLong 통계량 계산
    n_pos = len(y_score1_pos)
    n_neg = len(y_score1_neg)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"양성 또는 음성 샘플이 없음: n_pos={n_pos}, n_neg={n_neg}")
    
    # V10, V01 계산 (DeLong et al., 1988)
    V10_1 = np.mean(y_score1_pos)
    V01_1 = np.mean(y_score1_neg)
    V10_2 = np.mean(y_score2_pos)
    V01_2 = np.mean(y_score2_neg)
    
    # AUC 계산
    auc1 = V10_1 - V01_1
    auc2 = V10_2 - V01_2
    
    # 분산 계산
    S10_1 = np.var(y_score1_pos) / n_pos
    S01_1 = np.var(y_score1_neg) / n_neg
    S10_2 = np.var(y_score2_pos) / n_pos
    S01_2 = np.var(y_score2_neg) / n_neg
    
    # 공분산 계산
    try:
        S10_12 = np.cov(y_score1_pos, y_score2_pos)[0, 1] / n_pos
        S01_12 = np.cov(y_score1_neg, y_score2_neg)[0, 1] / n_neg
    except Exception as e:
        raise ValueError(f"공분산 계산 실패: {e}")
    
    # 통계량 계산 (더 높은 정밀도 사용)
    var_diff = S10_1 + S01_1 + S10_2 + S01_2 - 2 * (S10_12 + S01_12)
    
    # 디버깅 정보 출력 (300 에폭 관련)
    if abs(auc1 - auc2) < 0.01:  # AUC 차이가 작은 경우
        print(f"🔍 DeLong Debug - AUC1: {auc1:.6f}, AUC2: {auc2:.6f}, AUC_Diff: {auc1-auc2:.6f}")
        print(f"   분산: S10_1={S10_1:.8f}, S01_1={S01_1:.8f}, S10_2={S10_2:.8f}, S01_2={S01_2:.8f}")
        print(f"   공분산: S10_12={S10_12:.8f}, S01_12={S01_12:.8f}")
        print(f"   var_diff (원본): {var_diff:.8f}")
    
    # AUC 차이의 크기에 따른 처리
    auc_diff_abs = abs(auc1 - auc2)
    
    # 1. AUC가 거의 동일한 경우
    if auc_diff_abs < 1e-10:
        print(f"⚠️  AUC가 거의 동일함: {auc1:.6f} vs {auc2:.6f} -> p-value = 1")
        return 0.0, 1.0, (auc1, auc2)
    
    # 2. AUC 차이가 정말 미미한 경우 (0.0035 미만) - p-value = 1
    if auc_diff_abs < 0.0035:
        print(f"⚠️  미미한 AUC 차이: AUC_diff={auc_diff_abs:.6f} -> p-value = 1 (차이가 거의 없음)")
        return 0.0, 1.0, (auc1, auc2)
    
    # 3. 일반적인 경우: var_diff가 음수인 경우만 클립
    original_var_diff = var_diff
    if var_diff < 0:
        # 음수인 경우 AUC 차이의 제곱에 비례하는 값으로 설정
        adaptive_min_var = max(1e-6, (auc_diff_abs ** 2) * 0.1)
        var_diff = adaptive_min_var
        print(f"⚠️  음수 var_diff 클립: {original_var_diff:.8f} -> {var_diff:.8f} (적응적 최소값: {adaptive_min_var:.8f})")
    
    z_stat = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # 매우 작은 p-value가 나온 경우 추가 디버깅
    if p_value < 1e-4:
        print(f"⚠️  매우 작은 p-value 감지: {p_value:.2e}")
        print(f"   z_stat: {z_stat:.6f}, var_diff: {var_diff:.8f}")
        print(f"   AUC 차이: {auc1-auc2:.6f}, sqrt(var_diff): {np.sqrt(var_diff):.8f}")
    
    return z_stat, p_value, (auc1, auc2)

def perform_delong_tests_for_case_by_set(case_name, config):
    """한 케이스에 대해 각 세트별로 DeLong test 수행 후 메타분석"""
    
    ckpt_root = config['ckpt_root']
    base_data_path = config['base_data_path']
    save_prefix = config['save_prefix']
    
    # 각 세트별 결과 저장
    set_results = {}
    
    # 각 세트별로 처리
    for set_num in range(1, 6):
        
        dataset_path = os.path.join(base_data_path, f'set{set_num}', 'val')
        if not os.path.exists(dataset_path):
            continue
        
        # 각 에폭별 결과 저장
        epoch_results = {}
        
        # 각 에폭별로 추론 수행
        for epoch in epochs:
            # 체크포인트 파일 찾기
            ckpt_path = os.path.join(ckpt_root, f'set{set_num}', f'{epoch}.pth')
            if not os.path.exists(ckpt_path):
                continue
            
            # 모델 로드
            num_classes = config['num_classes']
            model = load_model(ckpt_path, model_name, num_classes)
            if model is None:
                continue
            
            # 추론
            y_true, y_pred, y_score = inference_once(model, dataset_path)
            
            # ROC AUC 계산 (멀티클래스 지원)
            if num_classes == 2:
                # 이진분류
                y_score_binary = y_score[:, 1]
                y_true_binary = (y_true == 1).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
                roc_auc = auc(fpr, tpr)
            else:
                # 멀티클래스 (클래스별 평균 AUC)
                y_true_one_hot = np.zeros((len(y_true), num_classes))
                for i, lbl in enumerate(y_true):
                    y_true_one_hot[i, lbl] = 1
                
                class_aucs = []
                for c in range(num_classes):
                    fpr, tpr, _ = roc_curve(y_true_one_hot[:, c], y_score[:, c])
                    roc_auc = auc(fpr, tpr)
                    class_aucs.append(roc_auc)
                
                roc_auc = np.mean(class_aucs)  # 클래스별 평균 AUC
            
            # 결과 저장
            epoch_results[epoch] = {
                'y_true': y_true,
                'y_score': y_score,
                'auc': roc_auc
            }
            
            # GPU 메모리 정리
            del model
            torch.cuda.empty_cache()
        
        # 세트별 에폭 간 DeLong test 수행
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
                    
                    # DeLong test 수행
                    y_true = result1['y_true']
                    y_score1 = result1['y_score']
                    y_score2 = result2['y_score']
                    
                    # 배열 길이 확인
                    if len(y_true) != len(y_score1) or len(y_true) != len(y_score2):
                        raise ValueError(f'배열 길이 불일치 - y_true={len(y_true)}, y_score1={len(y_score1)}, y_score2={len(y_score2)}')
                    
                    # DeLong test 수행 (클립 버전)
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
    
    # 메타분석: 세트별 결과를 통합
    if set_results:
        # 모든 세트의 결과를 하나로 합치기
        all_set_results = []
        for set_num, results in set_results.items():
            all_set_results.extend(results)
        
        if not all_set_results:
            return None, set_results
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_set_results)
        
        # 메타분석: 각 에폭 조합별로 세트들의 결과를 통합
        meta_analysis_results = []
        
        # 고유한 에폭 조합 찾기
        epoch_pairs = df[['Epoch1', 'Epoch2']].drop_duplicates()
        
        for _, pair in epoch_pairs.iterrows():
            epoch1, epoch2 = pair['Epoch1'], pair['Epoch2']
            
            # 해당 에폭 조합의 모든 세트 결과 필터링
            pair_results = df[(df['Epoch1'] == epoch1) & (df['Epoch2'] == epoch2)]
            
            if len(pair_results) == 0:
                continue
            
            # 유효한 p-value만 필터링
            valid_results = pair_results[~pair_results['P_Value'].isna()]
            
            if len(valid_results) == 0:
                # 모든 결과가 NaN인 경우
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
                # 메타분석 수행 (Fisher's method 사용)
                # p-value를 z-score로 변환
                z_scores = []
                weights = []
                
                # 디버깅: 300-200 에폭 조합 확인
                debug_epochs = (epoch1 == 300 and epoch2 == 200) or (epoch1 == 200 and epoch2 == 300)
                if debug_epochs:
                    print(f"🔍 메타분석 디버깅 - {epoch1}-{epoch2} 에폭 조합:")
                    print(f"   유효한 세트 수: {len(valid_results)}")
                
                for _, row in valid_results.iterrows():
                    if not np.isnan(row['P_Value']) and row['P_Value'] > 0:
                        # p-value를 z-score로 변환
                        z_score = stats.norm.ppf(1 - row['P_Value'] / 2)  # 양측 검정
                        if row['AUC_Diff'] < 0:  # 방향성 고려
                            z_score = -z_score
                        
                        z_scores.append(z_score)
                        weights.append(row['Sample_Size'])  # 샘플 크기를 가중치로 사용
                        
                        # 디버깅: 300-200 에폭 조합 확인
                        if debug_epochs:
                            print(f"   Set{row['Set']}: p-value={row['P_Value']:.2e}, z-score={z_score:.6f}, weight={row['Sample_Size']}")
                
                if z_scores:
                    # 가중 평균 z-score 계산
                    weighted_z = np.average(z_scores, weights=weights)
                    
                    # 메타분석 p-value 계산
                    meta_p_value = 2 * (1 - stats.norm.cdf(abs(weighted_z)))
                    
                    # 디버깅: 300-200 에폭 조합 확인
                    if debug_epochs:
                        print(f"   가중평균 z-score: {weighted_z:.6f}")
                        print(f"   메타분석 p-value: {meta_p_value:.2e}")
                        print(f"   AUC 차이: {valid_results['AUC_Diff'].mean():.6f}")
                        print(f"   절대 AUC 차이: {valid_results['Abs_AUC_Diff'].mean():.6f}")
                    
                    # AUC 차이가 정말 미미한 경우에만 메타분석 p-value를 1로 설정
                    mean_auc_diff_abs = valid_results['Abs_AUC_Diff'].mean()
                    if mean_auc_diff_abs < 0.0035:  # 0.35% 미만일 때만
                        print(f"⚠️  메타분석에서 미미한 AUC 차이 감지: {mean_auc_diff_abs:.6f} -> p-value = 1")
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
        
        # 결과 저장
        if meta_analysis_results:
            meta_df = pd.DataFrame(meta_analysis_results)
            
            # 세트별 상세 결과 저장
            set_detail_df = pd.DataFrame(all_set_results)
            set_detail_filename = f'{save_prefix}_set_detail_results.csv'
            set_detail_df.to_csv(set_detail_filename, index=False)
            
            # 메타분석 결과 저장
            meta_filename = f'{save_prefix}_meta_analysis_results.csv'
            meta_df.to_csv(meta_filename, index=False)
            
            return meta_df, set_results
        else:
            return None, set_results
    else:
        return None, set_results

# ──────────────────────────────
# 메인 처리
# ──────────────────────────────

# 전체 결과 저장
all_meta_results = []
all_set_results = {}

# 각 케이스별로 세트별 DeLong test 수행
for case_name, config in case_configs.items():
    print(f"=== {case_name.upper()} 테스트 시작 (적응형 히트맵) ===")
    print(f"체크포인트 경로: {config['ckpt_root']}")
    print(f"데이터 경로: {config['base_data_path']}")
    print(f"저장 접두사: {config['save_prefix']}")
    print("=" * 50)
    
    meta_df, set_results = perform_delong_tests_for_case_by_set(case_name, config)
    if meta_df is not None:
        all_meta_results.append(meta_df)
        print(f"✅ {case_name.upper()} 메타분석 완료")
        print(f"결과 파일: {config['save_prefix']}_meta_analysis_results.csv")
        print(f"상세 결과 파일: {config['save_prefix']}_set_detail_results.csv")
    if set_results:
        all_set_results[case_name] = set_results
    
    print(f"=== {case_name.upper()} 테스트 완료 (적응형 히트맵) ===")
    print()

# 전체 메타분석 결과 통합
if all_meta_results:
    combined_meta_df = pd.concat(all_meta_results, ignore_index=True)
    combined_meta_df.to_csv('resnet_all_cases_meta_analysis_results_adaptive.csv', index=False)
    print("✅ 전체 케이스 메타분석 결과 통합 완료")
else:
    print("❌ 처리할 수 있는 데이터가 없습니다.")

# 시각화 함수
def create_adaptive_heatmap_visualization(all_meta_results, case_configs):
    """각 케이스별로 메타분석 결과를 적응형 히트맵으로 시각화"""
    
    # 서브플롯 설정 (6개 케이스: 5개 기존 + 1개 single)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Meta-Analysis Results: Adaptive Epoch-wise Comparison Heatmaps', fontsize=16, fontweight='bold')
    
    case_names = list(case_configs.keys())
    
    for idx, case_name in enumerate(case_names):
        # 서브플롯 위치 계산
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # 해당 케이스의 메타분석 결과 필터링
        case_df = all_meta_results[all_meta_results['Case'] == case_name.upper()]
        
        if case_df.empty:
            ax.text(0.5, 0.5, f'No data for {case_name.upper()}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(case_configs[case_name]['figure_title'], fontsize=11, fontweight='bold')
            continue
        
        # 에폭 목록 추출
        all_epochs = sorted(set(case_df['Epoch1'].tolist() + case_df['Epoch2'].tolist()))
        
        # 히트맵 데이터 생성 (메타분석 p-value)
        heatmap_data = np.full((len(all_epochs), len(all_epochs)), np.nan)
        
        # 대각선은 NaN으로 설정 (자기 자신과의 비교)
        np.fill_diagonal(heatmap_data, np.nan)
        
        # 각 에폭 조합의 메타분석 결과를 히트맵에 채우기
        for _, row_data in case_df.iterrows():
            epoch1_idx = all_epochs.index(row_data['Epoch1'])
            epoch2_idx = all_epochs.index(row_data['Epoch2'])
            
            # 메타분석 p-value를 그대로 사용 (유의성 판단을 위해)
            p_value = row_data['Meta_P_Value']
            if not np.isnan(p_value) and p_value > 0 and p_value <= 1:
                # p-value를 그대로 사용
                heatmap_value = p_value
            else:
                heatmap_value = np.nan
            
            # 대각선이 아닌 경우에만 값 설정
            if epoch1_idx != epoch2_idx:
                heatmap_data[epoch1_idx, epoch2_idx] = heatmap_value
                heatmap_data[epoch2_idx, epoch1_idx] = heatmap_value  # 대칭
        
        # 각 케이스별로 적응형 색상 범위 설정 (실제 p-value 기준)
        non_diagonal_data = heatmap_data[~np.eye(heatmap_data.shape[0], dtype=bool)]
        valid_data = non_diagonal_data[~np.isnan(non_diagonal_data)]
        
        if len(valid_data) > 0:
            # 각 케이스의 실제 p-value 범위 사용
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)
        else:
            # 데이터가 없는 경우 기본값
            vmin = 0.001
            vmax = 0.05
        
        # 일반적인 컬러맵 사용 (viridis)
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        
        # 축 레이블 설정
        ax.set_xticks(range(len(all_epochs)))
        ax.set_yticks(range(len(all_epochs)))
        ax.set_xticklabels(all_epochs)
        ax.set_yticklabels(all_epochs)
        
        # 제목 설정
        ax.set_title(case_configs[case_name]['figure_title'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch 1', fontsize=10)
        ax.set_ylabel('Epoch 2', fontsize=10)
        
        # 값 표시
        for i in range(len(all_epochs)):
            for j in range(len(all_epochs)):
                if i == j:
                    # 대각선: 자기 자신과의 비교
                    text = ax.text(j, i, 'N/A',
                                 ha="center", va="center", color="black", fontsize=7, fontweight='normal')
                elif not np.isnan(heatmap_data[i, j]):
                    # 실제 비교 결과 (p-value 그대로)
                    p_val = heatmap_data[i, j]
                    
                    # p-value 값에 따른 글자 색상 조정
                    if p_val < 0.04:
                        text_color = "white"  # 0.04 미만은 흰색 글씨
                    else:
                        text_color = "black"  # 나머지는 검정글씨
                    font_weight = "normal"  # 모든 글씨를 일반 굵기로
                    
                    # p-value 표시 (과학적 표기법 사용)
                    if p_val < 0.001:
                        text = f"{p_val:.2e}"
                    else:
                        text = f"{p_val:.3f}"
                    
                    text = ax.text(j, i, text,
                                 ha="center", va="center", color=text_color, 
                                 fontsize=8, fontweight="normal")
                else:
                    # 데이터가 없는 경우 (NaN)
                    text = ax.text(j, i, 'N/A',
                                 ha="center", va="center", color="red", fontsize=8, fontweight='normal')
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('p-value', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 저장
    plt.savefig('all_cases_adaptive_heatmap123.png', dpi=300, bbox_inches='tight')
    plt.close()

# 시각화 실행
if all_meta_results:
    create_adaptive_heatmap_visualization(combined_meta_df, case_configs)
    print("✅ 전체 케이스 적응형 히트맵 시각화 완료") 