import os
import csv
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from util.datasets import build_transform
##############################################
# 1. Grad-CAM 클래스 정의
##############################################
class GradCAM:
    """
    지정한 모델의 특정(target) convolution layer에 hook을 달아
    Grad-CAM heatmap을 계산하는 클래스입니다.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # forward hook: 해당 레이어의 출력을 저장합니다.
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        # backward hook: 해당 레이어의 gradient를 저장합니다.
        # PyTorch 1.9 이상에서는 register_full_backward_hook 사용 권장
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output은 튜플 형태이므로 첫 번째 요소를 사용합니다.
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        """
        입력 x에 대해 모델의 출력을 계산하고, target class에 대해 backward를 수행하여
        Grad-CAM heatmap을 계산합니다.
        
        Parameters:
            x: 전처리된 입력 텐서 (배치 크기 1)
            class_idx: 관심 있는 클래스 인덱스 (None인 경우 모델이 예측한 최고 점수 클래스를 사용)
        
        Returns:
            cam: (H, W) 크기의 heatmap (0~1로 정규화된 numpy 배열)
            class_idx: 사용한 대상 클래스 인덱스
            confidence: 해당 클래스의 softmax 확률 값
        """
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        # 예측 클래스의 확률 (softmax)
        confidence = torch.softmax(output, dim=1)[0, class_idx].item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # 저장된 gradient와 activation을 이용하여 채널별 global average 계산
        gradients = self.gradients            # shape: [B, C, H, W]
        activations = self.activations        # shape: [B, C, H, W]
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # shape: [B, C, 1, 1]

        # 각 채널을 weights로 가중합산하여 CAM 생성
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # 음수 값 제거

        # 입력 이미지 크기로 보간 (bilinear interpolation)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # shape: [H, W]

        # 0~1 범위로 정규화
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), class_idx, confidence

    def remove_hooks(self):
        """등록한 hook들을 제거합니다."""
        self.forward_handle.remove()
        self.backward_handle.remove()



##############################################
# 2. process_val_images: val 폴더 내 이미지에 대해 Grad-CAM 수행 및 CSV 저장
##############################################
def process_val_images(model, target_layer, device,args,
                       val_folder="val", output_folder="gradcam_results"):
    """
    val 폴더 내의 각 클래스별 폴더 안에 있는 이미지에 대해 Grad-CAM 결과를 계산하여,
    원본 이미지와 Grad-CAM 결과(캠 오버레이)를 좌우 결합한 형태로 output_folder에 저장하고,
    각 이미지의 파일명, 원본 레이블(클래스 폴더 이름), 예측 클래스, 신뢰도 정보를 CSV 파일로 저장합니다.
    
    Parameters:
        model: 학습 완료된 모델
        target_layer: Grad-CAM 계산에 사용할 대상 convolution layer
        device: torch.device (GPU 혹은 CPU)
        val_folder: 검증 이미지가 있는 폴더 (각 클래스별 하위 폴더 존재)
        output_folder: Grad-CAM 결과 이미지를 저장할 폴더 (클래스별 하위 폴더가 생성됨)
    """
    # Grad-CAM 객체 생성
    grad_cam = GradCAM(model, target_layer)

    # 이미지 전처리 transform (모델에 맞게 수정)
    if args is not None:
        preprocess = build_transform(is_train=False, args=args)
    # CSV 저장용 리스트 (헤더: 파일명, 원본 레이블, 예측 클래스, 신뢰도)
    results = []

    # val 폴더 내 각 클래스 폴더 순회
    for class_name in os.listdir(val_folder):
        class_dir = os.path.join(val_folder, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 출력 경로에 클래스 폴더 생성
        output_class_dir = os.path.join(output_folder, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # 클래스 폴더 내의 이미지 파일 순회
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(class_dir, img_file)
            print("Processing:", img_path)

            # 원본 이미지 로드 (PIL)
            raw_image = Image.open(img_path).convert("RGB")

            # 전처리 후 배치 차원 추가 (1, C, H, W)
            input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

            # Grad-CAM 계산: cam, 예측 클래스, 신뢰도 반환
            cam, predicted_class, confidence = grad_cam(input_tensor)
            print(f"Predicted: {predicted_class}, Confidence: {confidence:.4f}")

            # CSV에 [파일명, 원본 레이블, 예측 클래스, 신뢰도] 정보 저장
            results.append([img_file, class_name, predicted_class, confidence])

            # heatmap을 0~255 범위의 정수형으로 변환 후 컬러맵 적용 (OpenCV)
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # 원본 이미지를 모델 입력 크기(224x224)로 resize 후 numpy array 변환
            raw_image_resized = raw_image.resize((224, 224))
            raw_image_np = np.array(raw_image_resized)
            # OpenCV는 BGR 순서를 사용하므로 변환
            raw_image_cv2 = cv2.cvtColor(raw_image_np, cv2.COLOR_RGB2BGR)

            # heatmap의 크기가 원본 이미지와 다르면 리사이즈
            if heatmap.shape[:2] != raw_image_cv2.shape[:2]:
                heatmap = cv2.resize(heatmap, (raw_image_cv2.shape[1], raw_image_cv2.shape[0]))

            # alpha blending: 원본 이미지와 heatmap 오버레이
            alpha = 0.4
            superimposed_img = cv2.addWeighted(raw_image_cv2, 1 - alpha, heatmap, alpha, 0)
            # 다시 BGR에서 RGB로 변환
            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

            # 좌우 결합: 원본 이미지와 Grad-CAM 결과 이미지를 나란히 배치
            combined = np.hstack((raw_image_np, superimposed_img))

            # 저장 경로 (원본 파일명 그대로 저장)
            output_path = os.path.join(output_class_dir, img_file)
            cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print("Saved Grad-CAM result:", output_path)

    # CSV 파일로 결과 저장 (output_folder에 gradcam_results.csv 로 저장)
    csv_path = os.path.join(output_folder, "gradcam_results.csv")
    os.makedirs(output_folder, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "original_label", "predicted_class", "confidence"])
        writer.writerows(results)
    print("CSV file saved at:", csv_path)

    # 사용한 hook 제거
    grad_cam.remove_hooks()