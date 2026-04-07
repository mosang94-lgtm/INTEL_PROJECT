# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../1.AI모델 소스코드/code'))

import torch
import numpy as np
import cv2
from src.Models import Unet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), '../1.AI모델 소스코드/data/weight')

DAMAGE_WEIGHTS = {
    'Scratched': 'Unet_damage_label0_start:2026-04-07 15:03:09 KST+0900_2_epoch_IoU_2e+01.pt',
    'Separated': 'Unet_damage_label1_start:2026-04-07 15:01:00 KST+0900_2_epoch_IoU_0.9.pt',
    'Crushed':   'Unet_damage_label2_start:2026-04-07 15:02:24 KST+0900_2_epoch_IoU_0e+00.pt',
    'Breakage':  'Unet_damage_label3_start:2026-04-07 15:03:41 KST+0900_2_epoch_IoU_0.7.pt',
}

DAMAGE_LABELS = list(DAMAGE_WEIGHTS.keys())  # ['Scratched', 'Separated', 'Crushed', 'Breakage']


def _load_model(weight_filename: str) -> Unet:
    model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=2)
    weight_path = os.path.join(WEIGHT_DIR, weight_filename)
    state = torch.load(weight_path, map_location=torch.device(DEVICE), weights_only=False)
    try:
        model.model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def load_all_models() -> dict:
    """서버 시작 시 4개 모델을 모두 메모리에 로드"""
    models = {}
    for label, weight_file in DAMAGE_WEIGHTS.items():
        print(f'Loading model: {label}')
        models[label] = _load_model(weight_file)
    print('All models loaded.')
    return models


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """이미지 바이트 → 모델 입력 텐서 (1, 3, 256, 256)"""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))           # HWC → CHW
    tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 256, 256)
    return tensor.to(DEVICE)


def predict_masks(models: dict, image_bytes: bytes) -> dict:
    """4개 모델로 추론 → 각 손상 유형별 이진 마스크 반환"""
    tensor = preprocess(image_bytes)
    masks = {}
    with torch.no_grad():
        for label, model in models.items():
            logits = model(tensor)           # (1, 2, 256, 256)
            pred = torch.argmax(logits, dim=1).squeeze(0)  # (256, 256)
            masks[label] = pred.cpu().numpy().astype(np.uint8)
    return masks
