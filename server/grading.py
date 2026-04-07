# -*- coding: utf-8 -*-
import cv2
import numpy as np

# ai-army data.js의 등급 기준과 맞춤
DAMAGE_WEIGHTS = {
    'Scratched': 0.3,
    'Separated': 0.7,
    'Crushed':   0.8,
    'Breakage':  1.0,
}

GRADE_THRESHOLDS = [
    (0.0, 0.15, 'A'),
    (0.15, 0.35, 'B'),
    (0.35, 0.65, 'C'),
    (0.65, 1.0,  'D'),
]


def mask_to_bboxes(mask: np.ndarray, label: str, label_id: int, img_w: int, img_h: int) -> list:
    """이진 마스크 → 바운딩 박스 목록 (정규화 좌표)"""
    detections = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # 너무 작은 노이즈 제거
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # 정규화 (0~1)
        x1 = x / img_w
        y1 = y / img_h
        x2 = (x + w) / img_w
        y2 = (y + h) / img_h
        # 마스크 영역 비율로 confidence 근사
        confidence = min(float(area) / (img_w * img_h) * 20, 0.99)
        confidence = max(confidence, 0.50)
        detections.append({
            'class_id': label_id,
            'class_name': label,
            'part_name': '차량 손상',
            'confidence': round(confidence, 3),
            'bbox': [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
        })
    return detections


def calculate_grade(detections: list) -> tuple[str, float]:
    """검출 결과 → 등급(A~D) + severity_score(0~1)"""
    if not detections:
        return 'A', 0.0

    scores = []
    for det in detections:
        weight = DAMAGE_WEIGHTS.get(det['class_name'], 0.5)
        scores.append(det['confidence'] * weight)

    severity = float(np.mean(scores))
    severity = min(severity, 1.0)

    grade = 'D'
    for lo, hi, g in GRADE_THRESHOLDS:
        if lo <= severity < hi:
            grade = g
            break

    return grade, round(severity, 4)


def process_masks(masks: dict) -> tuple[list, str, float]:
    """마스크 dict → detections 목록 + 최종 등급"""
    all_detections = []
    for idx, (label, mask) in enumerate(masks.items()):
        h, w = mask.shape
        bboxes = mask_to_bboxes(mask, label, idx, w, h)
        all_detections.extend(bboxes)

    grade, severity = calculate_grade(all_detections)
    return all_detections, grade, severity
