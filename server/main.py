# -*- coding: utf-8 -*-
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import load_all_models, predict_masks
from grading import process_masks

# 서버 시작 시 모델 로드
models = {}

@asynccontextmanager
async def lifespan(app):
    global models
    models = load_all_models()
    yield

app = FastAPI(title='CarInspect AI Server', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'status': 'ok', 'models_loaded': list(models.keys())}


@app.post('/inspect')
async def inspect(image: UploadFile = File(...)):
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='이미지 파일만 업로드 가능합니다.')

    image_bytes = await image.read()

    start = time.time()
    masks = predict_masks(models, image_bytes)
    detections, grade, severity = process_masks(masks)
    inference_ms = int((time.time() - start) * 1000)

    return {
        'inspection_id': str(uuid.uuid4()),
        'model_id': 'carinspect-unet-resnet34',
        'model_version': '1.0.0',
        'overall_grade': grade,
        'severity_score': severity,
        'inference_ms': inference_ms,
        'detections': detections,
        'image_url': None,
        'overlay_url': None,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
