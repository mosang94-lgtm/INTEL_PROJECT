"""
New_Sample + 160.차량파손 이미지 데이터를 합쳐서 학습 가능한 환경을 세팅하는 스크립트.
- 디렉토리 구조 생성
- 두 데이터소스의 이미지/라벨을 data/ 폴더에 개별 심볼릭 링크로 통합
- damage_labeling.csv 재생성 (train/val/test 분할)
- Utils.py로 COCO 포맷 JSON 변환
"""
import os
import sys
import json
import glob
import random
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 소스 경로 (이미지, 라벨)
SOURCES = [
    {
        "img":   os.path.join(BASE_DIR, "../New_Sample/원천데이터/TS_damage/damage"),
        "label": os.path.join(BASE_DIR, "../New_Sample/라벨링데이터/TL_damage/damage"),
    },
    {
        "img":   os.path.join(BASE_DIR, "../160. 차량파손 이미지 데이터/01.데이터/1.Training/1.원천데이터/TS_damage/damage"),
        "label": os.path.join(BASE_DIR, "../160. 차량파손 이미지 데이터/01.데이터/1.Training/2.라벨링데이터/TL_damage/damage"),
    },
]

DATA_DIR   = os.path.join(BASE_DIR, "data")
IMG_DIR    = os.path.join(DATA_DIR, "Dataset/1.원천데이터/damage")
LABEL_DIR  = os.path.join(DATA_DIR, "Dataset/2.라벨링데이터/damage")

DIRS_TO_CREATE = [
    os.path.join(DATA_DIR, "datainfo"),
    os.path.join(DATA_DIR, "result_log"),
    os.path.join(DATA_DIR, "weight"),
    os.path.join(DATA_DIR, "Dataset/1.원천데이터/damage_part"),
    os.path.join(DATA_DIR, "Dataset/2.라벨링데이터/damage_part"),
    IMG_DIR,
    LABEL_DIR,
]

def make_dirs():
    for d in DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)
    print("디렉토리 구조 생성 완료")

def link_files():
    """두 소스의 이미지/라벨을 data/ 폴더에 개별 심볼릭 링크로 통합"""
    img_count = 0
    label_count = 0
    skip_count = 0

    for src in SOURCES:
        # 이미지 링크
        for fpath in glob.glob(os.path.join(src["img"], "*.jpg")):
            fname = os.path.basename(fpath)
            link = os.path.join(IMG_DIR, fname)
            if not os.path.exists(link):
                os.symlink(os.path.abspath(fpath), link)
                img_count += 1
            else:
                skip_count += 1

        # 라벨 링크
        for fpath in glob.glob(os.path.join(src["label"], "*.json")):
            fname = os.path.basename(fpath)
            link = os.path.join(LABEL_DIR, fname)
            if not os.path.exists(link):
                os.symlink(os.path.abspath(fpath), link)
                label_count += 1

    print(f"이미지 링크: {img_count}개, 라벨 링크: {label_count}개, 중복 스킵: {skip_count}개")

def parse_damage_type(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    counts = {"Scratched": 0, "Breakage": 0, "Separated": 0, "Crushed": 0}
    for ann in data.get("annotations", []):
        dmg = ann.get("damage", "")
        if dmg in counts:
            counts[dmg] += 1
    return counts

def make_csv():
    label_jsons = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))
    img_basenames = set(
        os.path.splitext(f)[0]
        for f in os.listdir(IMG_DIR) if f.endswith(".jpg")
    )

    rows = []
    for jpath in label_jsons:
        base = os.path.splitext(os.path.basename(jpath))[0]
        if base not in img_basenames:
            continue
        counts = parse_damage_type(jpath)
        total = sum(counts.values())
        if total == 0:
            continue
        rows.append({
            "index": base + ".jpg",
            "Scratched": counts["Scratched"],
            "Breakage": counts["Breakage"],
            "Separated": counts["Separated"],
            "Crushed": counts["Crushed"],
            "total_anns": total,
            "ran_var": random.random(),
            "dataset": ""
        })

    random.seed(42)
    random.shuffle(rows)
    n = len(rows)
    n_val  = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    for i, row in enumerate(rows):
        if i < n_val:
            row["dataset"] = "val"
        elif i < n_val + n_test:
            row["dataset"] = "test"
        else:
            row["dataset"] = "train"

    csv_path = os.path.join(BASE_DIR, "code/damage_labeling.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index","Scratched","Breakage","Separated","Crushed","total_anns","ran_var","dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"damage_labeling.csv 생성 완료: {len(rows)}개 (train: {sum(1 for r in rows if r['dataset']=='train')}, val: {sum(1 for r in rows if r['dataset']=='val')}, test: {sum(1 for r in rows if r['dataset']=='test')})")
    return rows

if __name__ == "__main__":
    print("=== 1. 디렉토리 구조 생성 ===")
    make_dirs()

    print("\n=== 2. 이미지/라벨 심볼릭 링크 통합 ===")
    link_files()

    print("\n=== 3. damage_labeling.csv 재생성 ===")
    rows = make_csv()

    print("\n=== 4. COCO 포맷 JSON 변환 ===")
    os.chdir(BASE_DIR)
    venv_python = os.path.join(BASE_DIR, "../.venv/bin/python")
    if not os.path.exists(venv_python):
        venv_python = "python"
    ret = os.system(f"{venv_python} code/src/Utils.py --make_cocoformat y --task damage")
    if ret != 0:
        print("Utils.py 실행 실패")
        sys.exit(1)

    print("\n=== 완료 ===")
    print("이제 main.py 실행 가능:")
    print("  python main.py --train y --task damage --label all")
