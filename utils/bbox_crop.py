import os
import cv2
from pycocotools import coco

# Coco 형식의 annotation 파일 경로
annotation_file = '/opt/ml/final_project/_annotations.coco.json'

# 이미지가 저장될 폴더 경로
output_folder = '/opt/ml/final_project/output_folder'

# annotation 파일 로드
coco_dataset = coco.COCO(annotation_file)

# 이미지가 저장될 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# Coco 형식의 annotation에서 이미지 정보 가져오기
image_ids = coco_dataset.getImgIds()
images = coco_dataset.loadImgs(image_ids)

# 모든 이미지에 대해 crop 수행
for image in images:
    image_path = os.path.join('/opt/ml/final_project/image_folder', image['file_name'])  # 이미지 파일 경로
    img = cv2.imread(image_path)  # 이미지 읽기

    # 현재 이미지의 annotation 정보 가져오기
    annotation_ids = coco_dataset.getAnnIds(imgIds=image['id'])
    annotations = coco_dataset.loadAnns(annotation_ids)

    # annotation box에 해당하는 부분만 crop하여 저장
    for annotation in annotations:
        x, y, w, h = annotation['bbox']  # annotation box 좌표 및 크기
        cropped_img = img[int(y):int(y + h), int(x):int(x + w)]  # 이미지 crop

        # crop된 이미지 저장
        output_path = os.path.join(output_folder, f"{annotation['id']}.jpg")  # 저장될 이미지 파일 경로 및 이름
        cv2.imwrite(output_path, cropped_img)
