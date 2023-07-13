import cv2
import random
from tqdm import tqdm

def extract_frames(input_video, output_folder, frame_numbers):
    cap = cv2.VideoCapture(input_video)
    count = 0
    success = True

    pbar = tqdm(total=len(frame_numbers), desc='Extracting Frames', unit='frame')

    while success:
        success, image = cap.read()
        if count in frame_numbers:
            cv2.imwrite(f"{output_folder}/frame_{count}.png", image)
            pbar.update(1)
            if len(frame_numbers) == pbar.n:
                break
        count += 1

    cap.release()
    pbar.close()

def count_frames(input_video):
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

input_video = "./VIDEO.mp4"  # 원하는 동영상 파일 이름으로 변경
output_folder = "./frame"  # 추출된 프레임 저장 폴더 이름
frame_count = count_frames(input_video)

frame_numbers = []
for i in range(100):  # 추출할 프레임 수
    frame_numbers.append(random.randrange(0, frame_count))

extract_frames(input_video, output_folder, frame_numbers)
print('Done!')
