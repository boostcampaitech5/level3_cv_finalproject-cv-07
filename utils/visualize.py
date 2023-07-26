import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
import faiss

from re_id import *
from stats_tracker.tracker import StatsTracker

classes_lst = {1:'ball', 2:'made', 3:'person', 4: 'rim', 5:'shoot'}
classes_color = {1: (100, 20, 60), 2: (50, 11, 32), 3: (100, 0, 142), 4:(50, 70, 230), 5:(10, 60, 228)}
id_color = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]


def collect_gallery_data(detect_model, re_id, video_path):
    print('collecting gallery data')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

    gallery_img_lst = []
    
    for index, f in enumerate(tqdm(range(total_frames_num))):
        # if np.random.uniform() < 0.1:
        if index%60==0:         ## select one frame per sec   1 * 60 * 6 360
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detect_model.predict(img)
            person_idx_lst, person_img_lst = re_id.person_query_lst(img, results, 0.9)
            gallery_img_lst.append(person_img_lst)

    print('Done')
    return gallery_img_lst


def make_predicted_video(detect_model, re_id, video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video open failed!")
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
    #tracker 선언
    tracker = StatsTracker(w, h, fps=fps)
    
    total_frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    first_frame = True
    re_id.shot_id = -1
    
    gallery_samples = collect_gallery_data(detect_model, re_id,video_path)
    re_id.init_gallery(gallery_samples)
    
    with tqdm(total = total_frames_num) as pbar:
        while total_frames_num != frame_num:
            ret, frame = cap.read()
            
            if not ret:
                pbar.update(1)
                frame_num+=1
                continue
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
            #전체 data
            results = detect_model.predict(img)
            if len(results) == 0:
                continue
            
            #Re-ID-Process
            if (3 in results[:,-1]): # person 수 기준 추가, if 순도 변경
                id_dict = re_id.re_id_process(img, results)
                if first_frame:
                    first_frame = False
            else:
                continue
    
            # 슛 시도 및 득점 tracking                
            outs = tracker.track(img, results, re_id, id_dict)
                
            #사람 그리기
            id_img = draw_id(img, id_dict, thr=0.6)
                
            #사람 제외한 label 그리기
            side_outs = side_results(outs)
            draw_img = draw_bbox_label(id_img, side_outs, thr=0.5)            
    
            #scor board 동적 그리기
            s_w, s_h = (50,65)
            ply_num = 0
    
            #Score board 그리기
            if len(re_id.player_dict.keys()) <=5:
                # cv2.rectangle(draw_img, (30,30), (510,280),(0,0,0), -1)
                cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                for i in sorted(re_id.player_dict.keys()):
                    cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                    cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                    ply_num+=1
                
            else:
                # cv2.rectangle(draw_img, (30,30), (750,270),(0,0,0), -1)
                cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                
                for i in sorted(re_id.player_dict.keys()):
                    if ply_num < 5:
                        cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                        cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        ply_num+=1
                    else:
                        cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (400,105+(ply_num-5)*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                        cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (400,105+(ply_num-5)*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        ply_num+=1
            
            cv2.putText(draw_img, f'Shoot_ID_: {re_id.shot_id}', (10,800), 0, 1, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(draw_img, f'Shoot_Count: {tracker.shot_count}', (10,830), 0, 1, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(draw_img, f'Made_Count: {tracker.made_count}', (10,860), 0, 1, (255,255,255), thickness=2, lineType=cv2.LINE_AA)
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
            video_out.write(draw_img)
            pbar.update(1)
            frame_num+=1
            tracker.frame_num = frame_num
                
    cap.release()
    video_out.release()


def draw_bbox_label(img, results, thr=0.5):
    thick = 3
    txt_color = (255, 255, 255)
    
    for bb in results:
        x1, y1, x2, y2, c, l = bb
        x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)

        if c < thr:
            continue
            
        #bbox 그리기
        img = cv2.rectangle(img, (x1,y1),(x2,y2), classes_color[l], thick)

        #label 그리기
        p1, p2 = (x1, y1), (x2,y2)
        tf = max(thick - 1, 1)
        w, h = cv2.getTextSize(classes_lst[l]+f':{c:.2f}', 0, fontScale=thick / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        img = cv2.rectangle(img, p1, p2, classes_color[l], -1, cv2.LINE_AA) 
        img = cv2.putText(img, classes_lst[l]+f':{c:.2f}', (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, thick / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        

    return img

    
def draw_id(img, id_dict, thr=0.5):
    thick = 3
    txt_color = (255, 255, 255)
    for id in id_dict:
        x1, y1, x2, y2, c, l = id_dict[id]
        x1, y1, x2, y2, c, l = int(x1), int(y1), int(x2), int(y2), c.item(), int(l)

        if c < thr:
            continue
        
        #bbox 그리기
        cv2.rectangle(img, (x1,y1),(x2,y2), id_color[id], thick)

        #label 그리기
        p1, p2 = (x1, y1), (x2,y2)
        tf = max(thick - 1, 1)
        w, h = cv2.getTextSize(f'ID_{str(id)}:{c:.2f}', 0, fontScale=thick / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, id_color[id], -1, cv2.LINE_AA) 
        cv2.putText(img, f'ID_{str(id)}', (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, thick / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        

    return img
    

def draw_scoreboard(draw_img, re_id):
    s_w, s_h = (50,65)
    ply_num = 0
    
    if len(re_id.player_dict.keys()) <=5:
        cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for i in sorted(re_id.player_dict.keys()):
            cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            ply_num+=1
        
    else:
        cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        
        for i in sorted(re_id.player_dict.keys()):
            if ply_num < 5:
                cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                ply_num+=1
            else:
                cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (400,105+(ply_num-5)*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(draw_img, f'ID-{i} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (400,105+(ply_num-5)*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                ply_num+=1
    
    return draw_img


def side_results(results):
    side_idx = (results[:,-1] != 3)
    side_data = results[side_idx]
    return side_data