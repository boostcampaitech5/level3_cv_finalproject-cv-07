import os
import sys
import torch
import cv2
import numpy as np
import tqdm
import faiss

sys.path.append('/opt/ml/total')
from association import *


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


def make_predicted_video(detect_model, re_id, video_path, save_path, emb_dim=960):

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Videio open failed!")

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    #player 관리
    shot_try_on = False
    shot_try_done = False
    shot_made_try = False
    shot_made_done = False
    deep_shot_mode = False
    first_frame = True
    exit_frame_num = -1
    shot_cnt=0
    
    pre_rim_data = []

    total_frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    with tqdm.tqdm(total = total_frames_num) as pbar:
        while total_frames_num != frame_num:
            ret, frame = cap.read()
            
            if not ret:
                pbar.update(1)
                frame_num+=1
                continue
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #전체 data
            results = detect_model.predict(img)
            
            #### Re-ID 시작 ####
            if (3 in results[:,-1]):
                id_dict = re_id.re_id_process(img, results, frame_num, first_frame)
                if first_frame:
                    first_frame = False
            else:
                continue
            #### Re-ID 종료 ####

            
            #### 슛시도 및 득점 인식 알고리즘 시작 ####
            rim_state, out = rim_check(results)
            if not rim_state:
                if len(pre_rim_data) !=0:
                    out =  torch.cat((out.detach().cpu(), pre_rim_data.clone().view(1,-1)),0)
                else:
                    continue
                    
            ball_state, out = ball_check(out, thr=0.5)
            
            #공이 존재하면     
            if ball_state:
                ball_data, rim_data = ball_rim_bbox(out)
                ball_x_state, ball_y_state = shot_try_check(ball_data, rim_data)

                #### shoot class ID 인식 #####
                if not deep_shot_mode and (5 in results[:,-1]):
                    sh_idx = (results[:,-1] == 5)
                    max_sh_idx = torch.argmax(results[sh_idx][:,4])
                    sh_data =results[sh_idx][max_sh_idx]
                    
                    shot_thr = 0.65
                    if (sh_data[4] >= shot_thr) and (cal_iou(sh_data, ball_data) > 0):

                        iou_lst = []
                        iou_id = []
                        for img_id in id_dict:
                            iou = cal_iou(id_dict[img_id], sh_data)
                            iou_lst.append(iou.item())
                            iou_id.append(img_id)
                        max_ps_idx = np.argmax(iou_lst)
                        
                        re_id.shot_id =iou_id[max_ps_idx]
                        img = draw_bbox_label(img, [sh_data], thr=shot_thr)
                        deep_shot_mode = True
                        exit_frame_num = frame_num + fps
                #### shoot class ID 인식 종료 #####
                
                                         
                #슛 시도 포인트 체크
                if (ball_x_state & ball_y_state):
                    shot_try_on = True
                elif not (ball_x_state | ball_y_state):
                    shot_try_on = False
                    shot_try_done = False
                    shot_made_try = False
                    shot_made_done = False  
                    shot_cnt = 0
                    
                #조건 만족하면 슛시도 인정
                if not shot_try_done and shot_try_on:
                    re_id.player_dict[re_id.shot_id].shot_try_plus()
                    shot_try_done = True    
                    deep_shot_mode = False
                # 3가지 조건 만족하면 득점 인정
                if not shot_made_done and shot_try_done:
                    if shot_made_check1(ball_data, rim_data):
                        shot_made_try = True

                    if shot_made_try and shot_made_check2(ball_data, rim_data):
                        shot_cnt+=1
                    
                    if shot_cnt >=2 and shot_made_check3(ball_data, rim_data):
                        re_id.player_dict[re_id.shot_id].shot_made_plus()
                        shot_made_done=True
                        shot_cnt=0
                        re_id.shot_id = -1
            
            
                        
                        
                #골대 위치 저장
                pre_rim_data = rim_data.clone()
            #### 슛시도 및 득점 인식 알고리즘 종료####
            
            # id 최대 1초 동안만 추척 
            if frame_num == exit_frame_num:
                deep_shot_mode=False
                
            #사람 그리기
            id_img = draw_id(img, id_dict, thr=0.6)
                
            #사람, 슛 제외한 label 그리기
            side_outs = side_results(out)
            draw_img = draw_bbox_label(id_img, side_outs, thr=0.5)            
            cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

                
            #scor board 그리기
            if len(re_id.player_dict) <=5:
                cv2.rectangle(draw_img, (30,30), (380,250),(0,0,0), -1)
            else:
                cv2.rectangle(draw_img, (30,30), (700,300),(0,0,0), -1)
            
            s_w, s_h = (50,65)
            ply_num = 0
            
            cv2.putText(draw_img, 'Score Board', (s_w,s_h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            
            for i in sorted(re_id.player_dict.keys()):
                if ply_num < 5:
                    cv2.putText(draw_img, f'ID-{re_id.player_dict[i]} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (50,105+ply_num*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    ply_num+=1
                else:
                    cv2.putText(draw_img, f'ID-{re_id.player_dict[i]} Shoot Try:{re_id.player_dict[i].stm} | Goal:{re_id.player_dict[i].smm}', (400,105+(ply_num-5)*30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    
                    
            # d_h = draw_img.shape[0]-100 
            # for i in re_id.player_dict.keys():
            #     draw_img = cv2.putText(draw_img, f'ID{i}-Shoot_Try: {re_id.player_dict[i].stm}  Goal: {re_id.player_dict[i].smm}', (30,d_h+(i*70)), cv2.FONT_ITALIC, 1.5, id_color[i], thickness=5, lineType=cv2.LINE_AA)

            # draw_img = cv2.putText(draw_img, f'ID1-Shoot_Try: {re_id.player_dict[1].stm}  Made: {re_id.player_dict[1].smm}', (30,draw_img.shape[0]-30), cv2.FONT_ITALIC, 1.5, (0,0,0), thickness=5, lineType=cv2.LINE_AA)
            # draw_img = cv2.putText(draw_img, f'ID0-Shoot_Try: {re_id.player_dict[0].stm}  Made: {re_id.player_dict[0].smm}', (30,draw_img.shape[0]-100), cv2.FONT_ITALIC, 1.5, (0,0,0), thickness=5, lineType=cv2.LINE_AA)
            # draw_img = cv2.putText(draw_img, f'ID3-Shoot_Try: {re_id.player_dict[3].stm}  Made: {re_id.player_dict[3].smm}', (1100,draw_img.shape[0]-30), cv2.FONT_ITALIC, 1.5, (0,0,0), thickness=5, lineType=cv2.LINE_AA)
            # draw_img = cv2.putText(draw_img, f'ID2-Shoot_Try: {re_id.player_dict[2].stm}  Made: {re_id.player_dict[2].smm}', (1100,draw_img.shape[0]-100), cv2.FONT_ITALIC, 1.5, (0,0,0), thickness=5, lineType=cv2.LINE_AA)
            # draw_img = cv2.putText(draw_img, f'Shot_ID:{re_id.shot_id}_{deep_shot_mode}', (30,100), cv2.FONT_ITALIC, 2, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)
                
            video_out.write(draw_img)
            pbar.update(1)
            frame_num+=1
                
    cap.release()
    video_out.release()




def draw_bbox_label(img, results, thr=0.3):
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
    


def shot_made_check1(ball_data, rim_data):
    #볼의 하단y의 값이 림의 상단 값의 1.05배를 넘었을 때
    if ball_data[3] <= (rim_data[1]+ (rim_data[3]-rim_data[1])*1.2):
        return True
    else:
        return False


def shot_made_check2(ball_data, rim_data):

    #IoU 계산
    iou = cal_iou(ball_data[:4], rim_data[:4]).item()
    if iou >= 0.1:
        #공의 x중심점과 상단 y 값 계산
        x1, x2, bot_y, top_y = ball_data[0].item(), ball_data[2].item(), ball_data[3].item(), ball_data[1].item()
        rx1, ry1, rx2 = rim_data[0].item(), rim_data[1].item(), rim_data[2].item()
        by_center = (bot_y+top_y)/2
        ball_length = bot_y - top_y
        
        if (rx1<= x1 <= rx2) and (rx1<= x2 <= rx2) and ((by_center - ball_length*0.35) > ry1):
            return True

    return False


def shot_made_check3(ball_data, rim_data):

    #IoU 계산
    iou = cal_iou(ball_data[:4], rim_data[:4]).item()
    if iou >= 0.1:
        #공의 x중심점과 상단 y 값 계산
        x1, x2, bot_y, top_y = ball_data[0].item(), ball_data[2].item(), ball_data[3].item(), ball_data[1].item()
        rx1, ry1, rx2 = rim_data[0].item(), rim_data[1].item(), rim_data[2].item()
        by_center = (bot_y+top_y)/2
        ball_length = bot_y - top_y
        
        if (rx1<= x1 <= rx2) and (rx1<= x2 <= rx2) and (top_y >= ry1):
            return True

    return False
    

def shot_try_check(ball_data, rim_data):
    #공의 하단 y 좌표가 림의 상단 y 좌표값을 넘어야 함
    if ball_data[3] <= rim_data[1]:
        ball_yposition = True
    else:
        ball_yposition = False

        
    ball_center = ((ball_data[0] + ball_data[2])/2).item()
    rim_length = (rim_data[2] - rim_data[0]).item()
    left_limit = rim_data[0].item() - rim_length
    right_limit = rim_data[2].item() + rim_length
    
    if left_limit <= ball_data[0].item() <= right_limit:
        ball_xposition = True 
    else:
        ball_xposition = False

    return ball_xposition, ball_yposition


def cal_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def ball_rim_bbox(out):
    ball_idx = (out[:,-1] == 1)
    rim_idx = (out[:,-1] == 4)
    ball_data = out[ball_idx]
    rim_data = out[rim_idx]
    
    return ball_data[0].detach().cpu(), rim_data[0].detach().cpu()


def ball_check(out, thr=0.4):
    ball_state = True
    ball_idx = (out[:,-1] == 1)
    side_idx = (out[:,-1] != 1)
    rim_idx = (out[:,-1] == 4)
    ball_data = out[ball_idx]
    rim_data = out[rim_idx][0]
    
    if len(ball_data) == 0:
        ball_state = False
        out = out

    elif len(ball_data) == 1:
        out = out
        
    else:
        closest_ball_data = ball_rim_closest(ball_data, rim_data)
        if closest_ball_data[4] < thr:
            out = out[side_idx]
            ball_state = False
        else:
            out = torch.cat((out[side_idx], closest_ball_data.view(1,-1)),0)
        
    return ball_state, out


def ball_rim_closest(ball_data, rim_data):
    rx1, ry1, rx2, ry2 = rim_data[0].item(), rim_data[1].item(), rim_data[2].item(), rim_data[3].item()
    c_rx = (rx2 + rx1)/2
    c_ry = (ry2 + ry1)/2
    
    d_lst = []
    for b in ball_data:
        bx1, by1, bx2, by2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
        c_bx = (bx1 + bx2)/2
        c_by = (by1 + by2)/2
        dist = np.sqrt((c_rx-c_bx)**2 + (c_ry-c_by)**2)
        d_lst.append(dist)

    close_idx = np.argmax(d_lst)
    
    return ball_data[close_idx]


def rim_check(results):
    rim_state = True
    rim_idx = (results[:,-1] == 4)
    side_idx = (results[:,-1] != 4)
    rim_data = results[rim_idx]

    if len(rim_data) == 0:
        rim_state = False
        out = results

    elif len(rim_data) ==1:
        out = results
        
    else:
        max_idx = torch.argmax(rim_data[:,4])
        max_rim_data = rim_data[max_idx]
        out = torch.cat((results[side_idx], max_rim_data.view(1,-1)),0)

    
    return rim_state, out


def side_results(results):
    side_idx = (results[:,-1] != 3) & (results[:,-1] != 5)
    side_data = results[side_idx]
    return side_data