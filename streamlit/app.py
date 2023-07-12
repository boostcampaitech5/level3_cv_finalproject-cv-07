import streamlit as st
import io
import torch
import torchvision.models as models
import warnings
import os
import random
import cv2
import faiss
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import timm
import time
import torchvision
import tempfile
import pandas as pd

from ptflops import get_model_complexity_info
from super_gradients.training import models
from super_gradients.common.object_names import Models
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

MODEL_PATH = '/opt/ml/input/code/workspace/yolo_nas_l_basketball_players_aug_demo4/'
VIDEO_PATH = '/opt/ml/input/code/video/2vs2_2.MOV'

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

def detect_video(net, tfile, cap):
    #player 관리
    # deep_shot_mode = False
    # shot_id = -1

    # res = faiss.StandardGpuResources()
    # faiss_index = faiss.GpuIndexFlatIP(res, 960)
    # faiss_index = faiss.IndexIDMap(faiss_index)
    # first_frame = True

    # id_dict = re_id_process(re_id_model, img, results, faiss_index=faiss_index, player_dict=player_dict, frame_num=frame_num, first_frame=first_frame, person_thr=0.5, cosine_thr=0.5)
    with st.chat_message("predict"):
        st.header("2. Predict Video")
        with st.spinner('Predicting... This may take few minutes...'):
            prediction1 = net.predict(tfile.name, fuse_model=False)
        st.success("Predict Success")

    with st.chat_message("Analysis"):
        st.header("3. Analysis")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_out = cv2.VideoWriter(MODEL_PATH + VIDEO_PATH.split('/')[-1].split('.')[0] + '_result_average.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        total_frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        shot_try_on = False
        shot_try_done = False
        shot_made_try = False
        shot_made_done = False
        cnt=0
        id_max = 4
        delay = 0
        player_id = 0
        shoot_board = [0 for i in range(id_max)]
        score_board = [0 for i in range(id_max)]
        class_names = prediction1[0].class_names
         
        pre_rim_data = []
        percent = 0.0
    
        # Text config
        text_rgb = (255, 255, 255)
        text_height = [40 * (i + 1) for i in range(id_max * 3)]
        text_thickness = 2
        text_lineType = cv2.LINE_AA
        text_font = cv2.FONT_ITALIC
        
        detect_bar = st.progress(percent, text=f'{percent:.1f}% 완료')
        
        for frame_index, frame_prediction in enumerate(prediction1):
            labels = frame_prediction.prediction.labels.tolist()
            # confidence = frame_prediction.prediction.confidence
            bboxes = frame_prediction.prediction.bboxes_xyxy.tolist()
            frame = frame_prediction.draw(box_thickness=1, show_confidence=False)
            
            if(delay >= 30 and float(class_names.index('shoot')) in labels):
                player_index = [i for i, ele in enumerate(labels) if ele == class_names.index('person')]
                player_iou = [cal_iou(bboxes[labels.index(float(class_names.index('shoot')))], bboxes[idx]) for idx in player_index]
                player_id = player_iou.index(max(player_iou))
                st.write(f"{player_id} Shoot!")
        
                # person_img_lst = shot_person_query_lst(img, ps_data[max_ps_idx])
        
                # matched_list = re_id_inference(model=re_id_model, detected_query=person_img_lst, gallery_dataset=gallery_dataset, num_class=id_num_classes)
        
                # #top3 hard voting
                # matched_list = hard_voting(matched_list=matched_list)
                # shot_id = matched_list[0]
                
                shoot_board[player_id] = shoot_board[player_id] + 1
                delay = 0
            elif(delay >= 30 and float(class_names.index('made')) in labels):
                st.write(f"{player_id} Goal!")
                score_board[player_id] = score_board[player_id] + 1
                delay = 0
        
            for id in range(id_max):
                index = id * 3
                frame = cv2.putText(frame, f'Player : {id}', (frame.shape[1] - 300, text_height[index]), text_font, 1, text_rgb, thickness=text_thickness, lineType=text_lineType)
                frame = cv2.putText(frame, f'Score : {score_board[id]}', (frame.shape[1] - 300, text_height[index+1]), text_font, 1, text_rgb, thickness=text_thickness, lineType=text_lineType)
                frame = cv2.putText(frame, f'Shoot_try : {shoot_board[id]}', (frame.shape[1] - 300, text_height[index+2]), text_font, 1, text_rgb, thickness=text_thickness, lineType=text_lineType)
        
            delay = delay + 1
            percent = int(frame_index / int(len(prediction1)) * 100)
            detect_bar.progress(percent, text=f'{percent}% 완료')
            video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        video_out.release()
        detect_bar.progress(100, text=f'{percent}% 완료')
        st.success('Analysis Success!')


def person_query_lst(frame, results, thr=0.7):
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = A.Compose([A.Resize(224,224),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    person_img_lst = []
    person_idx_lst = []

    labels = results[0].prediction.labels.tolist()
    confidence = results[0].prediction.confidence.tolist()
    bboxes = results[0].prediction.bboxes_xyxy.tolist()
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        c = confidence[i]
        l = labels[i]
    
        if (l == 3) and (c >= thr):
            x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
            person_img = img[y1:y2, x1:x2, :]
            if (person_img.shape[0] ==0) or (person_img.shape[1] ==0):
                continue
            person_idx_lst.append(i)
            transformed = tf(image=person_img)
            tf_person_img = transformed['image']
            tf_person_img = tf_person_img.astype(np.float32)
            tf_person_img = torch.from_numpy(tf_person_img)
            tf_person_img = torch.permute(tf_person_img, (2,0,1))
            person_img_lst.append(tf_person_img)
            
    return person_idx_lst, person_img_lst

def draw_bbox_label(img, results, thr=0.3):
    thick = 3
    txt_color = (255, 255, 255)
    
    for bb in results:
        x1, y1, x2, y2, c, l = bb
        x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
        
        if c < thr:
            continue
        
        #bbox 그리기
        cv2.rectangle(img, (x1,y1),(x2,y2), classes_color[l], thick)
        
        #label 그리기
        p1, p2 = (x1, y1), (x2,y2)
        tf = max(thick - 1, 1)
        w, h = cv2.getTextSize(classes_lst[l]+f':{c:.2f}', 0, fontScale=thick / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, classes_color[l], -1, cv2.LINE_AA) 
        cv2.putText(img, classes_lst[l]+f':{c:.2f}', (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, thick / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

        
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
    p_idx = (results[:,-1] == 3)
    side_idx = (results[:,-1] != 3)
    side_data = results[side_idx]
    return side_data

    
class Player:
    def __init__(self, id):
        self.id = id
        self.stm=0
        self.smm=0
        
    def shot_try_plus(self):
        self.stm+=1
        
    def shot_made_plus(self):
        self.smm+=1
        
def main():
    st.set_page_config(layout="wide")
    main1_col, main2_col = st.columns([0.6, 0.4])
    
    isPlayerNum = True
    isUpload = True
    
    with main1_col:
        file_upload_container = st.container()
        setting_container = st.container()
        record_container = st.container()

        with file_upload_container:
            st.title('Basketball Scoreboard 자동 분석')
            input_video = st.file_uploader('영상 파일 업로드', type=['mp4', 'MOV', 'gif'])
    
            if(input_video):
                isUpload = False
                st.success('Upload Success!')
            
                with setting_container:
                    net = models.get(Models.YOLO_NAS_L, num_classes=6, checkpoint_path=os.path.join(MODEL_PATH, "ckpt_best.pth")).cuda()
                    tfile = tempfile.NamedTemporaryFile(suffix='.mp4')
                    tfile.write(input_video.getvalue())
        
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.warning("Videio open failed!")
                
                    ret, frame = cap.read()
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = net.predict(img, fuse_model=False)
                                
                    setting_person_id, setting_person_img = person_query_lst(frame, results)
                    
                    team_list = []
                    player_dict = dict()
                    id_max = 4
            
                    for i in range(0, len(setting_person_id), 5):
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        with col1:
                            st.image(np.array(torch.permute(setting_person_img[i], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {setting_person_id[i]}')
                            team_list.append({"id": setting_person_id[i], "name": '', "team": "Team1"})
                            if(i == id_max - 1):
                                continue
                        with col2:
                            st.image(np.array(torch.permute(setting_person_img[i + 1], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {setting_person_id[i + 1]}')
                            team_list.append({"id": setting_person_id[i + 1], "name": '', "team": "Team1"})
                            if(i + 1 == id_max - 1):
                                continue
                        with col3:
                            st.image(np.array(torch.permute(setting_person_img[i + 2], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {setting_person_id[i + 2]}')
                            team_list.append({"id": setting_person_id[i + 2], "name": '', "team": "Team1"})
                            if(i + 2 == id_max - 1):
                                continue
                        with col4:
                            st.image(np.array(torch.permute(setting_person_img[i + 3], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {setting_person_id[i + 3]}')
                            team_list.append({"id": setting_person_id[i + 3], "name": '', "team": "Team1"})
                            if(i + 3 == id_max - 1):
                                continue
                        with col5:
                            st.image(np.array(torch.permute(setting_person_img[i + 4], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {setting_person_id[i + 4]}')
                            team_list.append({"id": setting_person_id[i + 4], "name": '', "team": "Team1"})
                            if(i + 4 == id_max - 1):
                                continue
                                       
                    team_df = pd.DataFrame(
                        team_list
                    )
                    data_editor = st.data_editor(team_df, column_config={
                                                    "id": st.column_config.NumberColumn("Player Id"),
                                                    "name": st.column_config.TextColumn("Player Name"),
                                                    "team": st.column_config.SelectboxColumn("Player Team", options=["Team1", "Team2"])
                                                },
                                                 use_container_width=True, hide_index=True)
            # radio_check = st.radio('경기 형식', ('직접입력', '1vs1', '2vs2', '3vs3', '5vs5'))
            # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            # if(radio_check == '직접입력'):
            #     isPlayerNum = False
            # elif(radio_check == '1vs1'):
            #     playerNum1 = 1
            #     playerNum2 = 1
            # elif(radio_check == '2vs2'):
            #     playerNum1 = 2
            #     playerNum2 = 2
            # elif(radio_check == '3vs3'):
            #     playerNum1 = 3
            #     playerNum2 = 3
            # elif(radio_check == '5vs5'):
            #     playerNum1 = 5
            #     playerNum2 = 5
            
            # numInput1, numInput2, empty = st.columns([0.5, 0.5, 1])
            # if(isPlayerNum == False):
            #     with numInput1:
            #         st.text('1팀')
            #         playerNum1 = st.number_input(label="hi", min_value=1, format='%d', disabled=isPlayerNum, label_visibility="collapsed")
            #     with numInput2:
            #         st.text('2팀')
            #         playerNum2 = st.number_input(label="hi2", min_value=1, format='%d', disabled=isPlayerNum, label_visibility="collapsed")
    
            # team1_roaster, team2_roaster = st.columns([0.5, 0.5])
            # with team1_roaster:
            #     team1_list = []
            #     for i in range(playerNum1):
            #         team1_list.append({"id": i, "name": ''})

            #     team1_df = pd.DataFrame(
            #         team1_list
            #     )
            #     data_editor1 = st.data_editor(team1_df, num_rows="dynamic", key="team1", use_container_width=True)

            # with team2_roaster:
            #     team2_list = []
            #     for i in range(playerNum2):
            #         team2_list.append({"id": i, "name": ''})
                
            #     team2_df = pd.DataFrame(
            #         team2_list
            #     )
            #     data_editor2 = st.data_editor(team2_df, num_rows="dynamic", key="team2", use_container_width=True)
    

                    isStart = st.button(label='Start', type='primary', disabled=isUpload, use_container_width=True)

        
                    with main2_col:
                        if(isStart):
                            with st.chat_message("model"):
                                st.header("1. Load Model")
                                if(net):
                                    st.success('Model Load!')
                            # st.write(tfile.name + '/' + input_video.name)
                            
                            detect_video(net, tfile, cap)
            

if __name__ == '__main__':
    main()