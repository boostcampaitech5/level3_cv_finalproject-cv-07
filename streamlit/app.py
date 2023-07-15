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
import open_clip

from ptflops import get_model_complexity_info
from super_gradients.training import models
from super_gradients.common.object_names import Models
from detection_model import Yolo_Nas_L
from tqdm import tqdm
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from re_id.reid import ReId
from re_id.model import TimmModel
from utils.visualize3 import *


YOLO_MODEL_PATH = '/opt/ml/input/code/streamlit/'
REID_MODEL_PATH = '/opt/ml/input/code/streamlit/'

@st.cache_resource
def load_model():
    detect_checkpoint = YOLO_MODEL_PATH + 'ckpt_best.pth'
    net = Yolo_Nas_L(num_classes=6, checkpoint_path=detect_checkpoint).cuda()
    
    re_id_checkpoint = REID_MODEL_PATH + 'convnext_h.pt'
    re_id_model = TimmModel('convnextv2_huge.fcmae_ft_in1k',pretrained=True).cuda()
    re_id_model.load_state_dict(torch.load(os.path.join(re_id_checkpoint)))
    reid_net = ReId(re_id_model, checkpoint=re_id_checkpoint, person_thr=0.6, cosine_thr=0.5)
    
    return net, reid_net

@st.cache_data
def person_query_lst(_frame, _results, thr = 0.7):
    img = _frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    person_img_lst = []
    person_idx_lst = []
    tf = A.Compose([A.Resize(224,224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
    for idx, result in enumerate(_results):
        x1, y1, x2, y2 = result[0:4]
        c = result[4]
        l = result[5]
    
        if (l == 3) and (c.float() >= thr):
            x1, y1, x2, y2, l = int(x1), int(y1), int(x2), int(y2), int(l)
            person_img = img[y1:y2, x1:x2, :]
            if (person_img.shape[0] ==0) or (person_img.shape[1] ==0):
                continue
            person_idx_lst.append(idx)
            transformed = tf(image=person_img)
            tf_person_img = transformed['image']
            tf_person_img = tf_person_img.astype(np.float32)
            tf_person_img = torch.from_numpy(tf_person_img)
            tf_person_img = torch.permute(tf_person_img, (2,0,1))
            person_img_lst.append(tf_person_img)
            
    return person_idx_lst, person_img_lst


def naming_players(_setting_person_id, _setting_person_img, team_list, show_image=True):
    id_max = len(_setting_person_id)
    id_start = 0

    for i in range(0, len(_setting_person_id), 5):
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col1:
            if(show_image):
                st.image(np.array(torch.permute(_setting_person_img[i], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {id_start}')
            team_list.append({"id": id_start, "name": '', "team": "Team1"})
            id_start += 1
            if(i == id_max - 1):
                continue
        with col2:
            if(show_image):
                st.image(np.array(torch.permute(_setting_person_img[i + 1], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {id_start}')
            team_list.append({"id": id_start, "name": '', "team": "Team1"})
            id_start += 1
            if(i + 1 == id_max - 1):
                continue
        with col3:
            if(show_image):
                st.image(np.array(torch.permute(_setting_person_img[i + 2], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {id_start}')
            team_list.append({"id": id_start, "name": '', "team": "Team1"})
            id_start += 1
            if(i + 2 == id_max - 1):
                continue
        with col4:
            if(show_image):
                st.image(np.array(torch.permute(_setting_person_img[i + 3], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {id_start}')
            team_list.append({"id": id_start, "name": '', "team": "Team1"})
            id_start += 1
            if(i + 3 == id_max - 1):
                continue
        with col5:
            if(show_image):
                st.image(np.array(torch.permute(_setting_person_img[i + 4], (1, 2, 0)), dtype=np.float32), clamp=True, caption=f'ID = {id_start}')
            team_list.append({"id": id_start, "name": '', "team": "Team1"})
            id_start += 1
            if(i + 4 == id_max - 1):
                continue
    
    return team_list

def save_edits():
    # st.session_state.team_list = naming_players(st.session_state.setting_persion_id, st.session_state.setting_person_img, st.session_state.team_list, show_image=False)
    st.session_state.data_editor_copy = st.session_state.data_editor.copy()
    for i in range(len(st.session_state.data_editor_copy['id'])):
        st.session_state.team_list[i] = ({"id": st.session_state.data_editor_copy['id'][i], "name": st.session_state.data_editor_copy['name'][i], "team": st.session_state.data_editor_copy['team'][i]})

    st.session_state.team_list_edit = st.session_state.team_list.copy()

# @st.cache_resource(show_spinner=False)
# def detect_video(_net, _tfile, _cap):
#     with st.chat_message("predict"):
#         st.header("2. Predict Video")
#         with st.spinner('Predicting... This may take few minutes...'):
#             prediction1 = _net.predict(_tfile.name, fuse_model=False)
        

#     with st.chat_message("Analysis"):
#         st.header("3. Analysis")
        
#         fps = _cap.get(cv2.CAP_PROP_FPS)
#         w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         video_out = cv2.VideoWriter(MODEL_PATH + VIDEO_PATH.split('/')[-1].split('.')[0] + '_result_average.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
#         total_frames_num = _cap.get(cv2.CAP_PROP_FRAME_COUNT)
#         frame_num = _cap.get(cv2.CAP_PROP_POS_FRAMES)
        
#         shot_try_on = False
#         shot_try_done = False
#         shot_made_try = False
#         shot_made_done = False
#         cnt=0
#         id_max = 4
#         delay = 0
#         player_id = 0
#         shoot_board = [0 for i in range(id_max)]
#         score_board = [0 for i in range(id_max)]
#         class_names = prediction1[0].class_names
         
#         pre_rim_data = []
#         percent = 0.0
    
#         # Text config
#         text_rgb = (255, 255, 255)
#         text_height = [40 * (i + 1) for i in range(id_max * 3)]
#         text_thickness = 2
#         text_lineType = cv2.LINE_AA
#         text_font = cv2.FONT_ITALIC
        
#         detect_bar = st.progress(percent, text=f'{percent:.1f}% 완료')
        
#         for frame_index, frame_prediction in enumerate(prediction1):
#             labels = frame_prediction.prediction.labels.tolist()
#             # confidence = frame_prediction.prediction.confidence
#             bboxes = frame_prediction.prediction.bboxes_xyxy.tolist()
#             frame = frame_prediction.draw(box_thickness=1, show_confidence=False)
            
#             if(delay >= 30 and float(class_names.index('shoot')) in labels):
#                 player_index = [i for i, ele in enumerate(labels) if ele == class_names.index('person')]
#                 player_iou = [cal_iou(bboxes[labels.index(float(class_names.index('shoot')))], bboxes[idx]) for idx in player_index]
#                 player_id = player_iou.index(max(player_iou))
#                 st.write(f"{player_id} Shoot!")
                
#                 shoot_board[player_id] = shoot_board[player_id] + 1
#                 delay = 0
#             elif(delay >= 30 and float(class_names.index('made')) in labels):
#                 st.write(f"{player_id} Goal!")
#                 score_board[player_id] = score_board[player_id] + 1
#                 delay = 0
        
#             for id in range(id_max):
#                 index = id * 3
#                 frame = cv2.putText(frame, f'Player : {id}', (frame.shape[1] - 300, text_height[index]), text_font, 1, text_rgb, thickness=text_thickness, lineType=text_lineType)
#                 frame = cv2.putText(frame, f'Score : {score_board[id]}', (frame.shape[1] - 300, text_height[index+1]), text_font, 1, text_rgb, thickness=text_thickness, lineType=text_lineType)
#                 frame = cv2.putText(frame, f'Shoot_try : {shoot_board[id]}', (frame.shape[1] - 300, text_height[index+2]), text_font, 1, text_rgb, thickness=text_thickness, lineType=text_lineType)
        
#             delay = delay + 1
#             percent = int(frame_index / int(len(prediction1)) * 100)
#             detect_bar.progress(percent, text=f'{percent}% 완료')
#             video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
#         video_out.release()
#         detect_bar.progress(100, text=f'{percent}% 완료')
#         st.success('Analysis Success!')
#         return predection1, video_out

@st.cache_resource(show_spinner=False)
def make_predicted_video(_detect_model, _re_id_model, _cap, _emb_dim=960):
    with st.chat_message("predict"):
        st.header("2. Predict Video")
        with st.spinner('Predicting... This may take few minutes...'):         
            #player 관리
            shot_try_on = False
            shot_try_done = False
            shot_made_try = False
            shot_made_done = False
            deep_shot_mode = False
            first_frame = True
            exit_frame_num = -1
            shot_cnt=0
            
            cap = _cap
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            re_id = _re_id_model
            
            pre_rim_data = []
        
            total_frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

            percent = 0.0
            detect_bar = st.progress(percent, text=f'{percent:.1f}% 완료')
            
            while total_frames_num != frame_num:
                ret, frame = cap.read()
                
                if not ret:
                    frame_num+=1
                    continue
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                #전체 data
                results = _detect_model.predict(img)
                
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

                percent = int(frame_num / int(total_frames_num) * 100)
                detect_bar.progress(percent, text=f'{percent}% 완료')
                frame_num += 1

            detect_bar.progress(100, text=f'{percent}% 완료')
            st.success("Predict Success")

            return re_id

def main():
    st.set_page_config(layout="wide")
    st.cache_data.clear()
    # st.cache_resource.clear()
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
                    net, reid_net = load_model()
                    tfile = tempfile.NamedTemporaryFile(suffix='.mp4')
                    tfile.write(input_video.getvalue())
        
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.warning("Videio open failed!")
                
                    ret, frame = cap.read()
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = net.predict(img)

                    st.session_state.setting_persion_id = []
                    st.session_state.setting_person_img = []
                    st.session_state.team_list = []
                    
                    st.session_state.setting_persion_id, st.session_state.setting_person_img = person_query_lst(frame, results)
                    st.session_state.team_list = naming_players(st.session_state.setting_persion_id, st.session_state.setting_person_img, st.session_state.team_list)
                    
                    team_df = pd.DataFrame(
                        st.session_state.team_list
                    )
                    st.session_state.data_editor = st.data_editor(team_df, column_config={
                                                    "id": st.column_config.NumberColumn("Player Id"),
                                                    "name": st.column_config.TextColumn("Player Name"),
                                                    "team": st.column_config.SelectboxColumn("Player Team", options=["Team1", "Team2"])
                                                    },
                                                     use_container_width=True, hide_index=True, on_change=save_edits)

                    save_edits()
                    
                    isStart = st.button(label='Start', type='primary', disabled=isUpload, use_container_width=True)
                    with main2_col:
                        if(isStart):
                            with st.chat_message("model"):
                                st.header("1. Load Model")
                                
                                if(net):
                                    st.success('Model Load!')
                            # st.write(tfile.name + '/' + input_video.name)
                            
                            prediction = make_predicted_video(net, reid_net, cap)
                            
                            for team in st.session_state.team_list_edit:
                                team['shoot'] = prediction.player_dict.get(team['id']).stm
                                team['goal'] = prediction.player_dict.get(team['id']).smm
                                if(team['shoot'] == 0):
                                    team['ratio'] = 0
                                else:
                                    team['ratio'] = int(team['goal'] / team['shoot'] * 100)

                            st.session_state.score_df = pd.DataFrame(
                                st.session_state.team_list_edit
                            )
                            st.session_state.data_editor_score = st.dataframe(st.session_state.score_df, column_config={
                                                            "id": st.column_config.NumberColumn("Id"),
                                                            "name": st.column_config.TextColumn("Name"),
                                                            "team": st.column_config.SelectboxColumn("Team", options=["Team1", "Team2"]),
                                                            "shoot": st.column_config.NumberColumn("Shoot"),
                                                            "goal": st.column_config.NumberColumn("Goal"),
                                                            "ratio": st.column_config.NumberColumn("FG(%)"),
                                                            },
                                                             use_container_width=True, hide_index=True)


                            if(st.button(label='Reset', type='primary', use_container_width=True)):
                                st.cache_resource.clear()

            

if __name__ == '__main__':
    main()