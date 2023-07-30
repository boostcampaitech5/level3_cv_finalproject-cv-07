import torch
import faiss
import numpy as np

class StatsTracker():
    def __init__(self,  frame_w, frame_h, fps=30):
        self.shot_try_done = False
        self.shot_made_done = False
        self.deep_shot_mode = False
        self.exit_frame_num = -1
        self.shot_exit = -1
        self.fps = fps
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frame_num = 0
        self.shot_count = 0
        self.made_count = 0
        self.no_ball = 0
        self.pre_ball_data = []


    def track(self, img, results, re_id, id_dict):            

        results = results.numpy()
        
        ball_data = results[results[:,-1]==1]
        made_data = results[results[:,-1]==2]
        person_data = results[results[:,-1]==3]
        rim_data = results[results[:,-1]==4]
        shoot_data = results[results[:,-1]==5]

        ball_num = len(ball_data)
        made_num = len(made_data)
        person_num = len(person_data)
        rim_num = len(rim_data)
        shoot_num = len(shoot_data)

        #림 결정하기
        if rim_num > 0:
            #사람 x 좌표 정렬 
            temp = (person_data[:,0] + person_data[:,2])/2
            temp.sort()

            #x좌표 기준 중위값 사람 선별
            if person_num%2 == 1:
                person_x = temp[int(person_num/2)]
            else:
                person_x = (temp[int(person_num/2)]+temp[int(person_num/2)-1])/2

            #중위값에 위치한 사람과 가까운 림 찾기
            temp = abs(person_x - (rim_data[:,0] + rim_data[:,2]/2))
            rim_data = rim_data[[np.argmin(temp)]]
            results = np.concatenate((rim_data, results[results[:,-1]!=4]))
        
        #볼 결정하기
        if len(rim_data) != 0 and len(ball_data) != 0:
            ball_data = self.ball_rim_closest(ball_data, rim_data[0])
            results = np.concatenate((ball_data, results[results[:,-1]!=1]))
        
        outs = results
        
        ball_state = True
        if len(ball_data) == 0:
            ball_state = False
            
        rim_state = True
        if len(rim_data) == 0:
            rim_state = False
        
        #### 슛시도 및 득점 인식 알고리즘 시작 ####    
                
        #공과 림이 존재하면     
        if ball_state and rim_state:
        
            #### shoot class ID 인식 #####
            if not self.deep_shot_mode and shoot_data.any():
                max_sh_idx = np.argmax(shoot_data[:,4])
                sh_data = shoot_data[max_sh_idx]
                if sh_data[4] > 0.5:
                    re_id.shot_id = self.shoot_id_check_reid(re_id, img, person_data, sh_data)
                    # re_id.shot_id = self.shoot_id_check_iou(id_dict, sh_data[0])
                    self.deep_shot_mode = True
                    
                    self.shot_try_done = False
                    self.shot_made_done = False
                    self.shot_count = 0
                    self.made_count=0
                
                    self.exit_frame_num = self.frame_num + self.fps*1
                    self.shot_count = 0
            #### shoot class ID 인식 종료 #####

            
            #슛 시도 포인트 체크
            ball_x_state, ball_y_state = self.shot_try_check(ball_data, rim_data[0])                
               
            #조건 만족하면 슛시도 인정
            if not self.shot_try_done and re_id.shot_id != -1:

                # shoot_class 2개 이하일 땐 x,y 좌표 모두 고려
                if self.shot_count < 2:
                    if (ball_x_state & ball_y_state):
                        self.shot_count+=1
                        
                # shoot_class 2개 이상일 땐 x좌표만 고려
                else:
                    if ball_x_state:
                        self.shot_count+=1
                
                if self.shot_count >=4: #4 frame 이상 조건 만족할 시
                    re_id.player_dict[re_id.shot_id].shot_try_plus()
                    self.shot_try_done = True    
                    self.shot_count = 0
                    self.shot_exit = self.frame_num + int(self.fps*(2/3))
 
                
        # 특정 thr 넘는 made class 추적 기간 동안 3번 이상 등장하면 득점 인정
        if not self.shot_made_done and self.shot_try_done:
            if self.shot_made_check(made_data, 0.65):
                self.made_count+=1
                if self.made_count >= 3: #3 frame 이상 조건 만족할 시
                    self.shot_made_done=True
                    re_id.player_dict[re_id.shot_id].shot_made_plus()
                    self.made_count=0
            

         # id 최대 1초 동안만 강제 유지 
        if self.frame_num == self.exit_frame_num:
            self.deep_shot_mode=False
           
        # 슛 시도 flag 최대 0.7초 동안만 강제 유지
        # if self.frame_num == self.shot_exit:
        
        return outs
        

    def shot_try_check(self, ball_data, rim_data):
        ball_xposition = False
        ball_yposition = False
        b = ball_data[0]
        
        #공 중점의 y 좌표가 림의 상단 y 좌표값을 넘어야 함
        if (b[1]+b[3])/2 <= rim_data[1]:
            ball_yposition = True
        
        ball_center = ((b[0] + b[2])/2)
        rim_length = rim_data[2] - rim_data[0]
        left_limit = rim_data[0] - rim_length
        right_limit = rim_data[2]+ rim_length
    
        if left_limit <= ball_center <= right_limit:
            ball_xposition = True 

        return ball_xposition, ball_yposition


    def shoot_id_check_iou(self, id_dict, sh_data):
        iou_lst = []
        iou_id = []
        for img_id in id_dict:
            iou = self.cal_iou(id_dict[img_id], sh_data)
            iou_lst.append(iou)
            iou_id.append(img_id)
        max_ps_idx = np.argmax(iou_lst)
        return iou_id[max_ps_idx]


    def shoot_id_check_reid(self, re_id, img, person_data, sh_data):
        #shooting 동작과 가장 가까운 person_data 찾기
        iou_lst = []
        for p in person_data:
            iou = self.cal_iou(p, sh_data)
            iou_lst.append(iou)
        max_ps_idx = np.argmax(iou_lst)

        #reid process
        person_idx_lst, person_img_lst1 = re_id.person_query_lst(img, person_data, thr=0.5)
        detected_query1 = person_img_lst1
        detected_query_stack1 = torch.stack(detected_query1, dim=0).to('cuda')
    
        with torch.no_grad():
            detected_query_vector1 = re_id.model(detected_query_stack1).detach().cpu().numpy()                       
        faiss.normalize_L2(detected_query_vector1)
        
        C, I = re_id.faiss_index.search(detected_query_vector1, 5)
    
        matched_list1 = re_id.hard_voting(I,C)
        # print(matched_list1[max_ps_idx])
        return matched_list1[max_ps_idx]

    
    def shot_made_check(self, made_data, thr):
        flag = False 
        if made_data.any():
            #made classs는 하나라고 가정
            if made_data[0][4] > thr:
                flag = True
        return flag


    def close_pre_rim(self, pre_rim_data, rim_data):
        pr_x, pr_y, r_x, r_y = (pre_rim_data[0]+pre_rim_data[2])/2, (pre_rim_data[1]+pre_rim_data[3])/2, (rim_data[0]+rim_data[2])/2, (rim_data[1]+rim_data[3])/2
        x, y = abs(pr_x-r_x), abs(pr_y-r_y)
        
        return (x**2+y**2)**(1/2)


    def cal_iou(self, box1, box2):
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

    
    def ball_rim_closest(self, ball_data, rim_data):
        
        rx1, ry1, rx2, ry2 = rim_data[0], rim_data[1], rim_data[2], rim_data[3]
        c_rx = (rx2 + rx1)/2
        c_ry = (ry2 + ry1)/2
        
        d_lst = []
        for b in ball_data:
            bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
            c_bx = (bx1 + bx2)/2
            c_by = (by1 + by2)/2
            dist = np.sqrt((c_rx-c_bx)**2 + (c_ry-c_by)**2)
            d_lst.append(dist)
    
        close_idx = np.argmin(d_lst)
        return ball_data[[close_idx]]
        

    # def ball_slope_cal(self, ball_data, pre_ball_data, eps=1e-9):
    #     x1, y1, x2, y2, c, l = ball_data[0]
    #     px1, py1, px2, py2, c, l = pre_ball_data[0]
    #     bc = np.array([(x1+x2)//2, (y1+y2)//2])
    #     pbc = np.array([(px1+px2)//2, (py1+py2)//2])
    #     x, y = bc - pbc
    #     slope = abs(y/(x+eps))
    #     return slope