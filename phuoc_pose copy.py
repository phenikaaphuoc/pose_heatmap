import os
from argparse import ArgumentParser
import cv2
import numpy as np
# from mmpose.apis import inference_topdown
# from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
import glob
from tqdm import tqdm
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
lines = [(0,1),(0,2),(1,2),(1,3),(2,4),(4,6),(3,5),(5,6),(11,12),(5,11),(6,12),(6,8),(8,10),(5,7),(7,9),(12,14),(14,16),(11,13),(13,15)]
colors = [*([[0,255,0]]*7),*([[203, 192, 255]]*4),*([[0, 165, 255]]*4),*([[255, 0, 0]]*4)]
mediapipe_lines = [(0,2),(0,5),(5,8),(2,7),(0,9),(0,10),(10,9),(11,12),(11,23),(12,24),(23,24),(12,14),(14,16),(11,13),(13,15),(24,26),(26,28),(23,25),(25,27),(16,22),(16,20),(16,18),(20,18),(15,21),(15,17),(15,19),(19,17)]
mediapipe_colors = [*([[0,255,0]]*7),*([[203, 192, 255]]*4),*([[0, 165, 255]]*4),*([[255, 0, 0]]*4),*([[0, 0, 255]]*8)]
parser = ArgumentParser()
parser.add_argument(
    '--input', type=str, default=r"", help='root_folder')
parser.add_argument(
    '--output', type=str, default=r"output_folder", help='')
parser.add_argument(
    '--device', default='cpu', help='Device used for inference')


args = parser.parse_args()

def extract_frame(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret,frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return frames




def make_heatmap(result,thresh = 0,width = 56,height = 112,thickness = 2):
    mask = result[:,2]>thresh
    result = np.int32(result[mask,:-1])
    max_x,max_y = result[:,0].max()+30,result[:,1].max()+50
    heatmap = np.zeros((max_y,max_x,3),dtype=np.uint8)
    for i,line in enumerate(lines):
        color = colors[i]
        pt1 = result[line[0]]
        pt2 = result[line[1]]
        cv2.line(heatmap, pt1, pt2, color, thickness)
    return heatmap
    # return cv2.resize(heatmap,(width,height))
def make_mediapipe_heatmap(result,thresh = 0,width = 224,height = 448,thickness = 6):
    mask = result[:,2]>thresh
    result = np.int32(result[mask,:-1])
    max_x,max_y = result[:,0].max()+30,result[:,1].max()+50
    heatmap = np.zeros((max_y,max_x,3),dtype=np.uint8)
    for i,line in enumerate(mediapipe_lines):
        color = mediapipe_colors[i]
        pt1 = result[line[0]]
        pt2 = result[line[1]]
        cv2.line(heatmap, pt1, pt2, color, thickness)
        
    # return heatmap
    return cv2.resize(heatmap,(width,height))

def mediapipe_post_process(landmarks,image_shape):
    w,h,_= image_shape
    landmark_array = []
    for idx, landmark in enumerate(landmarks):
        x, y, z, c = int(landmark.x*w), int(landmark.y*h), landmark.z, landmark.visibility
        landmark_array.append([x, y, c])
    return np.array(landmark_array)

def mediapipe_process_one_image(image):
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        # Access pose landmarks
        landmarks = results.pose_landmarks.landmark
        return mediapipe_post_process(landmarks,image.shape)
    else:
        return None

def process_one_video(video_path,take_full = True):
    
    frames  = extract_frame(video_path)
    results = []
    count = 0
    for frame in tqdm(frames):
        count+=1
        if count != 35:
            continue
        result = mediapipe_process_one_image(frame)
        if result is None:
            if take_full:
                result  = np.zeros((17,3))
            else: 
                continue
        results.append(result)
        cv2.imshow("khung xuong",make_heatmap(result))
        cv2.imshow("frame",frame)
        cv2.waitKey(0)
        import pdb;pdb.set_trace()
    return results
            
            
    

def main():

            
            
    input_folder = args.input
    output_folder = args.output

    for action_name in  os.listdir(input_folder):
        sub_int_folder = os.path.join(input_folder,action_name)
        sub_out_folder = os.path.join(output_folder,action_name)
        os.makedirs(sub_out_folder,exist_ok=True)

        for file_name in os.listdir(sub_int_folder):
            file_path = os.path.join(sub_int_folder,file_name)
            # results  = process_one_video(file_path)
            out_name = os.path.basename(file_name)+".npy"
            out_path = os.path.join(sub_out_folder,out_name)
            np.save(out_path,np.array(results))
          
            
        
    

    # input_video = r"D:\phuoc_sign\dataset\train_test_split\test\ban_ghe_sofa\20240107_155650_sub_2.mp4"
    # process_one_video(input_video)
    image = r"C:\Users\phuoc\Downloads\nam2.png"
    image = cv2.imread(image)
    result = mediapipe_process_one_image(image)
    # import pdb;pdb.set_trace()
    heatmap = make_mediapipe_heatmap(result)
    cv2.imshow("heat map",heatmap)
    cv2.waitKey(0)
    
    # if True:
    #     cap = cv2.VideoCapture(0)
    #     while cap.isOpened():
    #         success, frame = cap.read()

    #         if not success:
    #             break

    #         # topdown pose estimation
    #         result = process_one_image(frame)

    #         import pdb;pdb.set_trace()
    #         if cv2.waitKey(5) & 0xFF == 27:
    #                 break

    #     cap.release()




if __name__ == '__main__':
    main()

