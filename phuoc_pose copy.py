import os
from argparse import ArgumentParser
import cv2
import numpy as np
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
import glob
from tqdm import tqdm
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
lines = [(0,1),(0,2),(1,2),(1,3),(2,4),(4,6),(3,5),(5,6),(11,12),(5,11),(6,12),(6,8),(8,10),(5,7),(7,9),(12,14),(14,16),(11,13),(13,15)]
colors = [*([[0,255,0]]*7),*([[203, 192, 255]]*4),*([[0, 165, 255]]*4),*([[255, 0, 0]]*4)]
parser = ArgumentParser()
parser.add_argument('--det_config',default="mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py",help='Config file for detection')
parser.add_argument('--det_checkpoint',default="config/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth", help='Checkpoint file for detection')
parser.add_argument('--pose_config',default="config/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py", help='Config file for pose')
parser.add_argument('--pose_checkpoint',default="config/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth", help='Checkpoint file for pose')
parser.add_argument(
    '--input', type=str, default=r"", help='root_folder')
parser.add_argument(
    '--output', type=str, default=r"output_folder", help='')
parser.add_argument(
    '--device', default='cpu', help='Device used for inference')
parser.add_argument(
    '--kpt-thr',
    type=float,
    default=0.3,
    help='Visualizing keypoint thresholds')
parser.add_argument(
    '--draw-heatmap',
    action='store_true',
    default=True,
    help='Draw heatmap predicted by the model')
parser.add_argument(
    '--show-kpt-idx',
    action='store_true',
    default=False,
    help='Whether to show the index of keypoints')
parser.add_argument(
    '--skeleton-style',
    default='mmpose',
    type=str,
    choices=['mmpose', 'openpose'],
    help='Skeleton style selection')
parser.add_argument(
    '--radius',
    type=int,
    default=3,
    help='Keypoint radius for visualization')
parser.add_argument(
    '--thickness',
    type=int,
    default=1,
    help='Link thickness for visualization')
parser.add_argument(
    '--alpha', type=float, default=0.8, help='The transparency of bboxes')
parser.add_argument(
    '--draw-bbox', action='store_true', help='Draw bboxes of instances')

args = parser.parse_args()

detector = init_detector(
    args.det_config, args.det_checkpoint, device=args.device)
detector.cfg = adapt_mmdet_pipeline(detector.cfg)

pose_estimator = init_pose_estimator(
    args.pose_config,
    args.pose_checkpoint,
    device=args.device,
    cfg_options=dict(
        model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

def extract_frame(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret,frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def process_one_image(img,iou_thresh = 0.3,bbox_score = 0.5):

    #img bgr mode  
    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > bbox_score)]
    bboxes = bboxes[nms(bboxes, iou_thresh), :4]

    #bboxes numppy array [[x1y1,x2,y2],[...]]
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results).get('pred_instances', None)
    if data_samples is not None:
        #select box with max a
        box_index = pred_instance.scores.argmax()
        x1,y1,x2,y2 = bboxes[box_index]
        kp = data_samples.keypoints[box_index] - np.array([x1,y1])#17, 2 #shift it to top left conner
        score = np.expand_dims(data_samples.keypoint_scores[box_index],-1) #17 , 1
        #17 3 x ,y , score
        
        return np.hstack((kp,score))
    return None 

def process_one_video(video_path,take_full = True):
    #take_full take the result event haven't anyone in frame
    frames  = extract_frame(video_path)
    results = []
    for frame in tqdm(frames):
        result = process_one_image(frame)
        if result is None:
            if take_full:
                result  = np.zeros((17,3))
            else: 
                continue
        
        results.append(result)
    return results
        
radius = 10
color = (0, 255, 0)  # Green color in BGR
thickness = -1
def resize_image_fixed_height(original_image, new_height = 112):

    ratio = new_height / float(original_image.shape[1])
    new_width = int(float(original_image.shape[1]) * float(ratio))
    image = cv2.resize(original_image,(new_width,new_height))
    return image 
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
    return cv2.resize(heatmap,(width,height))
def visualize(result,thresh = 0,width = 1000,height = 780):
    image = np.zeros((height,width,3),dtype=np.uint8)
    mask = result[:,2]>thresh
    result = np.int32(result[mask,:-1])
    x = result[:,0]
    y = result[:,1]
    for i in range(x.shape[0]):
        cv2.circle(image, (x[i],y[i]), radius, color, thickness)
    cv2.imshow("pose",image)
    cv2.waitKey(0)
    
                
            
    

def main():

            
            
    input_folder = args.input
    output_folder = args.output

    for action_name in  os.listdir(input_folder):
        sub_int_folder = os.path.join(input_folder,action_name)
        sub_out_folder = os.path.join(output_folder,action_name)
        os.makedirs(sub_out_folder,exist_ok=True)

        for file_name in os.listdir(sub_int_folder):
            file_path = os.path.join(sub_int_folder,file_name)
            results = np.array([10,10])
            # results  = process_one_video(file_path)
            out_name = os.path.basename(file_name)+".npy"
            out_path = os.path.join(sub_out_folder,out_name)
            np.save(out_path,np.array(results))
          
            
        
    

    # input_video = r"D:\phuoc_sign\dataset\train_test_split\test\ban_ghe_sofa\20240107_155650_sub_2.mp4"
    # process_one_video(input_video)
    
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

