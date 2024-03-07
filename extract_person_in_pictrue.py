
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np


from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

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

def process_one_image(img_path,out_folder,iou_thresh = 0.3,bbox_score = 0.5):
    os.makedirs(out_folder,exist_ok=True)
    im_name = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > bbox_score)]
    bboxes = bboxes[nms(bboxes, iou_thresh), :4]
    for i,box in enumerate(bboxes):
        box = [int(i) for i in box]
        x1,y1,x2,y2 = box
        sub_image = img[y1:y2,x1:x2,:]
        im_new_name = im_name+str(i)+".png"
        sub_path = os.path.join(out_folder,im_new_name)
        cv2.imwrite(sub_path,sub_image)
    #bboxes numppy array [[x1y1,x2,y2],[...]]



from tqdm import tqdm

def main():


    input = r"D:\HGGT\ImageEnhancement\datasets\DIV2K_train_HR"
    output = r"D:\HGGT\ImageEnhancement\datasets\DIV2K_train_HR_people"
    for image_path in tqdm(glob.glob(input+"/*")):
        process_one_image(image_path,output)
        
main()





