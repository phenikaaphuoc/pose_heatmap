import os
import numpy as np
import cv2

lines = [(0,1),(0,2),(1,2),(1,3),(2,4),(4,6),(3,5),(5,6),(11,12),(5,11),(6,12),(6,8),(8,10),(5,7),(7,9),(12,14),(14,16),(11,13),(13,15)]
colors = [*([[0,255,0]]*7),*([[203, 192, 255]]*4),*([[0, 165, 255]]*4),*([[255, 0, 0]]*4)]

def make_heatmap(result,thresh = 0.01,width = 56,height = 112,thickness = 6):

        mask = result[:,2]>thresh
        result = np.int32(result[mask,:-1])
        max_x,max_y = result[:,0].max()+30,result[:,1].max()+50
        heatmap = np.zeros((max_y,max_x,3),dtype=np.uint8)
        for i,line in enumerate(lines):
            color = colors[i]
            pt1 = result[line[0]]
            pt2 = result[line[1]]
            cv2.line(heatmap, pt1, pt2, color, thickness)
        # return heatmap
        return cv2.resize(heatmap,(width,height))
    
def make_video_heatmap(video_path):
    pose = np.load(video_path)
    heat_map = []
    for result in pose:
        heat_map.append(make_heatmap(result))
    return heat_map

def images_to_video(image_list, output_path, fps=10):
    height, width, _ = image_list[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in image_list: 
        video_writer.write(img)
    video_writer.release()
    

input_data   = [ r"C:\Users\phuoc\Downloads\output_folder_train" , r"C:\Users\phuoc\Downloads\output_folder_test"]
output_data = [r"D:\phuoc_sign\dataset\output_folder_train_image", r"D:\phuoc_sign\dataset\output_folder_test_image"]

import tqdm
for i,input_folder in enumerate(input_data):
    output_folder = output_data[i]
    for action in tqdm.tqdm(os.listdir(input_folder)):
        sub_input_folder = os.path.join(input_folder,action)
        sub_output_folder = os.path.join(output_folder,action)
        
        os.makedirs(sub_output_folder,exist_ok=True)
        for file_name in os.listdir(sub_input_folder)[2:]:
            try:
                input_path = os.path.join(sub_input_folder,file_name)
                heat_maps = make_video_heatmap(input_path)
                output_path = os.path.join(sub_output_folder,file_name.split(".")[0]+".mp4")
                
                images_to_video(heat_maps,output_path)
            except:
                continue
            

