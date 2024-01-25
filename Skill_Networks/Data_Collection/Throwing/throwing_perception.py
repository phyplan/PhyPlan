import os
import re
import sys
import supervision as sv
from typing import List
import torch
import yaml
import cv2

config_filename = "throwing_config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

#Setting up Segment Anything and Grounding Dino
HOME ="/home/dell/Desktop/isaacgym/python/examples/DataCollection2.0/"
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from groundingdino.util.inference import Model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Path to the folder containing images
folder_path = config['IMAGES']

# Regular expression pattern to match the filename format
pattern = r'rgb_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.png'


files = os.listdir(folder_path)

# Function to extract 'ep' and 'i' values from filename, episode no. and iteration number from images name
def extract_ep_i(filename):
    match = re.match(pattern, filename)
    ep, _, _, _,_,_,_, i = match.groups()
    ep = int(ep)
    i = int(i)
    return ep, i

# Filter files to exclude those that don't match the pattern
valid_files = [filename for filename in files if extract_ep_i(filename) != (None, None)]

# Sort valid files by 'ep' and 'i'
sorted_files = sorted(valid_files, key=lambda filename: extract_ep_i(filename))

# Iterate through sorted files
current_ep = None
ep_old = -1
files = []
for filename in sorted_files:
    ep, i = extract_ep_i(filename)

    if ep_old != ep:
      files.append([])
      ep_old = ep
    files[-1].append(filename)

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]



data = []
act_data = []
vel = []
dist = []
act_vel = []
act_dist = []
import pandas as pd
import numpy as np

pattern = r'rgb_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.png'

# List all files in the folder
# files = os.listdir(folder_path)

# Function to extract 'ep' and 'i' values from filename
def extract_ep(filename):
    match = re.match(pattern, filename)
    if match:
        ep, time, init_vel_y, init_vel_x, vel_y,dist_y, dist_x, i = match.groups()
        ep = int(ep)
        i = int(i)
        return int(ep), float(time), float(init_vel_y), float(init_vel_x)  ,float(vel_y),float(dist_y), float(dist_x),int(i)

t = 0
for j in range(0,len(files)):
  all_detections = []
  if len(files[j]) < 20:
    continue


  ep = j
  for i in range(0, len(files[j])):

    SOURCE_IMAGE_PATH = folder_path + "/"+ files[j][i]
    #threshold values that selected object is green coloured ball
    CLASSES = ['green ball']
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25


    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)


    #detect via grounding dino, draw bounding box
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    #storing the bounding box detected
    all_detections.append(detections.xyxy[0])

  all_detections = np.array(all_detections)

  #storing centre of detected object
  all_centres = []
  for xy in all_detections:
    all_centres.append([(xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2])
  all_centres = np.array(all_centres)

  #storing difference between image position of the object and image width/2
  diff = []
  for i in range(0,len(all_centres)):
      diff.append((all_centres[i][0]-500,all_centres[i][1]-500))
  diff = np.array(diff)


#applying camera matrix conversion for real world object location 
#earlier definedd camera properties
  horizontal_fov = 90
  image_width = 1000
  image_heigth = 1000
  vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
  horizontal_fov *= np.pi / 180
  f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
  f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
  diff_p = -1*diff.copy()

  #camera matrix conversion
  #real world position
  diff_p[:,0] =    diff_p[:,0] * 1/f_x
  diff_p[:,1] =    diff_p[:,1] * 1/f_x


  #for calculating difference do central with t-5 and t+5
  for i in range(5,len(diff_p)-5):
     
    ep, time, init_vel_y, init_vel_x, vel_y,dist_y, dist_x, _ = extract_ep(files[j][i])

    data.append([t,i-10,time,init_vel_y,init_vel_x,(diff_p[i+5][1]-diff_p[i-5][1])/0.01,-1*diff_p[i][1],diff_p[i][0]])
    act_data.append([t,i-10,time, init_vel_y, init_vel_x, vel_y,dist_y, dist_x])
    vel.append((diff_p[i+5][1]-diff_p[i-5][1])/0.01)
    dist.append(diff_p[i][1])
    act_vel.append(vel_y)
    act_dist.append(dist_y)
    
  t +=1


columns = ['Episode Number', 'Data Number in that Episode', 'Actual Mu', 'Initial Velocity', 'Time Elapsed', 'Current Velocity', 'Distance Covered']
df = pd.DataFrame(data)
# File name
file_name = config['PERCEPTION_SAVE_DIR']
# Save DataFrame to CSV
df.to_csv(file_name, index=False,header = False)
print(f"Data saved to {file_name}")


if config['WITHOUT PERCEPTION']:
    df = pd.DataFrame(act_data)
    # File name
    file_name = config['WITHOUT_PERCEPTION_SAVE_DIR']
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False,header = False)
    print(f"Data saved to {file_name}")

 




