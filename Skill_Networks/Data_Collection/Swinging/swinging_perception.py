import os
import re
import cv2
import numpy as np
import yaml

config_filename = "swinging_config.yaml"
with open(config_filename, "r") as f:
    config = yaml.safe_load(f)

# Path to the folder containing images
folder_path = config['IMAGES']

# Regular expression pattern to match the filename format
pattern = r'rgb_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.png'


files = os.listdir(folder_path)

# Function to extract 'ep' and 'i' values from filename, episode no. and iteration number from images name
def extract_ep_i(filename):
    match = re.match(pattern, filename)
    if match:
        ep, _, _, _,_, i = match.groups()
        ep = int(ep)
        i = int(i)
        return ep, i
    return None, None

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


data = []
vel = []
dist = []
act_vel = []
act_dist = []
act_data = []
import pandas as pd
import numpy as np

pattern = r'rgb_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.png'


# Function to extract 'ep' and 'i' values from filename
def extract_ep(filename):
    match = re.match(pattern, filename)
    if match:
        ep, time,init_angle,cur_angle,omega, i = match.groups()
        ep = int(ep)
        i = int(i)
        return int(ep), float(time), float(init_angle), float(cur_angle)  ,float(omega), int(i)

t = 0
for j in range(0,len(files)):
  all_detections = []
  if len(files[j]) < 20:
    continue

  
  ep = j
  for i in range(0, len(files[j])):

    SOURCE_IMAGE_PATH = folder_path + "/"+ files[j][i]
    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    # Calculate angle with vertical
    sum = 0
    i = 0
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(np.pi - theta)
        sum += (np.pi - theta)
        i += 1
        
        
    avg = sum/i 
    all_detections.append(avg)


  #for calculating difference do central with t-10 and t+10
  for i in range(10,len(all_detections)-10):
    ep, time,init_angle,cur_angle,omega, _ = extract_ep(files[j][i])

    data.append([t,i-10,init_angle,time,all_detections[i],(all_detections[i+10]-all_detections[i-10])/0.1])   #-5 5
    act_data.append([t,i-10,init_angle,time,cur_angle,omega])
    vel.append((all_detections[i+10]-all_detections[i-10])/0.1)
    dist.append(all_detections[i])
    act_vel.append(omega)
    act_dist.append(cur_angle)
    
  t +=1

df = pd.DataFrame(data)

# File name
file_name = config['PERCEPTION_SAVE_DIR']

# Save DataFrame to CSV
df.to_csv(file_name, index=False,header = False)
print(f"Data saved to {file_name}")

if config['WITHOUT PERCEPTION']:
    df = pd.DataFrame(act_data)
    # File name
    file_name =  config['WITHOUT_PERCEPTION_SAVE_DIR']
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False,header = False)
    print(f"Data saved to {file_name}")
