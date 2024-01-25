import os
import sys
import supervision as sv
import numpy as np
import torch
from GroundingDINO.groundingdino.util.inference import Model
#from segment_anything import sam_model_registry, SamPredictor
import cv2
from typing import List


def setup():
    GROUNDING_DINO_CONFIG_PATH =  "<path to parent directory of GroundingDINO>/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "<path to parent directory of GroundingDINO weights>/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    return grounding_dino_model
 
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

grounding_dino_model = setup()

def location(SOURCE_IMAGE_PATH, object_desc):
    CLASSES = [object_desc]
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    # print("DETECTIONS", detections)
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{confidence:0.2f}"
        for _, _, confidence, _, _
        in detections]
    print(labels)
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    cv2.imwrite("annotated_img.png", annotated_frame)

    max_ind = detections.confidence.argmax(axis=0)
    xy = detections.xyxy[max_ind]
    conf = detections.confidence[max_ind]
    centre = [(xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2]

    # if env == 'pendulum':
    #     # for cent in centre:
    #     # centre[0], centre[1] = -((centre[0] - 250) * 1.5) / 190, ((centre[1] - 250) * 1.5) / 190
    #     centre = -(centre[0] - 250) / 200, (centre[1] - 250) / 200
    # elif env == 'sliding':
    #     return centre
    # elif env == 'wedge':
    #     return centre
    # elif env == 'sliding_bridge':
    #     return centre
    # elif env == 'paddles':
    #     return centre

    return centre, conf


print(location("/home/dell/Desktop/Combined_copy/Agent/data/images_eval_sliding/rgb_1.png", "small blue ball"))
# print(location("/home/dell/Desktop/Combined_copy/Agent/annotated_img.png", "blue ball"))
