import os
import sys
import supervision as sv
import numpy as np
import torch
from GroundingDINO.groundingdino.util.inference import Model
#from segment_anything import sam_model_registry, SamPredictor
import cv2
from typing import List

class Detector:
    def __init__(self):
        # self.setup()
        pass
    
    def setup(self):
        GROUNDING_DINO_CONFIG_PATH =  "/home/dell/Desktop/isaacgym/python/examples/DataCollection2.0/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        GROUNDING_DINO_CHECKPOINT_PATH = "/home/dell/Desktop/isaacgym/python/examples/DataCollection2.0/weights/groundingdino_swint_ogc.pth"
        SAM_CHECKPOINT_PATH =  "/home/dell/Desktop/isaacgym/python/examples/DataCollection2.0/weights/sam_vit_h_4b8939.pth"
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
        SAM_ENCODER_VERSION = "vit_h"
    
    def enhance_class_name(self, class_names: List[str]) -> List[str]:
        return [
            f"all {class_name}s"
            for class_name
            in class_names
        ]


    def location(self, SOURCE_IMAGE_PATH, object_desc):
        CLASSES = [object_desc]
        BOX_TRESHOLD = 0.3
        TEXT_TRESHOLD = 0.25
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
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

        return centre, conf
