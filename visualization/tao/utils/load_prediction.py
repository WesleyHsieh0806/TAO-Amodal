import json
import random 
import os 
import numpy as np
import cv2

from collections import defaultdict

class Prediction:
    def __init__(self, prediction, threshold) -> None:
        with open(prediction, 'r') as f:
            print("Reading Output Prediction File {} on TAO-Amodal...".format(prediction))
            self.predictions = json.load(f)
            print("Complete!")
        
        # preprocess the annotation file
        print("preprocess prediction file...")
        self.preprocess(threshold)
        print("Complete!")
    
    

    def preprocess(self, threshold):
        self.imgToAnns = defaultdict(list)
        for ann in self.predictions:
            if ann["score"] < threshold: 
                # ignore boxes whose confidence < threshold
                continue
            self.imgToAnns[ann["image_id"]].append(ann)

