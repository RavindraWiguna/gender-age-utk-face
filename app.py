### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
from torch import nn
import dlib

from utils import extract_faces, pred_face
from typing import Tuple, Dict

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
import joblib

# just in case ya know
device = "cuda" if torch.cuda.is_available() else "cpu"

# all model locations
feature_extractor_weight_path = './models/feature_extractorGender.pth'
rf_baby_model = './models/BabyRF_GenEffnet.pkl'
gender_clf_weight_path = './models/classifierGender.pth'
male_age_clf_weight_path = './models/MaleAgeFromGenderEffnet_FixBestValAcc.pth'
female_age_clf_weight_path = './models/FemaleAgeFromGenderEffnet_fix7outBestWeight.pth'
landmark_detector_path = './models/shape_predictor_5_face_landmarks.dat'
# creating img transformer for torch models
img_transformer = EfficientNet_V2_S_Weights.DEFAULT.transforms()

# load them models

## 1. feature extractor
feature_extractor = efficientnet_v2_s()
feature_extractor.classifier = nn.Identity()
feature_extractor.features.load_state_dict(torch.load(feature_extractor_weight_path, map_location=torch.device(device)))

## 2. baby classifier
rf_baby = joblib.load(rf_baby_model)

## 3. gender classifier
gender_classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=2)
)
gender_classifier.load_state_dict(torch.load(gender_clf_weight_path, map_location=torch.device(device)))

## 4. male age classifier
male_age_clf = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=7)
)

male_age_clf.load_state_dict(torch.load(male_age_clf_weight_path, map_location=torch.device(device)))

## 5. female age classifier
female_age_clf = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=7)
)

female_age_clf.load_state_dict(torch.load(female_age_clf_weight_path, map_location=torch.device(device)))

## 6. face detector & landmark detector
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(landmark_detector_path)

# our main predict function
def predict(img):
  # extract faces
  faces_detected = extract_faces(img, face_detector, landmark_predictor)

  # pred each faces
  pred_results = []
  for face in faces_detected:
    pred_result = pred_face(face,
                            img_transformer,
                            feature_extractor,
                            rf_baby,
                            gender_classifier,
                            male_age_clf,
                            female_age_clf)
    pred_results.append((face, pred_result))
  
  return pred_results

### Gradio app ###

# Create title, description and article strings
title = "Age Estimation & Gender Classification 👶🧓"
description = "Age estimation and gender classification model using effifient net v2 (S) architecture trained and fine-tuned on UTKFace Dataset, and more..."
article = "Face and its alignment is done via dlib's face and landmark detection. For each faces, we classify the gender, and estimate the age using our trained model."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=gr.Gallery(label="Detected Faces with Attributes"), # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
