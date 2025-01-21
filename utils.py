import numpy as np
import torch
from PIL import ImageOps
import pandas as pd
from torch import nn

GLOBAL_FEAT_COL = [f'features_{i}' for i in range(1280)]

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])

	return (startX, startY, endX, endY)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((5, 2), dtype=dtype)
	# loop over the 5 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 5):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def calc_distance(p1, p2):
  dx = p1[0] - p2[0]
  dy = p1[1] - p2[1]
  return np.sqrt(dx*dx + dy*dy)


def calc_degree(p1, p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]

  rad = np.arctan2(dy, dx)
  return np.rad2deg(rad)

def preprocess_image_for_face(img):
  width, height = img.size

  scale_size = min(height, 1080)/height
  new_width, new_height = int(width*scale_size), int(height*scale_size)

  procesed_img = img.resize((new_width, new_height))
  procesed_img = np.array(procesed_img)
  return procesed_img

def cut_face(keypoints, img_color, rect):
  sx,sy,ex,ey = rect
  w = ex-sx
  h = ey-sy
  if(h > w):
    # make it square
    ex = sx+h
  else:
    ey = sy+w

  cntr_left_eye = keypoints['left_eye']
  cntr_right_eye = keypoints['right_eye']

  # check which one is higher
  deg = calc_degree(cntr_left_eye, cntr_right_eye)
  if(np.abs(deg) < 11.25):
    print(deg)
    deg = 0
  croped_face = img_color.crop((sx, sy, ex, ey))
  aligned = croped_face.rotate(deg, expand=True)
  return aligned
  # return croped_face

def extract_faces(img_color, face_detector):
  procesed_img = preprocess_image_for_face(img_color)

  # detect faces
  result = face_detector.detect_faces(procesed_img, box_format="xyxy")

  # cut for each ect
  faces_cut = []
  for data in result:
    rect = data['box']
    kp = data['keypoints']

    cutted_face = cut_face(kp, img_color, rect)
    faces_cut.append(cutted_face)

  return faces_cut

def extract_class_probability_torch(probabilities, index_to_class, is_from_numpy=False):
  if(is_from_numpy):
    probabilities = torch.from_numpy(probabilities)

  index_max = torch.argmax(probabilities, dim=0)
  index_class = index_to_class[index_max]
  index_prob = probabilities[index_max]
  return index_class, index_prob

def turn_pred_to_human_readable(is_baby_prob, is_female_prob, ages_prob):

  gender_class, gender_prob = extract_class_probability_torch(is_female_prob, ['Male', 'Female'])
  baby_class, baby_prob = extract_class_probability_torch(is_baby_prob, ['Age >= 5', 'Age < 5'], True)
  age_class, age_prob = extract_class_probability_torch(ages_prob, ['0-4','5-9','10-19','20-29','30-39','40-49','50-64','65++'])

  return f'Gender: {gender_class} ({gender_prob:.2%}) | Age: {age_class} ({age_prob:.2%})'
  # return {
  #     'Age < 5?': f'{baby_class} with probabilty of: {baby_prob:.2%}',
  #     'Gender': f'{gender_class} with probability of: {gender_prob:.2%}',
  #     'Age': f'{age_class} with probability of: {age_prob:.2%}'
  # }

def pred_model_torch(model: nn.Module,
               input_data: torch.tensor,
               add_batch_dim=True):
  
  # make it into batch
  if(add_batch_dim):
    input_data = input_data.unsqueeze(0)

  # set model to eval mode
  model.eval()

  with torch.inference_mode():
    pred = model(input_data)

  return pred

def pred_face(face_img,
              img_transformer,
              feature_extractor,
              baby_model,
              gender_model,
              male_age_model,
              female_age_model):
  
  # extract img features
  features = pred_model_torch(feature_extractor, img_transformer(face_img), True)

  # check if baby
  is_baby_pred = baby_model.predict_proba(pd.DataFrame(features, columns=GLOBAL_FEAT_COL))[0]

  # also wether baby or not pass to gender
  gender_pred = pred_model_torch(gender_model, features, False)[0]
  gender_prob = torch.softmax(gender_pred, dim=0)
  is_female = torch.argmax(gender_pred, dim=0)

  if(is_baby_pred[1] > is_baby_pred[0]):
    # is a baby
    return turn_pred_to_human_readable(is_baby_pred, gender_prob, [0.9]+[0.1/6]*6)
  

  # ok not a baby
  if(is_female):
    age_pred = pred_model_torch(female_age_model, features, False)[0]
  else:
    age_pred = pred_model_torch(male_age_model, features, False)[0]

  ages_prob = torch.softmax(age_pred, dim=0)
  ages_prob_with_baby = torch.cat((torch.tensor([0]), ages_prob))

  return turn_pred_to_human_readable(is_baby_pred, gender_prob, ages_prob_with_baby) # we skip baby
