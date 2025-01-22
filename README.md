# Age and Gender Classification
This repository is the mirrored version of the app on huggingface space.

## What does this app do?
This app on huggingface can be used to classify a given image to try to identify the gender and age of each faces detected on the image.

## How does this app work?
Here's the flow of the app:
1. User will input an image of a person or a group of people
2. The system will try to resize the image into height of 1080 if the given size is too large.
3. The System will then pass this image into a pre-trained face and landmark detection model using MTCNN, to detect faces available on the image.
4. For each faces detected, the system will try to align the face orientation if the tilt of the face is higher than a threshold ( in this case 11.25 degree ).
5. Aligning process involved detecting the location of both eyes on the face, and then checking the slope from those twow points. For example if the left eye is significantly lower than right eye, that means the face is not properly aligned and need to be rotated. The rotation will be equal to the degree from those two points.
6. After we extracted faces from the image, we then pass those faces to an image transformer to resize our image into (384,384) in size as the requirement for our feature extractor model.
7. The processed face will be passed to 3 of our models, a feature extractor, a gender classifier, and an age estimator.
8. Feature extractor model is an EfficientNetV2S architecture model, fine-tuned and trained on UTKFace Dataset. This model will learn to extract features on the face to be the input of our gender classifier and age model. PyTorch is used to train this model. The output of this model is a learned representation of the image in form of vector of 1280 elements. 
9. After the face's features been extracted, we then pass this vector of 1280 elements to a gender classifier and age estimator.
10. We take the highest probability class and output it as the label of each faces. Each face will have the predicted gender and age group.

## Performance and other notes
Since these models are deployed without GPU, the inference time will be a little bit slow (~20 seconds).

While the gender classification model achieve great result (93% on both training and testing accuracy), estimating one's age based on face is rather a difficult task. My model only achieved around ~35% Accuracy on both training and testing set.

The age will be classified into 8 age groups: (0-4), (5-9), (10-19), (20-29), (30-39), (40-49), (50-64), and (65++)

## Live Demo
visit: https://huggingface.co/spaces/ravindrawiguna/my-age-gender-utkface

## Screenshots
![Initial app looks](https://github.com/RavindraWiguna/gender-age-utk-face/blob/8c0f7dc3f7cd51108f48a2926ceb43820014d0cc/screenshots/ClearSS.png)

![Example with 1 person](https://github.com/RavindraWiguna/gender-age-utk-face/blob/8c0f7dc3f7cd51108f48a2926ceb43820014d0cc/screenshots/LennaSS.png)

![Example with 2 person](https://github.com/RavindraWiguna/gender-age-utk-face/blob/8c0f7dc3f7cd51108f48a2926ceb43820014d0cc/screenshots/SS2Teen.png)

