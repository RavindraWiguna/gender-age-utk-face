# Age and Gender Classification
This repository is the mirrored version of the app on huggingface space.

## What does this app do?
This app on huggingface can be used to classify a given image to try to identify the gender and age of each faces detected on the image.

## How does this app work?
Here's the flow of the app:
1. User will input an image of a person or a group of people
2. The System will then pass this image into a pre-trained face and landmark detection model using MTCNN, to detect faces available on the image.
3. For each faces detected, the system will try to align the face orientation if the tilt of the face is higher than a threshold ( in this case 11.25 degree ).
4. Aligning process involved detecting the location of both eyes on the face, and then checking the slope from those twow points. For example if the left eye is significantly lower than right eye, that means the face is not properly aligned and need to be rotated. The rotation will be equal to the degree from those two points.
5. After we extracted faces from the image, we then pass those faces to 3 of our models, a feature extractor, a gender classifier, and an age estimator.
6. Feature extractor model is an EfficientNetV2S architecture model, fine-tuned and trained on UTKFace Dataset. This model will learn to extract features on the face to be the input of our gender classifier and age model (This is not pre-trained ). PyTorch is used to train this model.
7. After the face's features been extracted, we then pass this vector of 1280 elements to a gender classifier and age estimator.
8. We take the highest probability class and output it as the label of each faces. Each face will have the predicted gender and age group.
