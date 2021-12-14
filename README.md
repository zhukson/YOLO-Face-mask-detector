# YOLO-Face-mask-detector 
## Supervisor: Liang Liang
## 1. The implementation of two-stages face mask detection model:
        YOLO V1 face detector + AlexNet face mask classifier
        Using two stages model because YOLO v1 performs bad on small objects like face mask. 
        Therefore we used larger object face for detection.
## 2. YOLO V5 face mask detection
both of the two projects were done in jupyter notebook version
## Dataset:
### for two-stages yolo v1:

#### download code for face detection:
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="AigV8d7mGnZ9MMZyXOX0")

project = rf.workspace().project("face-maks-yolo-v1")

dataset = project.version(2).download("voc")

#### download code for face mask classification:
!pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="AigV8d7mGnZ9MMZyXOX0")

project = rf.workspace().project("facemask-vzqch")

dataset = project.version(3).download("folder")

### for yolo v5:
https://drive.google.com/drive/folders/1Ol5BBz20njw1ifR0Po6i6n8pnGDbRC5U?usp=sharing

## pretrained weights(best.pt for YOLO V5, 60 epoechs for YOLO V1):
https://drive.google.com/drive/u/1/folders/1Gm-UEpnZxVOHVtNcAUvwA9jh8-BzUzUw

## Inference result:
## image inference:
### for yolo v5
![image](https://user-images.githubusercontent.com/83719401/145906171-f391a8f9-9a52-4cd4-90b7-9ed4b19ecc6c.png)

### for two-stages yolo v1
![image](https://user-images.githubusercontent.com/83719401/145906164-49d16bbc-7f11-4dbd-8c75-a26ab045ba44.png)

## video inference:
see demo
### for yolo v5
https://user-images.githubusercontent.com/83719401/145906967-483b7d4d-0df3-46c2-89af-15860df7fb16.mp4



### for two-stages yolo v1
https://user-images.githubusercontent.com/83719401/145906886-5e10e9cf-579f-470c-ab9a-dff5a19b76c3.mp4










