# YOLO-Face-mask-detector 
## Supervisor: Liang Liang
## 1. The implementation of two-stages face mask detection model:
        YOLO V1 face detector + AlexNet classifier
## 2. YOLO V5 face mask detection
both of the two projects are done in jupyter notebook version
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
### for yolo v5

![no mask](https://user-images.githubusercontent.com/83719401/143724404-d0372a48-4827-46d3-9104-800cc9e0c073.PNG)
![withmask](https://user-images.githubusercontent.com/83719401/143724406-81975f75-4e0e-4d78-8481-c3859e8706c9.PNG)

### for two-stages yolo v1

![mask](https://user-images.githubusercontent.com/83719401/143785112-f6780f58-43ce-4917-9b0a-b558f314016b.PNG)
![no mask](https://user-images.githubusercontent.com/83719401/143785064-e0742d72-fe5e-4c1e-9953-8bbd00d86989.PNG)
