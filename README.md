# The code of Detection and localization of citrus fruit based on improved YOLO v5s and binocular vision in the orchard.

![Image](detection/2Dbbox.png) ![Image](detection/3Dbbox.png)

### What we are doing and going to do

- [x] Upload the citrus dataset. [Google Drive](https://drive.google.com/drive/folders/1VfC0dWsXjhxyKIeAVNtHsxxjXx_2VvV6?usp=sharing)
- [x] Provide Reference image. [Google Drive](https://drive.google.com/drive/folders/1VfC0dWsXjhxyKIeAVNtHsxxjXx_2VvV6?usp=sharing)
- [x] Provide checkpoint model.[Google Drive](https://drive.google.com/drive/folders/1VfC0dWsXjhxyKIeAVNtHsxxjXx_2VvV6?usp=sharing)
- [x] Provide detection code.
- [ ] Provide localization code.

## Preparation

First of all, clone the code

```bash
git clone https://github.com/AshesBen/citrus-detection-localization.git
```

### 1. Prerequisites

* Ubuntu 18.04
* Python 3.8
* Pytorch 1.9.0

* cd ./detection
```bash
pip install -r requirements.txt
```

### 2. Data Preparation

* [Google Drive](https://drive.google.com/drive/folders/1VfC0dWsXjhxyKIeAVNtHsxxjXx_2VvV6?usp=sharing)
* Download and move them into the ./detection/data/

### 3. Training Model

* Firstly, modified the code of ./detection/data/citrus.yaml. Writting the path of train, validation and test dataset.
* Such as: 
* train: ./detection/data/RGB/train/images
* val: ./detection/data/RGB/val/images
* test: ./detection/data/RGB/test/images 

* Secondly, download the pretrain weight of yolov5s in release, and move it into ./detection/

* Finally, open the ./detection/train.py, and select the different loss function in --hyp.
* Such as:
* --hyp data/hyps/oriloss.yaml     (using bce loss function)
* --hyp data/hyps/focalloss.yaml   (using focalloss function)
* --hyp data/hyps/poloss.yaml      (using polarity loss function)
* --hyp data/hyps/ourloss.yaml     (using our loss function)

* Then run it.

### 4. Testing Model

* Open ./detection/detect.py, and modify the path of weight after training in --weights
* such as: --weights ./runs/train/exp/weights/last.pt

* modify the path of test dataset in --source
* such as: --source ./detection/data/RGB/test/images

* Then run it.

### 5. The result

* Open ./detection/recall.py, and modify the path of predicted labels and ground truth labels
* such as: 
* in line 144: detect_path='./runs/detect/exp/labels/'
* in line 145: label_path='./data/RGB/test/labels/'

* Then run it.
