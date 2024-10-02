from ultralytics import YOLO
import yaml
model = YOLO("LOGOS/yolov8n.pt")
dict_classes = model.model.names
# print(dict_classes)
data = {'train' :  '/kaggle/working/data/train',
        'val' :  '/kaggle/working/data/val',
        'test' :  '/kaggle/working/data/val',
        'nc': len(classes),
        'names': classes
        }