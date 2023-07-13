import cv2 
import numpy as np
from architecture import *
from face_id import *
from detect_lib import YOLOv8_face
from train_v2 import *

if __name__ == "__main__":
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path = "./weights/facenet_keras_weights.h5"
    face_encoder.load_weights(path)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = YOLOv8_face('./weights/yolov8n-face.onnx', conf_thres=0.7, iou_thres=0.8)
    encoding_dict = load_pickle(encodings_path)
    # train()

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        frame= detect(frame , face_detector , face_encoder , encoding_dict)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    