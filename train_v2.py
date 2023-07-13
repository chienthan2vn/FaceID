from architecture import * 
import os 
import cv2
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from detect_lib import YOLOv8_face

######pathsandvairables#########
face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "./weights/facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = YOLOv8_face('./weights/yolov8n-face.onnx', conf_thres=0.7, iou_thres=0.8)
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def train():
    number_image = 0
    for face_names in os.listdir(face_data):
        person_dir = os.path.join(face_data, face_names)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir,image_name)

            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            boxes, scores, classids, kpts = face_detector.detect(img_RGB)
            if scores:
                x, y, w, h = np.array([i for i in boxes[0]]).astype(int)
                face = img_RGB[y:y+h, x:x+w]

                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = face_encoder.predict(face_d)[0]
                # print("encode predict: ", encode.shape)
                encodes.append(encode)
                number_image += 1
            else:
                print("image error: " + image_path + ". Delete")
                os.remove(image_path)
                
        print(face_names, number_image)
        number_image = 0

        if encodes:
            encode = np.sum(encodes, axis=0 )
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            # print("encode normal: ", encode)
            encoding_dict[face_names] = encode
        
    path = 'encodings/encodings.pkl'
    with open(path, 'wb') as file:
        pickle.dump(encoding_dict, file)






