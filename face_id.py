import cv2 
import numpy as np
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import pickle

recognition_t = 0.2
required_size = (160,160)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, scores, _, _ = detector.detect(img)

    for i in range(len(scores)):
        x, y, w, h = np.array([i for i in boxes[i-1]]).astype(int)
        face = img[y:y+h, x:x+w]

        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist

        if name == 'unknown':
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name + f'_{distance:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 1)
    return img 