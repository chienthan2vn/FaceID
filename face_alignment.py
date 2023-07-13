import math
from PIL import Image
import numpy as np

def alignment(img, args):
    left_eye_x, right_eye_x, left_eye_y, right_eye_y = ([i for i in args])

    a = abs(right_eye_y - left_eye_y)
    
    b = abs(right_eye_x - left_eye_x)
    
    c = math.sqrt(math.pow(abs(right_eye_x - left_eye_x), 2) 
                  + math.pow(abs(right_eye_y - left_eye_y), 2))

    cos_alpha = (b*b + c*c - a*a) / (2*b*c)
    alpha = np.arccos(cos_alpha)
    alpha = (alpha * 180) / math.pi

    if left_eye_y > right_eye_y:
        alpha = - alpha

    align_image = Image.fromarray(img)
    align_image = np.array(align_image.rotate(alpha))

    return align_image
