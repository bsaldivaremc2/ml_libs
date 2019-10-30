!pip install dlib
!pip install mtcnn
!unzip face_landmarks_detection.zip
predictor_path = "shape_predictor_68_face_landmarks.dat"
test_img = 'baby_sleeping_internet_0001.jpg'

import dlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def open_img(ifile):
  pi = Image.open(ifile).convert('RGB')
  npi = np.asarray(pi)
  return npi.copy()

def open_img_face(ifile):
  detector = dlib.get_frontal_face_detector()
  npi = open_img(ifile)
  faces = detector(npi,0)
  return npi, faces


def load_keras_model(json_model,weights):
    model = model_from_json(open(json_model, "r").read())
    model.load_weights(weights)
    return model

def detect_face(iimg,min_face_size=20):
  import base64, sys, json, io
  from mtcnn.mtcnn import MTCNN
  import numpy as np
  from PIL import Image
  
  iimgt = type(iimg)
  if iimgt == np.ndarray:
    np_img = iimg.copy()
  if iimgt==str:
    np_img = open_img(iimg) #np
  #if image_bytes
  #image_bytes = base64.b64decode(img64)
  #np_img = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
  detector = MTCNN(min_face_size=min_face_size)
  faces = []
  if len(np_img.shape)>2:
    np_img = np_img[:,:,:3]
    faces = detector.detect_faces(np_img)
  if len(faces)>0:
      for face in faces:
          box = face['box']
          x,y,w,h = box
          x,y = max(0,x),max(0,y)
          w = min(np_img.shape[1],x+w)-x
          h = min(np_img.shape[0],y+h)-y
          face['box'] = [x,y,w,h][:]
  return faces[:],np_img

def get_land_marks(iimg,predictor_path=None):
  if type(predictor_path)==str:
    predictor = dlib.shape_predictor(predictor_path)
  else:
    print("Please specify the route for shape_predictor_68_face_landmarks.dat") 
  faces, np_img = detect_face(iimg,min_face_size=20)
  if len(faces)<1:
    print("Faces not found")
    return None
  face = faces[0]
  col,row,w,h = face['box']
  box = [col,row,col+w,row+h]
  dlib_face = dlib.rectangle(col,row,col+w,row+h)
  points = list(predictor(np_img,dlib_face).parts())
  points = [ (p.x,p.y) for p in points]
  return {'numpy':np_img.copy(),'landmarks':points[:],'face':face.copy(),'box':box[:]}

lm = get_land_marks(test_img,predictor_path)
print(lm.keys())
plt.imshow(lm['numpy'])
for p in lm['landmarks']:
  plt.scatter(p[0],p[1],marker="x",color="red")
plt.show()
