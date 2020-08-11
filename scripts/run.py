from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from fdet import RetinaFace
import sys


image_h, image_w = 128, 128
detector = RetinaFace(backbone='RESNET50')

model = tf.keras.models.load_model('../data/mask_classification_model.h5')
model.summary()


class_map = {
    0: 'With mask',
    1: 'Without mask'
}

color_map_image = {
    0: [0,1,0],
    1:[1,0,0]
}

def visualize_detections(image, boxes):
  figsize=(7, 7)
  linewidth=1

  image = np.array(image, dtype=np.uint8)

  plt.figure(figsize=figsize)
  plt.axis('off')
  plt.imshow(image)

  ax = plt.gca()

  for box in boxes:
    x, y, w, h = box
  
    face_image = image[y:y+h,x:x+w]
    
    #To handle those cases where the  height and width of the generated cropped face become 0
    if face_image.shape[0] and face_image.shape[1]:

      face_image = tf.image.resize(face_image, [image_w, image_h])
      face_image = face_image/127.5-1

      _cls = model.predict(np.expand_dims(face_image,axis=0))
      _cls = np.argmax(_cls,axis=1)
      
      text = '{}'.format(class_map[_cls[0]])

      patch = plt.Rectangle([x, y], w, h, fill=False, 
                            edgecolor=color_map_image[_cls[0]], linewidth=linewidth)
      ax.add_patch(patch)
      ax.text(x, y, text, bbox={'facecolor':color_map_image[_cls[0]], 'alpha':0.2}, 
          clip_box=ax.clipbox, clip_on=True)

  plt.savefig("../data/output.jpg")
  

def annotate_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    result = detector.detect(image)

    boxes = []
    for i in range(len(result)):
        boxes.append(result[i]['box'])

    boxes = np.array(boxes)
    visualize_detections(image, boxes)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith('.jpg'):
        image_path = sys.argv[1]
    else:
        image_path = "../data/test/img0.jpg"

    annotate_image(image_path)
