import numpy as np
from keras.models import load_model
import cv2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
import imutils

# loading the face detection model
prototxtPath = r"/Users/jatin/Documents/College/RT/face-mask-detection/face_detector/deploy.prototxt"
weightsPath = r"/Users/jatin/Documents/College/RT/face-mask-detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
 
# loading the mask detection model
maskNet = load_model('mask_detector.keras')
 
cap = cv2.VideoCapture(0)

# start the video capture
while True:
      ret , frame = cap.read()
      frame = imutils.resize(frame, width=1080)
      height = frame.shape[0]
      width = frame.shape[1]
      average = frame.mean(axis=0).mean(axis=0)
      blob = cv2.dnn.blobFromImage(frame,1.0,(350,350),(77,77,77),True)

      faceNet.setInput(blob)
      detections = faceNet.forward()

      faces=[]
      loc=[]
      predictions=[]

      for i in range(0, detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > 0.7:
              box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
              (startX, startY, endX, endY) = box.astype("int")
              (startX, startY) = (max(0, startX), max(0, startY))
              (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

              face = frame[startY:endY, startX:endX]
              face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
              face = cv2.resize(face, (224, 224))
              face = img_to_array(face)
              face = preprocess_input(face)

              faces.append(face)
              loc.append((startX, startY, endX, endY))

      if len(faces) > 0:
          faces = np.array(faces, dtype="float32")
          predictions = maskNet.predict(faces, batch_size=32)

      for (box, pred) in zip(loc, predictions):
          (startX, startY, endX, endY) = box
          (mask, withoutMask) = pred
          label = "Mask" if mask > withoutMask else "No Mask"
          color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
          label = "{} - {:.1f}%".format(label, max(mask, withoutMask) * 100)
          cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
          cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

      cv2.imshow('frame', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()