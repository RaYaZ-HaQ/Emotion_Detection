import cv2
import keras

# Limits the GPU memory usage
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# The HAAR face detection model
haar_face_classifier = cv2.CascadeClassifier('models/haar_model.xml')

# The emotion detection model
model = keras.models.load_model('models/model.h5')
idx_to_emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Suprise', 6:'Neutral'}

def detect_faces_and_emotion(colored_img, scaleFactor=1.1):
    img_copy = colored_img.copy()
    gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
    faces = haar_face_classifier.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = img_copy[y:y+h, x:x+w]
        
        gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY) / 255 # Scale the result
        resized_image = cv2.resize(gray_face, (48,48))
        y_prob = model.predict(resized_image.reshape(-1,48,48,1))
        y_classes = y_prob.argmax(axis=-1)
        y_label = idx_to_emotion[y_classes[0]]

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_y = y - 10 if y - 10 > 10 else y + 10
        bottomLeftCornerOfText = (x, text_y)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2

        cv2.putText(img_copy, y_label, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
    return img_copy

video_stream = cv2.VideoCapture(0)
if not video_stream.isOpened():
    print("Cannot open camera")
    exit()
else:
    while True:
        _, img = video_stream.read()
        img = detect_faces_and_emotion(img)

        cv2.imshow("Frame, Press q to quit", img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
video_stream.release()
video_stream.destroyAllWindows()
