import cv2
import pickle
# filename = 'mypic.jpg'
# def detect(filename):
#     face_cascade =cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
#     img = cv2.imread(filename)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in faces:
#         img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.namedWindow('Vikings Detected!!')
#     cv2.imshow('Vikings Detected!!', img)
#     cv2.imwrite('./vikings.jpg', img)
#     cv2.waitKey(0)
# detect(filename)

# Doing it on video

def detect():
    '''
    using the trained LBPH model detect the faces!
    '''
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    camera = cv2.VideoCapture(0)
    labels = {}
    recognizer.read('trainer.yaml')
    with open('labels.pickle','rb') as f:
        og_labels = pickle.load(f)
        # reverse the labelling
        labels = {v:k for k,v in og_labels.items()}

    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            # predict on gray_scale images
            id_, conf = recognizer.predict(roi_gray)
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame, name, (x,y), font, 1,(0,255,0),2,cv2.LINE_AA)
            img = cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
            
        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.03,5,0,(40,40))
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        cv2.imshow('camera',frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()