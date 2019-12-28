import cv2


def generate():
    '''
    This function generates the .jpg files of the face using the haar cascade classifier 
    and resizes them to a  200*200 file in a grayscale format.
    '''
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascade/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    # '0' or any other number depending upon the number of cameras your pc is connected with

    count = 0
    delay= 0
    while(True):
        # grab frames from camera
        ret, frame = camera.read()
        delay += 1
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face using haarcascade_classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # draw bounding box around face.
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame , (x,y),(x+w,y+h), (255,0,0),2)

            f = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            if delay%5 ==0:
                # create the directory containing sub-dir for that person's image
                cv2.imwrite('G:/open cv/face_data/ashish/%s.jpg'%str(count),f)
                count += 1
        
        cv2.imshow('Camera',frame)
        if cv2.waitKey(5) & 0xff == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    generate()