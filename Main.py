import os
import cv2, time, json, face_recognition, ssl, email
import smtplib
import numpy as np
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



port = 465
password = "ProjectUseEmail189"
sender_email = "ProjectEmailJE@gmail.com"
reciever_emails = ["oelkind@yahoo.com","jelkind1011@gmail.com", "danny_elkind@yahoo.com"]
context = ssl.create_default_context()

known_check = 0
check_stranger = 0
y_count = 0
x_count = 0 
list_of_seen = []
face_cascade = cv2.CascadeClassifier("/home/pi/Pictures/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
video1 =  cv2.VideoCapture(-1)
a=1
ec = True
check_encoding = 0
checker34 = 0
face_re = []

'''sudo service motion stop
(sudo python3 /home/pi/VSCode/Python/FaceRecogRasPi/Main.py)
'''

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


first_frame = None
while True:
    a = a + 1


    check, frame = video1.read()

    frame1 = frame.copy()
    frame2 = frame.copy()
    cv2.imshow("frame_empty", frame)
    ROI_Crop = first_frame

    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21,21), 0)

    if first_frame is None:
        first_frame = grey
    
    for x in list_of_seen:
        x["count"] += 1

    for x in list_of_seen:
        
        if x["name"] == "Stranger":
            if x["count"] == 80:
                del list_of_seen[y_count]
        y_count +=1

    for x in list_of_seen:
        if x["count"] == 190:
            del list_of_seen[x_count]
            print('''
    
            !!!!!!!!!!!!timer_face_ended!!!!!!!!!!!!
            !!!!!!!!!!!!timer_face_ended!!!!!!!!!!!!
            !!!!!!!!!!!!timer_face_ended!!!!!!!!!!!!
            !!!!!!!!!!!!timer_face_ended!!!!!!!!!!!!
            !!!!!!!!!!!!timer_face_ended!!!!!!!!!!!!
            !!!!!!!!!!!!timer_face_ended!!!!!!!!!!!!

            
            
            ''')
            time.sleep(2)
        x_count += 1
    x_count = 0
    y_count = 0
    delta_frame = cv2.absdiff(first_frame, grey)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations=0)
    (_,cnts,_) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_fir1 = frame

    for contour in cnts:
        cv2.imshow("encodepic_frame", frame2)

        if cv2.contourArea(contour) < 1000:
            continue

        (x,y,w,h) = cv2.boundingRect(contour)
        gray_img1 = cv2.cvtColor(frame[y:(y+h),x:(x+w)], cv2.COLOR_BGR2GRAY)

        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0), 1)
        
        faces = face_cascade.detectMultiScale(gray_img1, scaleFactor = 1.3, minNeighbors = 5)
        
        if faces != ():
            
            for each_face in faces:
                print(each_face, "!!!!!!!!!!!!", faces)
                time.sleep(1)
                print(type(each_face.tolist()))                                    
                frame_fir = frame_fir1
                if 1 == 1:
                    x1 = each_face.tolist()[0]
                    y1 = each_face.tolist()[1]
                    w1 = each_face.tolist()[2]
                    h1 = each_face.tolist()[3]



                    print('entered')
                    frame_fir_shrink = frame_fir[y:(y+h),x:(x+w)]
                    cv2.rectangle(frame_fir_shrink, (x1,y1), (x1+w1,y1+h1), (0,255,0),10)
                    '''cv2.imshow("MovementBox", cv2.rectangle(frame_fir[y:(y+h),x:(x+w)], (x1,y1), (x1+w1,y1+h1), (0,255,0),1))'''
                    print("Face Found!!!", faces)
                    frame1_ver2 = frame1[y:(y+h),x:(x+w)]
                    cv2.imshow('crop_easy', frame1_ver2[y1:(y1+h1),x1:(x1+w1)])
                    encodings = face_recognition.face_encodings(frame1_ver2[y1:(y1+h1),x1:(x1+w1)], known_face_locations=face_recognition.face_locations(frame1_ver2[y1:(y1+h1),x1:(x1+w1)]))
                    '''print(frame_fir_shrink[y1:(y1+h1),x1:(x1+w1)])'''
                    '''known_face_                                    cv2.imwrite("/home/pi/VSCode/Python/FaceRecogRasPi/face_image.jpg", frame1_ver2[y1:(y1+h1),x1:(x1+w1)])
elocations=face_recognition.face_locations(frame_fir_shrink[y1:(y1+h1),x1:(x1+w1)], number_of_times_to_upsample=1,model='hog')'''
                    print("ENCODINGS!!!",encodings,"LOCATIONS_OF_FACE!!!",face_recognition.face_locations(frame_fir_shrink[y1:(y1+h1),x1:(x1+w1)]))
                    if encodings != []:
                        with open("face_recognition_data.json") as jsonFile:
                            jsonFile.seek(0)
                            jsonObject = json.load(jsonFile)
                            jsonFile.close()
                        for object in jsonObject:
                            print('numpy array', jsonObject[object]['face_encoding'])
                            print([np.asarray(jsonObject[object]['face_encoding'])], "ECDOIGNSFORMCAMCURRENT", encodings)
                            print(face_recognition.compare_faces([[np.asarray(jsonObject[object]['face_encoding'])]],np.asarray(encodings),tolerance=0.6))
                            print(face_recognition.compare_faces([[np.asarray(jsonObject[object]['face_encoding'])]],np.asarray(encodings),tolerance=0.6)[0][0])
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", encodings, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
                            for x in face_recognition.compare_faces([[np.asarray(jsonObject[object]['face_encoding'])]],np.asarray(encodings),tolerance=.14)[0][0]:
                                print(x)
                                if x == False:
                                    ec = False
                            
                            if ec == True:
                                for x in list_of_seen:
                                    if x["name"] == jsonObject[object]['name']:
                                        known_check = 1
                                if known_check == 0:
                                    cv2.imwrite("/home/pi/VSCode/Python/FaceRecogRasPi/face_image.jpg", frame1_ver2[y1:(y1+h1),x1:(x1+w1)])
                                    print("Hey, " + jsonObject[object]['name'] + "!")
                                    body = jsonObject[object]['name'] + "has arrived at your household!"
                                    subject = "Home Monitor"
                                    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                                        server.login(sender_email, password)
                                        for reciever_email in reciever_emails:
                                            message = MIMEMultipart()
                                            message["From"] = sender_email
                                            message["To"] = reciever_email
                                            message["Subject"] = subject
                                            message["Bcc"] = reciever_email  # Recommended for mass emails
                                            message.attach(MIMEText(body, "plain"))
                                            message.attach(MIMEText(body, "plain"))

                                            filename = "face_image.jpg"  # In same directory as script

                                            # Open PDF file in binary mode
                                            with open(filename, "rb") as attachment:
                                                # Add file as application/octet-stream
                                                # Email client can usually download this automatically as attachment
                                                part = MIMEBase("application", "octet-stream")
                                                part.set_payload(attachment.read())

                                            # Encode file in ASCII characters to send by email    
                                            encoders.encode_base64(part)

                                            # Add header as key/value pair to attachment part
                                            part.add_header(
                                                "Content-Disposition",
                                                f"attachment; filename= {filename}",
                                            )

                                            # Add attachment to message and convert message to string
                                            message.attach(part)
                                            text = message.as_string()

                                            server.sendmail(sender_email, reciever_email, text)
                                    time.sleep(2)
                                    
                                    list_of_seen.append(
                                    {
                                        "name":jsonObject[object]['name'],
                                        "count":0
                                    }
                                    )
                                check_encoding = 1
                            ec = True
                        if check_encoding == 0 and encodings != [] and known_check == 0:
                            for x in list_of_seen:
                                if x["name"] == "Stranger":
                                    check_stranger = 1
                            if check_stranger == 0:
                            
                                cv2.imwrite("/home/pi/VSCode/Python/FaceRecogRasPi/face_image.jpg", frame1_ver2[y1:(y1+h1),x1:(x1+w1)])
                                body = "A stranger has arrived at your household!"
                                subject = "Home RasberryPi Monitor"
                                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                                    server.login(sender_email, password)
                                    for reciever_email in reciever_emails:
                                        message = MIMEMultipart()
                                        message["From"] = sender_email
                                        message["To"] = reciever_email
                                        message["Subject"] = subject
                                        message["Bcc"] = reciever_email  # Recommended for mass emails
                                        message.attach(MIMEText(body, "plain"))
                                        message.attach(MIMEText(body, "plain"))

                                        filename = "face_image.jpg"  # In same directory as script

                                            # Open PDF file in binary mode
                                        with open(filename, "rb") as attachment:
                                                # Add file as application/octet-stream
                                                # Email client can usually download this automatically as attachment
                                            part = MIMEBase("application", "octet-stream")
                                            part.set_payload(attachment.read())

                                            # Encode file in ASCII characters to send by email    
                                        encoders.encode_base64(part)

                                            # Add header as key/value pair to attachment part
                                        part.add_header(
                                            "Content-Disposition",
                                            f"attachment; filename= {filename}",
                                            )

                                            # Add attachment to message and convert message to string
                                        message.attach(part)
                                        text = message.as_string()

                                        server.sendmail(sender_email, reciever_email, text)
                                
                            check_stranger = 0
                        if os.path.isfile("face_image") == True:
                            os.remove("face_image.jpg")
                        check_encoding = 0
                        known_check = 0

                        


        else:
            print("NoFace                                                            !")
            print(faces)
        
   

    cv2.imshow("frame", frame)
    '''cv2.imshow("Capturing", grey)'''
    '''cv2.imshow('delta', delta_frame)'''
    cv2.imshow("thresh", thresh_delta)
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
print(face_re)
video1.release()
cv2.destroyAllWindows()


'''face_cascade = cv2.CascadeClassifier("/home/pi/Pictures/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
img1 = cv2.imread ("/home/pi/imagetest5.jpg", 1)
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img1, scaleFactor = 1.05, minNeighbors = 5)
for x,y,w,h in faces:
    img1 = cv2.rectangle(img1, (x,y), (x+w,y+h), (0,255,0),3)
cv2.imshow("Grey",img1)
cv2.waitKey(0)  
cv2.destroyAllWindows()'''

 
'''print("What would you like the system to recognise you as?")
                                input1 = input()
                                
                                jsonObject["person" + str(len(jsonObject))] = {
                                    "name":input1,
                                    "face_encoding":encodings
                                    }
                                print(jsonObject)
                                with open("face_recognition_data.json", 'w') as jsonFile:
                                    json.dump(jsonObject, jsonFile, cls=NumpyEncoder, indent=4)
                                    jsonFile.close()
                                list_of_seen.append(
                                {
                                    "name":"Stranger",
                                    "count":0
                                }
                                )'''


'''img1_resized = cv2.resize(img1, (2000,2000))
cv2.imshow("Obama", img1_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
