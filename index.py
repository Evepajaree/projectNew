import face_recognition
import cv2
import numpy as np
import time


# เปิดการใช้ webcam
video_capture = cv2.VideoCapture(0)
frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


print(frame_size)
print(cv2.CAP_PROP_FPS)


prevTime = 0

 # Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


# โหลดภาพ Peen.jpg และให้ระบบจดจำใบหน้า
person1_image = face_recognition.load_image_file("Peen.png")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

# โหลดภาพ Stop.jpg และให้ระบบจดจำใบหน้า
person2_image = face_recognition.load_image_file("Stop.png")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]


# โหลดภาพ Eve.jpg และให้ระบบจดจำใบหน้า
person3_image = face_recognition.load_image_file("Eve.jpg")
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

# โหลดภาพ Pla.jpg และให้ระบบจดจำใบหน้า
person4_image = face_recognition.load_image_file("Pla.jpg")
person4_face_encoding = face_recognition.face_encodings(person4_image)[0]

# สร้าง arrays ของคนที่จดจำและกำหนดชื่อ ตามลำดับ
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding,
    person4_face_encoding,
]

known_face_names = [
    "Peen",
    "Stop",
    "Eve",
    "Pla"

]

# ตัวแปรเริ่มต้น
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # ดึงเฟรมภาพมาจากวีดีโอ
    ret, frame = video_capture.read()

    # ย่อขนาดเฟรมเหลือ 1/4 ทำให้ face recognition ทำงานได้เร็วขึ้น
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # แปลงสีภาพจาก BGR (ถูกใช้ใน OpenCV) เป็นสีแบบ RGB (ถูกใช้ใน face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]
   
    #  การตรวจจับเฟรมเรต
    retval, frame = video_capture.read()
    if not retval:
        break

    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
   
    fps = 1/(sec)

    str_fps1 = "FPS Dectection : %d" % fps
    cv2.putText(frame, str_fps1, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    

    # str_fps2 = "FPS Capture : %d" % fps
    # cv2.putText(frame, str_fps2, (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 200, 255))
    # cv2.imshow('frame', frame)


    # ประมวลผลเฟรมเว้นเฟรมเพื่อประหยัดเวลา
    if process_this_frame:
        # ค้นหาใบหน้าที่มีทั้งหมดในภาพ จากนั้นทำการ encodings ใบหน้าเพื่อจะนำไปใช้เปรียบเทียบต่อ
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # ทำการเปรียบเทียบใบหน้าที่อยู่ในวีดีโอกับใบหน้าที่รู้จักในระบบ
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # ถ้า encoding แล้วใบหน้าตรงกันก็จะแสดงข้อมูล
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # แสดงผลลัพธ์
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # ขยายเฟรมที่ลดลงเหลือ 1/4 ให้กลับไปอยู่ในขนาดเดิม 
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # วาดกล่องรอบใบหน้า
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # เขียนตัวหนังสือที่แสดงชื่อลงที่กรอบ
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # แสดงรูปภาพผลลัพธ์
    cv2.imshow('Video', frame)
    cv2.imshow('frame', frame)

    # # กด 'q' เพื่อปิด!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # key = cv2.waitKey(1)
    # if (key == 27):
    # break

    key = cv2.waitKey(1)
    if (key == 27):
        break

if video_capture.isOpened():
    video_capture.release()
        

cv2.destroyAllWindows()