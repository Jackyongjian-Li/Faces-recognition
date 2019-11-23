# -*- coding: utf-8 -*-
"""
Created on Friday Dec 22 22:00:00 2019
@author: Jackyongjian-Li
This is an easy example for face recognition in video which runs fast but it may be less accurate,a more accurate example
is using the k-nearest-neighbors (KNN) algorithm in 'recognition_people_from_video.py',while it runs slower.
"""
import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("test_video/4_1080p_20s.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output_video/4_1080p_20s_output.avi', fourcc, 25.00, (1920, 1080))

# Load some sample pictures and learn how to recognize them.
lxp_image = face_recognition.load_image_file("Images/sample_people/li xiaopang.png")
lxp_face_encoding = face_recognition.face_encodings(lxp_image)[0]

xukai_image = face_recognition.load_image_file("Images/sample_people/xu kai.png")
xukai_face_encoding = face_recognition.face_encodings(xukai_image)[0]

zm_image = face_recognition.load_image_file("Images/sample_people/wu.png")
zm_face_encoding = face_recognition.face_encodings(zm_image)[0]

hwj_image = face_recognition.load_image_file("Images/sample_people/he wenjun.png")
hwj_face_encoding = face_recognition.face_encodings(hwj_image)[0]

xlc_image = face_recognition.load_image_file("Images/sample_people/xu linchen.png")
xlc_face_encoding = face_recognition.face_encodings(xlc_image)[0]

zyb_image = face_recognition.load_image_file("Images/sample_people/zhu yuanbing.png")
zyb_face_encoding = face_recognition.face_encodings(zyb_image)[0]

known_faces = [
    lxp_face_encoding,
    xukai_face_encoding,
    zm_face_encoding,
    hwj_face_encoding,
    xlc_face_encoding,
    zyb_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        print("match:",match)
        name = None
        if match[0]:
            name = "Li xiaopang"
        elif match[1]:
            name = "Xu kai"
        elif match[2]:
            name = "zhangmenren"
        elif match[3]:
            name = "He wenjun"
        elif match[4]:
            name = "Xu linchen"
        elif match[5]:
            name = "Zhu yuanbing"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
