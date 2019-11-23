# -*- coding: utf-8 -*-
"""
Created on Friday Dec 22 22:00:00 2019
@author: Jackyongjian-Li
This is an easy example for face recognition in image.
"""
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os

#global variables
num_jitters = 10
number_of_times_to_upsample = 2

def get_all_picturepath(directory):
    all_picture_names = []
    all_picture_path = []
    for home, dirs, files in os.walk(directory):
        for picture_name in files:
            if picture_name not in all_picture_names:
                all_picture_names.append(picture_name)
                picture_path = os.path.join(home, picture_name)
                all_picture_path.append(picture_path)

    return all_picture_path


#Initialize some variables
all_known_face_encodings = []
all_known_people_names = []

picture_path = get_all_picturepath("Images/sample_people")
for i in range(len(picture_path)):
    all_known_face_encodings.append(object)

adder = 0
for path_ in picture_path:
    image_ = face_recognition.load_image_file(path_)
    all_known_face_encodings[adder] = face_recognition.face_encodings(image_,num_jitters=num_jitters)[0]
    picture_name = str(path_.split('\\')[len(path_.split('\\'))-1])
    all_known_people_names.append(picture_name.split('.')[0])
    adder += 1

print(len(all_known_face_encodings),all_known_people_names)
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("Images/unknown_people/11.jpg")

# Find all the faces and face encodings in the unknown image
unknown_face_locations = face_recognition.face_locations(unknown_image,number_of_times_to_upsample=number_of_times_to_upsample)
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations,num_jitters=num_jitters)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(all_known_face_encodings, face_encoding, tolerance=0.5)
    print("matches:", matches)
    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    if True in matches and matches.count(True) == 1:
        print("only one matches.\n")
        first_match_index = matches.index(True)
        name = all_known_people_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    if True in matches and matches.count(True) > 1:
        face_distances = face_recognition.face_distance(all_known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = all_known_people_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)

    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# # You can also save a copy of the new image to disk if you want by uncommenting this line
# # pil_image.save("image_with_boxes.jpg")
