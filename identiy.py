import face_recognition
from PIL import Image, ImageDraw

image_of_bill = face_recognition.load_image_file('img/XpgonN0X_400x400.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

image_of_steve = face_recognition.load_image_file(
    'img/Steve-Jobs-action-figure-portrait.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]


# create array of encodings and names

known_face_encodings = [
    bill_face_encoding,
    steve_face_encoding
]
known_face_names = [
    "bill gates",
    "steve jobs"
]

# load test image to find faces
test_image = face_recognition.load_image_file(
    'img/steve-jobs-musk-gates_56037.jpg')

# find faces in test
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# conver to PIL format

pil_image = Image.fromarray(test_image)

# create imagedraw instances

draw = ImageDraw.Draw(pil_image)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding)

    name = "unkown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # draw box
    draw .rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))

    # draw label

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10),
                    (right, bottom)), fill=(255, 255, 0), outline=(255, 255, 0))
    #draw.text((left+6, bottom-text-height-5), name, fill(255, 255, 255, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 0, 0))
del draw

pil_image.save()
