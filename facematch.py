import face_recognition

image_of_bill = face_recognition.load_image_file('img/XpgonN0X_400x400.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

unknown_image = face_recognition.load_image_file('img/download.jpeg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# compare faces

results = face_recognition.compare_faces(
    [bill_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is bill gates')
else:
    print("not bill gates")
