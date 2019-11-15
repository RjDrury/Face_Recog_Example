import face_recognition

image = face_recognition.load_image_file("img/46950216-large-group-of-people-community-teamwork-concept.jpg")
face_locations = face_recognition.face_locations(image)

print(f'there are {len(face_locations)} people in this image')