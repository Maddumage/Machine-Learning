"""
This module contains face detections functions.
"""
import cv2

from constants import CASC_PATH

faceCascade = cv2.CascadeClassifier(CASC_PATH)


def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)


def _normalize_face(face):
    cv2.imwrite("images/cutted_face.jpg",face)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("images/gray.jpg",face)
    face = cv2.resize(face, (48, 48))
    cv2.imwrite("images/resized.jpg",face)

    return face


def _locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )
    return faces  # list of (x, y, w, h)


if __name__ == "__main__":
    image = cv2.imread('test_sample/web.jpg')
    cv2.imshow("face", image)

    for index, face in enumerate(find_faces(image)):
        cv2.imshow("face %s" % index, face[0])

    cv2.waitKey(0)
