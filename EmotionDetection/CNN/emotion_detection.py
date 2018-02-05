"""
This module is the main module in this package. It loads emotion recognition model from a file,
shows a webcam image, recognizes face and it's emotion and draw emotion on the image.
"""

import cv2
from face_detection import find_faces
import pickle
from keras.models import model_from_json
from keras.preprocessing.image import *


def show_webcam_and_run(model, window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, web_cam_image = vc.read()
        cv2.imwrite("images/web.jpg", web_cam_image)
    else:
        print("web camera not found!")
        return

    while read_value:
        for normalized_face, (x, y, w, h) in find_faces(web_cam_image):
            numpy_image = img_to_array(normalized_face)
            cv2.imwrite("images/numpy.jpg", numpy_image)
            image_batch = np.expand_dims(numpy_image, axis=0)
            cv2.imwrite("images/batch.jpg", image_batch)
            prediction = model.predict(image_batch)  # do prediction
            print(prediction)

        cv2.imshow(window_name, web_cam_image)
        read_value, web_cam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)


if __name__ == '__main__':
    emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # load the model from disk
    json_file = open('trained_data/model_4layer_2_2_pool.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # use learnt model
    window_name = 'WEB_CAM (press ESC to exit)'
    show_webcam_and_run(model, window_size=(1280, 800), window_name=window_name, update_time=8)
