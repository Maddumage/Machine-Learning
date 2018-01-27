#------------------Facial Emotion Recognition Using Convolution Neural Network -----------------------------

This convolution neural network is based on the http://www.paulvangent.com emotion recognition tutorial and this (https://github.com/PiotrDabrowskey/facemoji) repository. 
This use ,
01. Python 3.6 with Anaconda3
02. sklearn and SVM(Support Vector Machine) classifiers
03. Cohn Canade Emotion Dataset, Download iit from http://www.consortium.ri.cmu.edu/ckagree/.
04. haar_cascade_classifiers for face detection and use shape_predictor_68_face_landmarks.dat file for facial land mark 
    Download shape_predictor - https://osdn.net/projects/sfnet_dclib/downloads/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/
    Download haar_cascade files - https://github.com/opencv/opencv/tree/master/data/haarcascades
05. There is breaf details about face detection using Haar Cascade https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

#-----------How to prepare dataset----------------- 
01. First download the Cohn Canade Dataset from the above mentioned link.
02. Extract the downloaded file and it contains two folders called 'cohn-kanade-images' and 'Emotion'
03. In project location create new two folders called 'source_images' and 'source_emotion'.
04. Put Emotion folder contents into the source_emotion folder in the project, and put cohn-kanade-images into source_emotion folder.
05. Then create an another folder to store dataset as 'dataset'.
06. In the dataset folder create another 8 folders for 8 emotions such as 'neutral', 'anger', 'disgust', 'contempt', 'fear', 'happy', 'sadness' and surprice'.
07. All are ready and hit run the process_dataset.py file..
08. Above all the details include in the (https://github.com/PiotrDabrowskey/facemoji) 
