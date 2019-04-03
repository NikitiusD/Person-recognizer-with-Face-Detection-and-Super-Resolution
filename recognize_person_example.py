from recognizer import Recognizer
from skimage.io import imread

THRESHOLD = 0.9
photo1 = imread('me(1).jpg')
photo2 = imread('matthew.jpg')
recognizer = Recognizer()
similarity = recognizer.get_best_similarity(photo1, photo2)
if similarity < THRESHOLD:
    print('\nRECOGNIZED')
else:
    print('\nNOT RECOGNIZED')