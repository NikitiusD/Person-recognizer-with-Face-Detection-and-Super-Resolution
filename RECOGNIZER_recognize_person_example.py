from RECOGNIZER.recognizer import Recognizer
from skimage.io import imread
import sys

THRESHOLD = 0.9
photo1 = imread(sys.argv[1])
photo2 = imread(sys.argv[2])
recognizer = Recognizer()
similarity = recognizer.get_best_similarity(photo1, photo2)
if similarity < THRESHOLD:
    print('\nRECOGNIZED')
else:
    print('\nNOT RECOGNIZED')