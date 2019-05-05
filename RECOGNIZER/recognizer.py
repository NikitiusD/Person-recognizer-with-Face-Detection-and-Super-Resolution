from scipy.spatial import distance
from skimage.transform import rotate
import numpy as np
import tensorflow as tf
from RECOGNIZER.MTCNN.mtcnn import MTCNN

from RECOGNIZER.facenet import load_model, load_image

class Recognizer():
    def __init__(self, image_size=160):
        self.face_detector = MTCNN()
        with tf.Graph().as_default():
            self.sess = tf.Session()
            load_model('RECOGNIZER/facenet_models/facenet/', self.sess)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.images_placeholder = tf.image.resize_images(
                images_placeholder, (image_size, image_size))
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(
                "phase_train:0")
            self.image_size = image_size

    def _get_descriptor(self, image, face):
        images = load_image(face, self.image_size)
        feed_dict = {self.images_placeholder: images,
                     self.phase_train_placeholder: False}
        descriptor = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return descriptor

    def _get_decriptors(self, image):
        descriptors = []
        faces = self.face_detector.detect_faces(image)
        for face in faces:
            x = min(max(0, face['box'][0]), image.shape[0])
            y = min(max(0, face['box'][1]), image.shape[1])
            w = min(max(10, face['box'][2]), image.shape[1] - x)
            h = min(max(10, face['box'][3]), image.shape[0] - y)
            aligned = image[y:y + h, x:x + w, :]
            descriptor = self._get_descriptor(image, aligned)
            if len(descriptor) == 1:
                descriptors.append(descriptor[0])
        return descriptors

    def get_descriptors(self, image):
        descriptors = self._get_decriptors(image)
        for angle in [90, 180, 270]:
            descriptors.extend(self._get_decriptors(
                np.uint8(rotate(image, angle, resize=True) * 255)
            ))
        return descriptors

    def get_best_similarity(self, image1, image2):
        descriptors1 = self.get_descriptors(image1)
        descriptors2 = self.get_descriptors(image2)
        best_similarity = 5.0
        for d1 in descriptors1:
            for d2 in descriptors2:
                best_similarity = min(best_similarity, distance.euclidean(d1, d2))
        return best_similarity