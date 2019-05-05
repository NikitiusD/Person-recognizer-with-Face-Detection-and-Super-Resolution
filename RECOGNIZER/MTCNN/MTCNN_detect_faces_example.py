import cv2
import sys
from mtcnn import MTCNN


if __name__ == "__main__":
    detector = MTCNN()
    file_name = sys.argv[1]
    image = cv2.imread(f"{file_name}.jpg")
    result = detector.detect_faces(image)

    color = (0,155,255)

    for face in result:
        bounding_box = face['box']
        keypoints = face['keypoints']

        cv2.rectangle(image, 
                    (bounding_box[0], bounding_box[1]), 
                    (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), 
                    color, 
                    2)

        cv2.circle(image, (keypoints['left_eye']), 2, color, 2)
        cv2.circle(image, (keypoints['right_eye']), 2, color, 2)
        cv2.circle(image, (keypoints['nose']), 2, color, 2)
        cv2.circle(image, (keypoints['mouth_left']), 2, color, 2)
        cv2.circle(image, (keypoints['mouth_right']), 2, color, 2)

    cv2.imwrite(f"{file_name}_face.jpg", image)
    print(result)
