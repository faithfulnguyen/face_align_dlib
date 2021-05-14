import dlib
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import os

def list_all_img(folder):
    all_file = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            path_file_name = os.path.join(root, name)
            all_file.append(path_file_name)
    return all_file

def calc_coordinates_after_rotate(xn, yn, w, h, angle, eyes_center):
    theta = angle * math.pi / 180.0
    xcenter = eyes_center[0]
    ycenter = eyes_center[1]
    xnew = xcenter + int((xn - xcenter) * math.cos(theta) - (yn - xcenter) * math.sin(theta));
    ynew = ycenter + int((xn - xcenter) * math.sin(theta) + (yn - xcenter) * math.cos(theta));
    return xnew, ynew


def calc_bounding_box(angle, bounding_box, eyes_center, w, h):
    bb = np.zeros(4, dtype=np.int32)
    bb[0], bb[1] = calc_coordinates_after_rotate(bounding_box[0], bounding_box[1], w, h, angle, eyes_center)
    bb[2], bb[3] = calc_coordinates_after_rotate(bounding_box[2], bounding_box[3], w, h, angle, eyes_center)
    return bb


def check_condition(top_left, bottom_right, w, h):
    if (top_left[0] < 0):
        top_left[0] = 0

    if (top_left[1] < 0):
        top_left[1] = 0

    if (bottom_right[0] > w - 1):
        bottom_right[0] = w - 1

    if (bottom_right[1] > h - 1):
        bottom_right[1] = h - 1

    return top_left, bottom_right


def recalc_coordinates(top_left, top_right, bottom_right, bottom_left):
    top_left[1] = np.min([top_left[1], top_right[1]])
    bottom_right[1] = np.max([bottom_right[1], bottom_left[1]])
    return top_left, bottom_right


def get4corner(bounding_box):
    top_left = np.array([bounding_box[0], bounding_box[1], 1])
    top_right = np.array([bounding_box[2], bounding_box[1], 1])
    bottom_right = np.array([bounding_box[2], bounding_box[3], 1])
    bottom_left = np.array([bounding_box[0], bounding_box[3], 1])
    return top_left, top_right, bottom_right, bottom_left


def calc_bounding_box_with_matrix(bounding_box, matrix_rotate, w, h):
    bb = np.zeros(4, dtype=np.int32)
    top_left, top_right, bottom_right, bottom_left = get4corner(bounding_box)

    top_left = np.dot(matrix_rotate, top_left.T)
    bottom_right = np.dot(matrix_rotate, bottom_right.T)

    top_right = np.dot(matrix_rotate, top_right.T)
    bottom_left = np.dot(matrix_rotate, bottom_left.T)

    top_left, bottom_right = recalc_coordinates(top_left, top_right, bottom_right, bottom_left)
    top_left, bottom_right = check_condition(top_left, bottom_right, w, h)
    bb[0] = top_left[0]
    bb[1] = top_left[1]
    bb[2] = bottom_right[0]
    bb[3] = bottom_right[1]
    return bb


def euclidean(p1, p2):
    dis = (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
    return math.sqrt(dis)


'''
bug still need to fix.
'''


def expand_with_eyes(w_in, h_in, landmark, bounding_box, matrix_rotate, percent_expand_w, percent_expand_h):
    left_eye = np.array([landmark[0], landmark[1], 1])
    right_eye = np.array([landmark[2], landmark[3], 1])

    left_eye = list(map(int, np.dot(matrix_rotate, left_eye.T)))
    right_eye = list(map(int, np.dot(matrix_rotate, right_eye.T)))

    w = euclidean(left_eye, right_eye)
    new_w = w * percent_expand_w
    new_h = (bounding_box[3] - bounding_box[1]) * percent_expand_h
    bounding_box[0] = max(left_eye[0] - new_w, 0)
    bounding_box[2] = min(left_eye[1] + new_w, w_in)
    bounding_box[1] = max(right_eye[0] - new_h, 0)
    bounding_box[3] = min(right_eye[1] + new_h, h_in)
    return bounding_box


def expand_boundingbox(w_in, h_in, bounding_box, matrix_rotate, percent_expand_w, percent_expand_h):
    expnd_w = (bounding_box[2] - bounding_box[0]) * percent_expand_w
    expnd_h = (bounding_box[3] - bounding_box[1]) * percent_expand_h
    bounding_box[0] = max(bounding_box[0] - expnd_w, 0)
    bounding_box[2] = min(bounding_box[2] + expnd_w, w_in)
    bounding_box[1] = max(bounding_box[1] - expnd_h, 0)
    bounding_box[3] = min(bounding_box[3] + expnd_h, h_in)
    return bounding_box


def calc_angle(img, landmark, bounding_box, percent_expand_w, percent_expand_h):
    h, w = img.shape[:2]
    left_eye = (landmark[0], landmark[1])
    right_eye = (landmark[2], landmark[3])

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    eyes_center = ((right_eye[0] + left_eye[0]) / 2, (right_eye[1] + left_eye[1]) / 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    bounding_box = calc_bounding_box_with_matrix(bounding_box, M, w, h)

    bounding_box = list(map(int, bounding_box))

    bounding_box = expand_boundingbox(w, h, bounding_box, M, percent_expand_w, percent_expand_h)
    return output, bounding_box


def convert_landmark(landmark):
    left_eye_xcenter = (landmark[2][0] + landmark[3][0]) / 2
    left_eye_ycenter = (landmark[2][1] + landmark[3][1]) / 2

    right_eye_xcenter = (landmark[0][0] + landmark[1][0]) / 2
    right_eye_ycenter = (landmark[0][1] + landmark[1][1]) / 2

    return [left_eye_xcenter, left_eye_ycenter, right_eye_xcenter, right_eye_ycenter]


def convert_to_boundingbox(bounding_box):
    (x, y, w, h) = face_utils.rect_to_bb(bounding_box)
    return [x, y, x + w, y + h]


def convert_to_orginal_coordinate(w_org, h_org, bounding_box, landmark, w_resize, h_resize):
    bounding_box[0] = (bounding_box[0] / w_resize) * w_org
    bounding_box[2] = (bounding_box[2] / w_resize) * w_org

    bounding_box[1] = (bounding_box[1] / h_resize) * h_org
    bounding_box[3] = (bounding_box[3] / h_resize) * h_org

    landmark[0] = (landmark[0] / w_resize) * w_org
    landmark[2] = (landmark[2] / w_resize) * w_org

    landmark[1] = (landmark[1] / h_resize) * h_org
    landmark[3] = (landmark[3] / h_resize) * h_org

    return bounding_box, landmark


def detect_face_use_dlib(image_path, detector, predictor, image_size, output_dir, file_name):
    image = cv2.imread(image_path)
    h_org, w_org, _ = image.shape

    w_resize = w_org
    ratio_resize = 1
    h_resize = h_org

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(image, 1)

    success = False
    for (i, rect) in enumerate(dets):
        print(type(rect))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        bounding_box = convert_to_boundingbox(rect)
        bounding_box = list(map(int, bounding_box))

        landmark = convert_landmark(shape)

        bounding_box = list(map(int, bounding_box))

        bounding_box, landmark = convert_to_orginal_coordinate(
            w_org, h_org, bounding_box, landmark, w_resize, h_resize
        )

        bounding_box = list(map(int, bounding_box))
        #cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]

        image, bb = calc_angle(image, landmark, bounding_box, percent_expand_w=0.25, percent_expand_h=0.25)
        bb = list(map(int, bb))
        cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
        h_crop, w_crop = cropped.shape[:2]
        if (h_crop > 0 and w_crop > 0):
            scaled = cv2.resize(cropped, (image_size, image_size), cv2.INTER_CUBIC)
            output_filename = os.path.join(output_dir, str(i) + "_" + file_name)
            cv2.imwrite(output_filename, scaled)
            success = True
    return success

if __name__ == '__main__':
    predictor_path = "./shape_predictor_5_face_landmarks.dat"
    face_file_path = "./lili.jpg"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    detect_face_use_dlib(
        face_file_path, detector, predictor, image_size = 128,
        output_dir = "./", file_name="image.jpg"
    )
