import argparse

import numpy as np
import matplotlib.pyplot as plt

import cv2

def read_pgm(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def yuv_to_rgb(frame):
    y_border = 2 * frame.shape[0] // 3
    Y = frame[:y_border, :]

    U = frame[y_border:, frame.shape[1] // 2:].repeat(2, axis = 0).repeat(2, axis = 1)
    V = frame[y_border:, :frame.shape[1] // 2].repeat(2, axis = 0).repeat(2, axis = 1)

    frame_yuv = np.stack((Y, U, V), axis=-1).astype(np.float32)
    m = np.array([
        [1.000,  1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])
    frame_yuv[:, :, 1:] -= 127.5
    frame_rgb = np.dot(frame_yuv, m).astype(np.float32)

    return frame_rgb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()

    img = read_pgm(args.file_path)
    rgb = yuv_to_rgb(img)

    plt.imshow(rgb)
    plt.show()


