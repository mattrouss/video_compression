import os
import time
import argparse

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2


def read_pgm(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def yuv_to_rgb(frame):
    y_border = 2 * frame.shape[0] // 3
    Y = frame[:y_border, :]

    V = frame[y_border:, frame.shape[1] // 2:].repeat(2, axis = 0).repeat(2, axis = 1)
    U = frame[y_border:, :frame.shape[1] // 2].repeat(2, axis = 0).repeat(2, axis = 1)

    yuv = np.stack((Y, U, V), axis=-1).astype(np.uint8)

    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])

    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    rgb = rgb.clip(0,255)

    return rgb.astype(np.uint8)

def write_ppm(img, file_path):
    maxval = 255
    height, width, _ = img.shape
    ppm_header = f'P6 {width} {height} {maxval}\n'
    with open(file_path, 'wb') as f:
        f.write(bytearray(ppm_header, 'ascii'))
        img.tofile(f)


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    if not args.disp and not os.path.isdir(output_path):
        print(f"Error: could not find output path: {output_path}")
        return

    filenames = os.listdir(input_path)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))

    frames = []
    # Load all frames
    for frame_path in sorted_filenames:
        yuv = read_pgm(f'{input_path}/{frame_path}')
        rgb = yuv_to_rgb(yuv)

        frames.append(rgb)

    for i, frame in enumerate(tqdm(frames)):
        if args.disp:
            cv2.imshow("Image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            time.sleep(1 / args.fps)
        else:
            write_ppm(frame, f'{output_path}/{i}.ppm')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--disp", action='store_true')
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    main(args)
