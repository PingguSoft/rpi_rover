import os
import cv2
import sys
from utils   import *

def play(log, filename):
    cap = cv2.VideoCapture(filename + '.mp4')
    in_dat = open(filename + '.dat', 'rb')

    try:
        fname_only = os.path.basename(filename)
        log.info(f'{fname_only=}')
        img_path = filename + '_img'
        if not os.path.exists(img_path):
            os.makedirs(img_path, exist_ok=True)
    except Exception as e:
        log.error(e)

    try:
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                b2 = in_dat.read(2)
                angle = int.from_bytes(b2, byteorder='little', signed=True)
                height, width, _ = frame.shape
                frame = frame[int(height/2):, :, :]
                cv2.imshow('video', frame)

                img_name = f"{img_path}/{fname_only}_{i:05d}_{angle:04d}.png"
                log.info(f'img_name={img_name}, angle={angle:4d}')
                cv2.imwrite(img_name, frame)
                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        in_dat.close()

if __name__ == '__main__':
    _log   = init_log(log_level=logging.INFO)
    _log.info('>> play !!')
    play(_log, sys.argv[1])
