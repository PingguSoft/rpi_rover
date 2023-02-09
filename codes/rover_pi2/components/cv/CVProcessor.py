import cv2
import numpy as np
import math
import time

###############################################################################
#
###############################################################################
class CVNodeBase:
    _log = None
    _roi = [0, 0, 0, 0]
    _roi_width  = 0
    _roi_height = 0
    _is_res_img_en  = True

    @classmethod
    def set_roi(cls, roi):
        cls._roi = roi
        cls._roi_width = ((roi[2][0] - roi[3][0]) + (roi[1][0] - roi[0][0])) // 2
        cls._roi_height = roi[3][1] - roi[0][1]

    @classmethod
    def roi(cls):
        return cls._roi

    @classmethod
    def set_log(cls, log):
        cls._log = log

    @classmethod
    def log(cls):
        return cls._log

    @classmethod
    def enable_res_img(cls, en):
        cls._is_res_img_en = en
        
    @classmethod
    def is_res_img_enabled(cls):
        return cls._is_res_img_en

    def __init__(self):
        pass

    def process(self, args):
        # 1st pipe in [pos, img_to_cur_pipe,  _,                        _]
        # return      [pos, img_to_next_pipe, img_from_cur_pipe_output, output]
        pass

    def warp_image(self, img, points, w, h, inv=False):
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(pts2, pts1) if inv else cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (w, h))
        return imgWarp

###############################################################################
#
###############################################################################
def drawPoints(img, points, diameter):
    for x in range(len(points)):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), diameter, (0, 0, 255), cv2.FILLED)
    return img


def stackImages(scale, imgArray, textArray = None):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

        if textArray:
            xw   = imgArray[0][0].shape[1]
            yw   = imgArray[0][0].shape[0]
            ypos = 20
            for x in range(0, rows):
                xpos = 10
                for y in range(0, cols):
                    cv2.putText(ver, textArray[x][y], (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(ver, textArray[x][y], (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    xpos += xw
                ypos += yw

    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)

        if textArray:
            xw   = imgArray[0].shape[1]
            ypos = 20
            xpos = 10
            for x in range(0, rows):
                cv2.putText(hor, textArray[x], (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(hor, textArray[x], (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                xpos += xw

        ver = hor
    return ver

class CVProcessor:
    def __init__(self, name, nodes, log, roi, is_show, cols):
        self._name      = name
        self._nodes     = nodes
        self._log       = log
        self._instances = []
        self._cols      = cols
        self._fps_list  = []
        self._last_ev   = 0

        self.set_roi(roi)
        CVNodeBase.enable_res_img(is_show)
        CVNodeBase.set_log(log)
        for node in self._nodes:
            inst = node()
            self._instances.append(inst)

    def process(self, pos, frame):
        args = [pos, frame, None, None]
        if CVNodeBase.is_res_img_enabled():
            img_list = []
            text_list = []
            imgs = []
            texts = []
            imgs.append(frame)
            texts.append(f'input {frame.shape[1]:3d}x{frame.shape[0]:3d}')
            imageBlank = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

        # loop nodes
        start = time.time()
        for i, p in enumerate(self._instances):
            self._log.debug(f'>>> {self._name:20s} NODE {i:2d} : {p.__class__.__name__} ')
            output = p.process(args)
            args = output
            if CVNodeBase.is_res_img_enabled():
                if len(imgs) >= self._cols:
                    img_list.append(imgs)
                    text_list.append(texts)
                    imgs  = []
                    texts = []
                imgs.append(output[2])
                texts.append(p.__class__.__name__)
            self._log.debug('<<< ' + ('-' * 80) + '\n')
        fps = int(1 / (time.time() - start))
        if len(self._fps_list) > 100:
            self._fps_list.pop(0)
        self._fps_list.append(fps)
        fps_avg = int(sum(self._fps_list) / float(len(self._fps_list)))

        if CVNodeBase.is_res_img_enabled():
            num_imgs = len(imgs)
            if num_imgs < self._cols:
                for i in range(0, self._cols - num_imgs):
                    imgs.append(imageBlank)
                    texts.append(' ')
            img_list.append(imgs)
            text_list.append(texts)

        # show stacked image and node names
        if CVNodeBase.is_res_img_enabled():
            img_stacked = stackImages(1.0, img_list, text_list)
            # draw roi points
            drawPoints(img_stacked, CVNodeBase.roi(), 4)
            # draw node name
            y, x = 40, img_stacked.shape[1] - 90
            cv2.putText(img_stacked, f'fps:{fps_avg:3d}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            img_stacked = frame

        ts   = time.time()
        diff = ts - self._last_ev
        if diff > 1:
            self._last_ev = ts
            self._log.info(f'fps:{fps_avg:3d}')

        return output[3], 0, img_stacked

    def set_roi(self, roi):
        CVNodeBase.set_roi(roi)

    def roi(self):
        return CVNodeBase.roi()

    def log(self):
        return self._log
