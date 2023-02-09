from components.cv.CVProcessor import *


def getHistogram(img, minPer=0.1, display=False, region=1):
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        startY = img.shape[0] - int(img.shape[0] / region)
        histValues = np.sum(img[startY:, :], axis=0)

    # print(histValues)
    maxValue = np.max(histValues)
    minValue = minPer * maxValue

    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    # print(basePoint)

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - int(intensity / 255 / region)), (255, 0, 255),
                     1)
            cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
        return basePoint, imgHist
    return basePoint



###############################################################################
# ProcEdgeDetect
###############################################################################
class NodeEdgeDetect(CVNodeBase):
    def _crop_warp_roi(self, roi, frame):
        height, width, _ = frame.shape
        img_cropped = self.warp_image(frame, roi, width, height)
        return img_cropped

    def _detect_edges(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([179, 255, 120])
        img_mask = cv2.inRange(hsv, lower_black, upper_black)
        img_edges = cv2.Canny(img_mask, 200, 400)
        return img_edges, img_mask

    def process(self, args):
        pos, frame, _, _ = args
        img_cropped = self._crop_warp_roi(CVNodeBase._roi, frame)
        img_edges, img_mask = self._detect_edges(img_cropped)
        return [pos, img_mask, img_mask, None]


###############################################################################
# ProcHistogram1
###############################################################################
class NodeHistogram1(CVNodeBase):
    def process(self, args):
        pos, img_mask, _, _   = args
        middlePoint, imgHist1 = getHistogram(img_mask, display=True, minPer=0.5, region=4)
        return [pos, img_mask, imgHist1, middlePoint]


###############################################################################
# ProcHistogram2
###############################################################################
class NodeHistogram2(CVNodeBase):
    def process(self, args):
        pos, img_mask, _, middlePoint = args
        curveAveragePoint, imgHist2 = getHistogram(img_mask, display=True, minPer=0.9, region=1.5)
        curve = curveAveragePoint - middlePoint
        if self._is_res_img_en:
            cv2.putText(imgHist2, f'angle {curve:5.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(imgHist2, f'angle {curve:5.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return [pos, img_mask, imgHist2, curve]


###############################################################################
# CVLaneDetHistogram
###############################################################################
class CVLaneHistogram(CVProcessor):
    def __init__(self, log, roi, is_show=True, cols=4):
        nodes = [NodeEdgeDetect, NodeHistogram1, NodeHistogram2]
        super().__init__(__class__.__name__, nodes, log, roi, is_show, cols)
