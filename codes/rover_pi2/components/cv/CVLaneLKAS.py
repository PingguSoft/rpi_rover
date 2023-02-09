import logging
from components.cv.CVProcessor import *


def disp_lines(self, frame, lines, dewarp=True, line_width=2):
    line_image = np.zeros_like(frame)
    colorTbl = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
    i = 0
    if lines is not None:
        for line in lines:
            if line is not None:
                for x1, y1, x2, y2 in line:
                    line_color = colorTbl[i]
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
                    i = (i + 1) % len(colorTbl)

    height, width, _ = frame.shape
    if dewarp:
        inv_warp = self.warp_image(line_image, CVNodeBase._roi, width, height, inv=True)
    else:
        inv_warp = line_image
    line_image = cv2.addWeighted(frame, 0.8, inv_warp, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # -90 ~ -1 degree: turn left
    # 0        degree: going straight
    # 1 ~ 90   degree: turn right
    steering_angle_radian = (steering_angle + 90) / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    cv2.putText(heading_image, f'{steering_angle:5.2f}',
                (heading_image.shape[1] // 2 - 40, heading_image.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 255), 2)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image


def length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def sort_lines(e):
    return e[2]

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
        # lower_black = np.array([0, 0, 0])     # black
        # upper_black = np.array([179, 255, 90])
        lower_black = np.array([10, 94, 100])   # orange
        upper_black = np.array([35, 255, 255])
        img_mask = cv2.inRange(hsv, lower_black, upper_black)
        img_edges = cv2.Canny(img_mask, 200, 400)
        return img_edges, img_mask

    def process(self, args):
        pos, frame, _, _ = args
        img_cropped = self._crop_warp_roi(CVNodeBase._roi, frame)
        img_edges, img_mask = self._detect_edges(img_cropped)
        return [pos, frame, img_edges, None]


###############################################################################
# ProcDetectLineSegments
###############################################################################
class NodeDetectLineSegments(CVNodeBase):
    def _detect_line_segments(self, cropped_edges):
        # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
        rho = 1  # precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # degree in radian, i.e. 1 degree
        min_threshold = 10  # minimal of votes
        line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                        np.array([]), minLineLength=40, maxLineGap=10)
        self.log().debug('detected line_segment:')
        ret = []
        if line_segments is not None:
            for i, line_segment in enumerate(line_segments):
                l = length(line_segment[0])
                x1, y1, x2, y2 = line_segment[0]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    self.log().debug(f'{i:3d} len={l:6.3f} slope={slope:6.3f} {line_segment} ')
                    if abs(slope) > 0.5:
                        ret.append(line_segment)
        return ret

    def process(self, args):
        pos, frame, img_edges, _ = args
        line_segs  = self._detect_line_segments(img_edges)
        img_edge_m = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
        img_line_segs = disp_lines(self, img_edge_m, line_segs, False) if self._is_res_img_en else img_edge_m
        return [pos, frame, img_line_segs, line_segs]


###############################################################################
# ProcAverageSlopeIntercept
###############################################################################
class NodeAverageSlopeIntercept(CVNodeBase):
    def _make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept, _, _ = line
        y1 = height
        y2 = int(y1 * 0.5) # 0
        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]


    def _calcFit(self, name, frame, fit, line_segments):
        height, width, _ = frame.shape

        str = ""
        for l in fit:
            str = str + f'({l[0]:6.3f}, {l[1]:6.3f}, {l[2]:6.3f}, {l[3]:3d}), '
        self.log().debug(f'{name:5s} fit  dat: {str}')

        mean = np.mean(fit, axis=0)
        if mean[0] == float("inf"):                 # vertical line
            self.log().debug(f'{name:5s} fit  vertical line !!')
            _, _, _, idx = fit[0]
            x, _, _, _ = line_segments[idx][0]
            lane = [[x, height, x, 0]]
        else:
            fit_avg = np.average(fit, axis=0)
            self.log().debug(f'{name:5s} fit  avg: {fit_avg}')
            slope, _, _, _ = fit_avg
            lane = self._make_points(frame, fit_avg)

        self.log().debug(f'{name:5s} fit lane: {lane}')
        return lane


    def _average_slope_intercept(self, frame, line_segments):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """
        lane_lines = []
        if line_segments is None:
            self.log().info('No line_segment segments detected')
            return lane_lines

        height, width, _ = frame.shape
        left_fit  = []
        right_fit = []
        boundary  = 1 / 3
        left_region_boundary  = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

        for i, line_segment in enumerate(line_segments):
            l = length(line_segment[0])
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    self.log().info('skipping vertical line segment (slope=inf): %s' % line_segment)
                    # vertical line segment
                    fit = float("inf"), float("inf")
                else:
                    fit = np.polyfit((x1, x2), (y1, y2), 1)

                slope, intercept = fit
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept, l, i))
                else:
                    if x1 >= right_region_boundary and x2 >= right_region_boundary:
                        right_fit.append((slope, intercept, l, i))

        if len(left_fit) > 0:
            left_fit.sort(reverse=True, key=sort_lines)
            lane = self._calcFit('left', frame, left_fit, line_segments)
            lane_lines.append(lane)
        if len(right_fit) > 0:
            right_fit.sort(reverse=True, key=sort_lines)
            lane = self._calcFit('right', frame, right_fit, line_segments)
            lane_lines.append(lane)
        return lane_lines

    def process(self, args):
        pos, frame, img_line_segs, line_segs = args
        lane_lines = self._average_slope_intercept(img_line_segs, line_segs)
        img_lane = disp_lines(self, img_line_segs, lane_lines, False, 5) if self._is_res_img_en else img_line_segs
        return [pos, frame, img_lane, lane_lines]


###############################################################################
# ProcCalcSteeringAngle
###############################################################################
class NodeCalcSteeringAngle(CVNodeBase):
    def __init__(self):
        super().__init__()
        self._curr_steering_angle = 0

    def _compute_steering_angle(self, shape, lane_lines):
        """ Find the steering angle based on lane line coordinate
            We assume that camera is calibrated to point to dead center
        """
        if len(lane_lines) == 0:
            self.log().info('No lane lines detected, do nothing')
            return -90

        height, width, _ = shape
        if len(lane_lines) == 1:
            x1, y1, x2, y2 = lane_lines[0][0]
            slope = ((y2 - y1) / (x2 - x1)) if x1 != x2 else 0
            self.log().debug(f'slope={slope}')
            if slope < 0:
                self.log().debug('just follow left  lane ' % lane_lines[0][0])
                # left_x2  = x2
                # right_x2 = left_x2 + CVNodeBase._roi_width
                x_offset = x2 - x1
            else:
                self.log().debug('just follow right lane ' % lane_lines[0][0])
                # right_x2 = x2
                # left_x2  = right_x2 - CVNodeBase._roi_width
                x_offset = x2 - x1
        else:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            camera_mid_offset_percent = 0.0 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(width / 2 * (1 + camera_mid_offset_percent))
            # find the steering angle, which is angle between navigation direction to end of center line
            x_offset = (left_x2 + right_x2) / 2 - mid

        # if left_lane == None:
        #     self.log().debug('just follow right lane. %s' % right_lane)
        #     _, _, right_x2, _ = right_lane
        #     left_x2 = right_x2 - CVProcNodeBase._roi_width
        # elif right_lane == None:
        #     self.log().debug('just follow left  lane. %s' % left_lane)
        #     _, _, left_x2, _ = left_lane
        #     right_x2 = left_x2 + CVProcNodeBase._roi_width


        y_offset = int(height / 2)
        angle_to_mid_radian = math.atan(x_offset / y_offset)            # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)   # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg                               # this is the steering angle needed by picar front wheel
        self.log().debug(f'x_offset: {x_offset}')
        self.log().debug(f'new steering angle: {steering_angle:5.2f}')

        return steering_angle

    def _stabilize_steering_angle(self, _curr_steering_angle, new_steering_angle, num_of_lane_lines,
                                 max_angle_deviation_two_lines=15, max_angle_deviation_one_lane=10):
        """
        Using last steering angle to stabilize the steering angle
        This can be improved to use last N angles, etc
        if new angle is too different from current angle, only turn by max_angle_deviation degrees
        """
        max_angle_deviation = max_angle_deviation_two_lines if num_of_lane_lines == 2 else max_angle_deviation_one_lane

        angle_deviation = new_steering_angle - _curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(_curr_steering_angle
                                            + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        #self.log().info(f'Proposed angle: {new_steering_angle:5.2f}, stabilized angle: {stabilized_steering_angle:5.2f}')
        return stabilized_steering_angle


    def _steer(self, shape, lane_lines):
        if len(lane_lines) == 0:
            #self.log().error('No lane lines detected, nothing to do.')
            return self._curr_steering_angle

        new_steering_angle = self._compute_steering_angle(shape, lane_lines)
        if -90 < new_steering_angle and new_steering_angle < 90:
            self._curr_steering_angle = self._stabilize_steering_angle(self._curr_steering_angle, new_steering_angle,
                                                                    len(lane_lines))
        return self._curr_steering_angle #new_steering_angle

    def process(self, args):
        pos, frame, img_lane, lane_lines = args
        angle = self._steer(img_lane.shape, lane_lines)
        if self._is_res_img_en:
            img_lane = disp_lines(self, frame, lane_lines, True, 5)
            img_heading = display_heading_line(img_lane, angle)
        else:
            img_heading = img_lane
        return [pos, img_heading, img_heading, angle]


###############################################################################
# CVLaneDetection
###############################################################################
class CVLaneLKAS(CVProcessor):
    def __init__(self, log, roi, is_show = 1.0, cols = 4):
        nodes = [NodeEdgeDetect, NodeDetectLineSegments, NodeAverageSlopeIntercept, NodeCalcSteeringAngle]
        super().__init__(__class__.__name__, nodes, log, roi, is_show, cols)

    # def set_roi2(self, width, height, top_h_ratio, bot_h_ratio, top_ratio, bottom_ratio):
    #     top_l = width * top_h_ratio
    #     top_r = width * (1 - top_h_ratio)
    #     bot_l = width * bot_h_ratio
    #     bot_r = width * (1 - bot_h_ratio)
    #     top   = height * top_ratio
    #     bot   = height * bottom_ratio
    #     self.roi = np.array([[
    #         (top_l,  top),
    #         (top_r,  top),
    #         (bot_r,  bot),
    #         (bot_l,  bot),
    #     ]], np.int32)
    #     CVProcNodeBase._roi_width  = ((bot_r - bot_l) + (top_r - top_l)) // 2
    #     CVProcNodeBase._roi_height = bot - top
    #     CVProcNodeBase.setROI(self.roi)
    #     #CVProcNodeBase._roi = self.roi
