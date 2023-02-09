import sys
import cv2
import socket
import time
import queue
import platform
import logging
from   utils        import *
from   threading    import Thread
from   http.server  import BaseHTTPRequestHandler, HTTPServer
from components.component import Component

_is_rpi     = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
_ipHost     = 'localhost'
_isRunning  = True
_qFrames    = queue.Queue()

class HttpHandler(BaseHTTPRequestHandler):
    global _isRunning
    global _qFrames
    global _ipHost

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()

            while _isRunning:
                try:
                    img = _qFrames.get(True, 0.05)
                    if img is not None:
                        r, buf = cv2.imencode(".jpg", img)
                        self.wfile.write("--jpgboundary\r\n".encode())
                        self.end_headers()
                        self.wfile.write(bytearray(buf))
                        _qFrames.task_done()
                except:
                    pass

        if self.path.endswith('.html') or self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head>'.encode())
            #self.wfile.write('<meta http-equiv="refresh" content="0" >'.encode())
            self.wfile.write('</head><body>'.encode())
            strSrc = '<img src="http://' + _ipHost + ':8080/stream.mjpg"/>'
            self.wfile.write(strSrc.encode())
            self.wfile.write('</body></html>'.encode())
            return


class HTTPServerMJPEG():
    def __init__(self, port, log=None):
        self._webServer    = None
        self._threadServer = None
        self._port         = port
        self._log          = log

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
        except Exception:
            ip_address = '127.0.0.1'
        finally:
            s.close()
        return ip_address

    def start(self):
        global _qFrames
        global _ipHost
        global _isRunning

        _isRunning = True
        _ipHost = self.get_ip_address()
        self._webServer = HTTPServer((_ipHost, self._port), HttpHandler)
        self._log.info(f'>> {__class__.__name__} started http://{_ipHost}.{self._port}')
        self._threadServer = Thread(target=self._webServer.serve_forever, args=())
        self._threadServer.start()

    def stop(self):
        global _isRunning

        _isRunning = False
        self._webServer.shutdown()
        self._threadServer.join()
        self._log.info(f'<< {__class__.__name__} stopped.')

    def putFrame(self, frame):
        frameCtr = _qFrames.qsize()
        if  frameCtr > 5:
            _qFrames.get(False)
        _qFrames.put(frame)

#
#
#
class StreamServer(Component):
    def __init__(self, rate_hz=30):
        Component.__init__(self)
        self.server   = HTTPServerMJPEG(8080, self.log())
        self.hz       = rate_hz
        self._inputs  = None

    def inputs(self):
        return ['disp/image']

    def start(self):
        self.server.start()

    def stop(self):
        self.server.stop()

    def update(self, image):
        if image is not None:
            self.server.putFrame(image)
#
#
#
def main():
    if _is_rpi:
        from picamera2 import Picamera2

    global _isRunning

    def init_log(log_level=logging.INFO):
        logger = logging.getLogger()
        logger.setLevel(log_level)
        hnd = logging.StreamHandler(sys.stdout)
        hnd.setFormatter(logging.Formatter('%(asctime)s [%(process)5d-%(thread)-5d] [%(levelname)8s] ' +
                                           '[%(filename)20s:%(lineno)6d] %(funcName)20s - %(message)s'))
        hnd.setLevel(log_level)
        logger.addHandler(hnd)
        return logger

    def open_camera(frameWidth, frameHeight):
        cap = cv2.VideoCapture()
        cap.open(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
        return cap

    log = init_log(log_level=logging.INFO)
    server = StreamServer()

    capture = Picamera2() if _is_rpi else open_camera(640, 480)
    if capture is None:
        log.error('camera is not opened')
        return

    if _is_rpi:
        capture.configure(capture.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        capture.start()

    server.start()
    while True:
        try:
            if _is_rpi:
                img = capture.capture_array()
            else:
                rc, img  = capture.read()
            if img is not None:
                server.update(img)
        except KeyboardInterrupt:
            break

    if _is_rpi:
        if capture is not None:
            capture.stop()
    else:
        if capture is not None and capture.isOpened():
            capture.release()
    server.stop()

if __name__ == '__main__':
    main()
