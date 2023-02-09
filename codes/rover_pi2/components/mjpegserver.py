import cv2
from   threading    import Thread
from   http.server  import BaseHTTPRequestHandler, HTTPServer
from   socketserver import ThreadingMixIn
import socket
import time
import queue
import platform

_is_rpi = platform.machine() in ('armv7l', 'armv6l', 'aarch64')

if _is_rpi:
    from picamera2 import Picamera2

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


class MjpegServer():
    def __init__(self, port):
        self._webServer    = None
        self._threadServer = None
        self._port         = port

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address

    def start(self):
        global _qFrames
        global _ipHost
        global _isRunning

        _isRunning = True
        _ipHost = self.get_ip_address()
        self._webServer = HTTPServer((_ipHost, self._port), HttpHandler)
        print(">> mjpeg server started http://%s:%s" % (_ipHost, self._port))
        self._threadServer = Thread(target=self._webServer.serve_forever, args=())
        self._threadServer.start()

    def stop(self):
        global _isRunning

        _isRunning = False
        self._webServer.shutdown()
        self._threadServer.join()
        print("<< mjpeg server stopped.")

    def putFrame(self, frame):
        frameCtr = _qFrames.qsize()
        if  frameCtr > 5:
            _qFrames.get(False)
        _qFrames.put(frame)


def openCam(frameWidth, frameHeight):
    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('cam is not opened')
        return

    print('cam is opened')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    #cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    return cap

def main():
    global _isRunning

    mjpegServer = MjpegServer(8080)

    if _is_rpi:
        capture = Picamera2()
        capture.configure(capture.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
        capture.start()
    else:
        capture = openCam(640, 480)

    mjpegServer.start()

    while _isRunning:
        try:
            if _is_rpi:
                img = capture.capture_array()
            else:
                rc, img  = capture.read()
            if img is not None:
                mjpegServer.putFrame(img)
        except KeyboardInterrupt:
            print('CTRL-C')
            _isRunning = False

    if _is_rpi:
        if capture is not None:
            capture.stop()
    else:
        if capture is not None and capture.isOpened():
            capture.release()

    mjpegServer.stop()


if __name__ == '__main__':
    main()
