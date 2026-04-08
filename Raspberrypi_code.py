from flask import Flask, render_template, Response, request, jsonify
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from time import sleep

app = Flask(__name__)

ai_tracking_active = False
ai_tracking_lock = threading.Lock()
recording_active = False
recording_lock = threading.Lock()

multi_thresholds = [
    ((0, 70, 50), (20, 255, 255)),
    ((160, 70, 50), (180, 255, 255)),
]

picam2 = Picamera2()
picam2.start()

video_filename = "output.avi"
frame_size = (640, 480)
fps = 20

factory = PiGPIOFactory()

# Create AngularServo objects for each servo
servo1 = AngularServo(17, min_angle=0, max_angle=180, pin_factory=factory)
servo2 = AngularServo(18, min_angle=0, max_angle=180, pin_factory=factory)
servo3 = AngularServo(27, min_angle=0, max_angle=180, pin_factory=factory)

def setLR(l, r):
    servo2.angle = 90 + l
    servo3.angle = 90 + r

def reset():
    servo1.angle = 90
    servo2.angle = 90
    servo3.angle = 90

reset()

def motor(l, r, c):
    setLR(l, r)
    if c == "up":
        servo1.angle = 160
        sleep(0.5)
        servo1.angle = 70
        sleep(0.5)
    elif c == "down":
        servo1.angle = 160
        sleep(0.5)
        servo1.angle = 70
        sleep(0.5)
    elif c == "forward":
        servo1.angle = 160
        sleep(0.5)
        servo1.angle = 70
        sleep(0.5)
    elif c == "left":
        servo1.angle = 160
        sleep(0.5)
        servo1.angle = 70
        sleep(0.5)
    elif c == "right":
        servo1.angle = 160
        sleep(0.5)
        servo1.angle = 70
        sleep(0.5)
    reset()
	
def get_mask_and_detection(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in multi_thresholds:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.medianBlur(mask, 7)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=25, maxRadius=150)

    h, w = frame.shape[:2]
    detection = frame.copy()
    cv2.line(detection, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.line(detection, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

    largest_red_circle = None
    largest_radius = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            mask_circle = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(mask_circle, (x, y), r, 255, -1)
            red_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=mask_circle))
            if red_pixels > 0.5 * np.pi * r * r * 0.5:
                if r > largest_radius:
                    largest_red_circle = (x, y, r)
                    largest_radius = r

    if largest_red_circle is not None:
        x, y, r = largest_red_circle
        cv2.circle(detection, (x, y), r, (0, 0, 255), 2)
        cv2.circle(detection, (x, y), 5, (0, 255, 0), -1)
        # Here you can add servo/motor control logic based on (x, y)

    return mask, detection

def gen_original():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_detection():
    while True:
        with ai_tracking_lock:
            if not ai_tracking_active:
                break
        frame = picam2.capture_array()
        mask, detection = get_mask_and_detection(frame)
        ret, buffer = cv2.imencode('.jpg', detection)
        detection_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + detection_bytes + b'\r\n')

def gen_mask():
    while True:
        with ai_tracking_lock:
            if not ai_tracking_active:
                break
        frame = picam2.capture_array()
        mask, detection = get_mask_and_detection(frame)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ret, buffer = cv2.imencode('.jpg', mask_bgr)
        mask_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + mask_bytes + b'\r\n')
               
def record_video():
    global recording_active
    print("Recording thread started.")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
    while True:
        with recording_lock:
            if not recording_active:
                break
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        time.sleep(1.0 / fps)
    out.release()
    print(f"Recording stopped. Video saved as {video_filename}")
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_original')
def video_feed_original():
    return Response(gen_original(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_detection')
def video_feed_detection():
    def stream():
        while True:
            with ai_tracking_lock:
                if not ai_tracking_active:
                    break
            for frame in gen_detection():
                yield frame
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_mask')
def video_feed_mask():
    def stream():
        while True:
            with ai_tracking_lock:
                if not ai_tracking_active:
                    break
            for frame in gen_mask():
                yield frame
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_ai', methods=['POST'])
def toggle_ai():
    global ai_tracking_active
    with ai_tracking_lock:
        ai_tracking_active = not ai_tracking_active
        print("AI Tracking:", ai_tracking_active)
    return jsonify({'ai_tracking_active': ai_tracking_active})

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    command = data.get('command')
    print(f"Received command: {command}")
    # Motor control logic
    if command == 'up':
        motor(-40, 30, 'up')
    elif command == 'down':
        motor(40, -50, 'down')
    elif command == 'left':
        motor(0, -10, 'left')
    elif command == 'right':
        motor(0, 10, 'right')
    elif command == 'forward':
        motor(0, 0, 'forward')
    return jsonify({'status': 'success', 'command': command})


@app.route('/record', methods=['POST'])
def record():
    global recording_active
    with recording_lock:
        if not recording_active:
            recording_active = True
            threading.Thread(target=record_video, daemon=True).start()
            return jsonify({'status': 'started'})
        else:
            recording_active = False
            return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
