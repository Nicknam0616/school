import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import uic
import openai
from deep_translator import GoogleTranslator
import cv2
from PyQt5.QtGui import QImage, QPixmap
from flask import Flask, render_template
from flask_socketio import SocketIO, send
import socketio
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio_server = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@socketio_server.on('message')
def handleMessage(msg):
    print('Message:', msg['msg'])
    send({'user': msg['user'], 'msg': msg['msg']}, broadcast=True)

form_class = uic.loadUiType("real.ui")[0]

class AIThread(QThread):
    response_signal = pyqtSignal(str, str)

    def __init__(self, user_input, client, translator, parent=None):
        super().__init__(parent)
        self.user_input = user_input
        self.client = client
        self.translator = translator

    def run(self):
        translated_input = self.translator.translate(self.user_input)
        completion = self.client.chat.completions.create(
            model="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF/llama-3.2-3b-instruct-q8_0.gguf",
            messages=[
                {"role": "system", "content": "You're an expert in strategy and tactics, help me ..."},
                {"role": "user", "content": translated_input}
            ],
            temperature=0.7,
        )
        translated_output = GoogleTranslator(source='en', target='ko').translate(completion.choices[0].message.content)
        self.response_signal.emit(self.user_input, translated_output)

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_line_signal = pyqtSignal(str)

    def __init__(self, mode="basic"):
        super().__init__()
        self.mode = mode
        self.running = True
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb_image = self.process_frame(frame)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)

                if self.mode == "red_only":
                    self.update_line_signal.emit("반자율 주행 중")

    def process_frame(self, frame):
        if self.mode == "basic":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.mode == "black_white":
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        elif self.mode == "red_only":
            return self.red_only_detection(frame)
        elif self.mode == "face_tracking":
            return self.face_tracking(frame)
        return frame

    def red_only_detection(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 90])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        result = cv2.bitwise_and(frame, frame, mask=mask)

        frame_center = frame.shape[1] // 2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                if cx < frame_center - 10:
                    self.update_line_signal.emit("Move Right")
                elif cx > frame_center + 10:
                    self.update_line_signal.emit("Move Left")
                else:
                    self.update_line_signal.emit("Move Forward")

        return result

    def face_tracking(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (int(bbox.xmin * iw), int(bbox.ymin * ih),
                              int(bbox.width * iw), int(bbox.height * ih))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def change_mode(self, mode):
        self.mode = mode

    def stop(self):
        self.running = False

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.translator = GoogleTranslator(source='ko', target='en')
        self.autonomous_mode = False

        self.gpt_button.clicked.connect(self.start_ai_thread)
        self.chat_b.clicked.connect(self.send_message)
        self.gpt_real.setReadOnly(True)
        self.under1.setReadOnly(True)
        self.chat.setReadOnly(True)
        self.line.setReadOnly(True)

        self.speed_set.setReadOnly(True)
        self.speed_real.setReadOnly(True)
        self.coordinate.setReadOnly(True)

        self.camera_thread = CameraThread("basic")
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.update_line_signal.connect(self.update_line)
        self.camera_thread.start()

        self.line.append("반자율 주행 OFF")

        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setSingleStep(1)
        self.speed_slider.setPageStep(10)
        self.speed_slider.setValue(0)
        self.speed_slider.setTracking(True)
        self.speed_slider.valueChanged.connect(self.update_speed_set)

        self.b_c.clicked.connect(lambda: self.change_camera_mode("basic"))
        self.b_d.clicked.connect(lambda: self.change_camera_mode("black_white"))
        self.b_r.clicked.connect(self.red_only_mode)
        self.b_f.clicked.connect(lambda: self.change_camera_mode("face_tracking"))
        self.b_r_2.clicked.connect(self.toggle_autonomous_mode)

        self.socket = socketio.Client()
        self.socket.connect('http://10.244.85.175:5000')
        self.socket.on('message', self.receive_message)

    def start_ai_thread(self):
        user_input = self.gpt_input.text()
        if user_input:
            self.ai_thread = AIThread(user_input, self.client, self.translator)
            self.ai_thread.response_signal.connect(self.update_chat)
            self.ai_thread.start()

    def send_message(self):
        message = self.chat_i.text()
        if message:
            self.socket.emit('message', {'user': 'ui', 'msg': message})
            self.chat_i.clear()

    def receive_message(self, data):
        self.chat.append(f"{data['user']}: {data['msg']}")

    def update_chat(self, user_input, translated_output):
        self.gpt_real.append(f"User: {user_input}")
        self.gpt_real.append(f"AI: {translated_output}")
        self.gpt_input.clear()

    def update_image(self, qt_image):
        self.main.setPixmap(QPixmap.fromImage(qt_image))

    def update_line(self, text):
        self.line.append(text)

    def change_camera_mode(self, mode):
        self.camera_thread.change_mode(mode)
        self.line.append("반자율 주행 OFF")

    def red_only_mode(self):
        self.camera_thread.change_mode("red_only")
        self.line.append("반자율 주행 OFF")

    def toggle_autonomous_mode(self):
        self.autonomous_mode = not self.autonomous_mode
        if self.autonomous_mode:
            self.line.setText("자율주행 ON")
        else:
            self.line.setText("자율주행 OFF")

    def update_speed_set(self):
        speed_value = self.speed_slider.value()
        self.speed_set.setText(f"Speed Set: {speed_value}")

    def update_real_speed(self, real_speed):
        self.speed_real.setText(f"Actual Speed: {real_speed}")

        def update_coordinate(self, latitude, longitude):
            self.coordinate.setText(f"Coordinate: {latitude}, {longitude}")

if __name__ == "__main__":
    import threading

    flask_thread = threading.Thread(
        target=lambda: socketio_server.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True))
    flask_thread.start()

    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()