import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import uic
import openai
from deep_translator import GoogleTranslator
import cv2
from PyQt5.QtGui import QImage, QPixmap
import socketio

# UI 파일 연결
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

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # OpenAI와 번역기 설정
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.translator = GoogleTranslator(source='ko', target='en')

        # 버튼과 입력창 설정
        self.gpt_button.clicked.connect(self.start_ai_thread)
        self.chat_b.clicked.connect(self.send_message)
        self.gpt_real.setReadOnly(True)
        self.under1.setReadOnly(True)
        self.chat.setReadOnly(True)

        # 카메라 스레드 시작
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.start()

        # 소켓 설정
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
            self.socket.emit('message', message)  # 딕셔너리가 아닌 문자열로 보냄
            self.chat.append(f"ui: {message}")
            self.chat_i.clear()

    def receive_message(self, data):
        self.chat.append(f"{data['user']}: {data['msg']}")  # data는 딕셔너리임

    def update_chat(self, user_input, translated_output):
        self.gpt_real.append(f"User: {user_input}")
        self.gpt_real.append(f"AI: {translated_output}")
        self.gpt_input.clear()

    def update_image(self, qt_image):
        self.main.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == "__main__":
    import threading
    flask_thread = threading.Thread(target=lambda: socketio.Server.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True))
    flask_thread.start()

    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()
    # 프로그램 화면을 보여주는 코드
    myWindow.show()
    # 프로그램을 이벤트 루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
