import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic

class NewWindowClass(QMainWindow):  # QDialog 대신 QMainWindow로 변경
    def __init__(self):
        super().__init__()
        uic.loadUi("set.ui", self)  # set.ui 파일 로드

        # t1: 배경용 QTextEdit
        self.t1.setReadOnly(True)
        self.t1.setStyleSheet("background-color: lightgray;")

        # t2: 패킷 로그용 QTextEdit
        self.t2.setReadOnly(True)
        self.t2.setStyleSheet("background-color: white;")

        # t3: 아두이노 이름 표시용 QTextEdit
        self.t3.setReadOnly(True)
        self.t3.setStyleSheet("background-color: lightyellow;")

        # t4: 아두이노 이름 입력용 QTextEdit
        self.t4.setPlaceholderText("새 아두이노 이름 입력...")

        # b1: 아두이노 이름 저장 버튼
        self.b1.clicked.connect(self.save_arduino_name)

    def show_arduino_name(self, name):
        self.t3.setText(name)  # 아두이노 이름 표시

    def save_arduino_name(self):
        arduino_name = self.t4.toPlainText().strip()  # t4에서 입력한 텍스트 가져오기
        if arduino_name:
            self.show_arduino_name(arduino_name)  # t3에 아두이노 이름 표시
            self.t4.clear()  # t4 비우기
            print(f"Arduino name changed to: {arduino_name}")  # 콘솔에 출력
        else:
            print("No Arduino name entered.")  # 예시: 콘솔에 출력

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NewWindowClass()  # NewWindowClass 인스턴스 생성
    window.show()  # 창 표시
    sys.exit(app.exec_())  # 이벤트 루프 시작
