import serial
import time

# 아두이노와 연결된 시리얼 포트 설정
ser = serial.Serial('COM10', 115200)  # 'COM7'을 송신 포트로 변경

def send_data(data):
    ser.write(data.encode())

while True:
    user_input = input("1: LED ON, 2: LED OFF, q: Quit\n")
    if user_input == '1':
        send_data('1')
    elif user_input == '2':
        send_data('2')
    elif user_input == 'q':
        break
    else:
        print("Invalid input. Please enter 1, 2, or q.")
    time.sleep(1)

ser.close()
