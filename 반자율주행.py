import PySimpleGUI as sg
import cv2
import requests
import numpy as np
import keyboard

# ESP32-CAM 주소
url = "http://10.244.230.218/capture"

# GUI 레이아웃 설정
layout = [
    [sg.Text('카메라 시점', size=(40, 1), justification='center', font=('Helvetica', 20), text_color='white')],
    [sg.Image(filename='', key='image')],
    [sg.Text('속도', size=(15, 1), text_color='white'), sg.Text('', key='speed', size=(15, 1), text_color='white'),
     sg.Text('방향', size=(15, 1), text_color='white'), sg.Text('', key='way', size=(15, 1), text_color='white')],
    [sg.Text('내부 온도', size=(15, 1), text_color='white'), sg.Text('', key='kal', size=(15, 1), text_color='white'),
     sg.Text('남은 탄환', size=(15, 1), text_color='white'),
     sg.Text('', key='fire_status', size=(15, 1), text_color='white')],
    [sg.Button('발사', size=(15, 1), key='fire_button', button_color=('white', 'red'))],
    [sg.Image(filename='', key='mask_image', size=(320, 240))],
    [sg.Image(filename='', key='result_image', size=(320, 240))]
]

# 창 설정
window = sg.Window('RC 탱크카 제어', layout, location=(800, 400), background_color='black', return_keyboard_events=True, resizable=True)

while True:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED:
        break
    if event == 'fire_button':
        break

    # ESP32-CAM에서 이미지 가져오기
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    if frame is None:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR 색상 공간을 HSV 색상 공간으로 변환합니다.
    lower_red = np.array([0, 120, 90])  # 빨간색의 하한값을 설정합니다.
    upper_red = np.array([10, 255, 255])  # 빨간색의 상한값을 설정합니다.
    mask1 = cv2.inRange(hsv, lower_red, upper_red)  # 첫 번째 빨간색 범위에 대한 마스크를 생성합니다.

    lower_red = np.array([170, 120, 70])  # 두 번째 빨간색의 하한값을 설정합니다.
    upper_red = np.array([180, 255, 255])  # 두 번째 빨간색의 상한값을 설정합니다.
    mask2 = cv2.inRange(hsv, lower_red, upper_red)  # 두 번째 빨간색 범위에 대한 마스크를 생성합니다.

    mask = mask1 + mask2  # 두 마스크를 합칩니다.
    result = cv2.bitwise_and(frame, frame, mask=mask)  # 원본 프레임과 마스크를 사용하여 결과 이미지를 생성합니다.

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 마스크에서 윤곽선을 찾습니다.
    for contour in contours:
        M = cv2.moments(contour)  # 윤곽선의 모멘트를 계산합니다.
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # 윤곽선의 중심 x 좌표를 계산합니다.
            cy = int(M['m01'] / M['m00'])  # 윤곽선의 중심 y 좌표를 계산합니다.
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # 중심에 원을 그립니다.

            # 화면의 중심과 비교하여 이동 명령 생성
            frame_center = frame.shape[1] // 2
            if cx < frame_center - 10:
                print("Move Right")  # 중심이 왼쪽에 있으면 오른쪽으로 이동합니다.
            elif cx > frame_center + 10:
                print("Move Left")  # 중심이 오른쪽에 있으면 왼쪽으로 이동합니다.
            else:
                print("Move Forward")  # 중심에 있으면 앞으로 이동합니다.

    # 이미지 업데이트
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['image'].update(data=imgbytes)

    maskbytes = cv2.imencode('.png', mask)[1].tobytes()
    window['mask_image'].update(data=maskbytes)

    resultbytes = cv2.imencode('.png', result)[1].tobytes()
    window['result_image'].update(data=resultbytes)

    # 키보드 입력 처리
    if keyboard.is_pressed('w'):
        window['way'].update('w')  # 'w' 키를 누르면 way 값에 'w'를 표시합니다.
    if keyboard.is_pressed('s'):
        window['way'].update('s')  # 's' 키를 누르면 way 값에 's'를 표시합니다.

    # 여기서 방향, 속도, 발사 가능 여부를 업데이트합니다.
    window['speed'].update('Null')
    window['kal'].update('Null')

window.close()
