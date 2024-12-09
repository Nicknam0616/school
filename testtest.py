import PySimpleGUI as sg
import cv2

# 카메라 설정
cap = cv2.VideoCapture(0)

# GUI 레이아웃 설정
layout = [
    [sg.Text('카메라 시점', size=(40, 1), justification='center', font=('Helvetica', 20), text_color='white')],
    [sg.Image(filename='', key='image')],
    [sg.Text('속도', size=(15, 1), text_color='white'), sg.Text('', key='speed', size=(15, 1), text_color='white'), sg.Text('방향', size=(15, 1), text_color='white'), sg.Text('', key='way', size=(15, 1), text_color='white')],
    [sg.Text('내부 온도', size=(15, 1), text_color='white'), sg.Text('', key='kal', size=(15, 1), text_color='white'), sg.Text('발사 가능 여부', size=(15, 1), text_color='white'), sg.Text('', key='fire_status', size=(15, 1), text_color='white')],
    [sg.Button('발사', size=(15, 1), key='fire_button', button_color=('white', 'red'))]
]

# 창 설정
window = sg.Window('RC 탱크카 제어', layout, location=(800, 400), background_color='black')

while True:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED:
        break
    if event == 'fire_button':
        break
    # 카메라 프레임 읽기
    ret, frame = cap.read()
    if ret:
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

    # 여기서 방향, 속도, 발사 가능 여부를 업데이트합니다.
    window['way'].update('Null')
    window['speed'].update('Null')
    window['kal'].update('Null')
    window['fire_status'].update('Null')

cap.release()
window.close()



