import serial
import time
from geopy.distance import geodesic

# 시리얼 포트와 통신 속도 설정
ser = serial.Serial('COM6', 9600, timeout=1)
time.sleep(2)  # 시리얼 포트 안정화 대기


def convert_to_decimal(degrees, direction):
    deg = float(degrees[:2])
    minutes = float(degrees[2:])
    decimal = deg + (minutes / 60.0)
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal


def get_current_location(timeout=10):
    start_time = time.time()
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("$GPGLL"):
                parts = line.split(',')
                if parts[6] == 'A':  # 데이터 유효 여부 확인
                    lat = convert_to_decimal(parts[1], parts[2])
                    lon = convert_to_decimal(parts[3], parts[4])
                    return (lat, lon)
        if time.time() - start_time > timeout:
            print("GPS 데이터를 수신할 수 없습니다.")
            return None


current_location = get_current_location()
if current_location:
    print("현재 위치: ", current_location)
else:
    print("프로그램을 종료합니다.")
    exit()

while True:
    target_location = input("목표 위치 (위도, 경도) 입력하세요 : ")
    target_lat, target_lon = map(float, target_location.split(','))
    target_coords = (target_lat, target_lon)

    while True:
        current_coords = get_current_location()
        if current_coords is None:
            break

        distance = geodesic(current_coords, target_coords).meters

        print(f"남은 거리: {distance:.2f} 미터")

        if distance <= 1.0:  # 1미터 이내 도착으로 간주
            print("도착했습니다!")
            break

        time.sleep(1)
