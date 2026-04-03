import os
import cv2
import base64
import requests
import json


def main():
    p_httpPushUrl = "http://10.58.42.29:48080/admin-api/iot/err-record/imageAlertAlarm"
    img_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\Test\A\ScreenShot_2025-12-31_103804_037.png"
    img2_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\Test\A\ScreenShot_2025-12-31_103804_037.png"
    img1 = cv2.imread(img_path)
    img2 = cv2.imread(img2_path)

    _, buffer = cv2.imencode(".jpg", img1)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    _, imgNoBbx = cv2.imencode(".jpg", img2)
    imgNoBbxBytes = imgNoBbx.tobytes()
    imgNoBbxBase64 = base64.b64encode(imgNoBbxBytes).decode('utf-8')

    data_str = {
        "processedImage": img_base64,
        "beforeImage": imgNoBbxBase64, 
        "location": "121",
        "robotIP": "10.58.128.30",
        # "alarmBbox": str(bbx),
        # "alarmTime": alarm_time
    }

    # headers = {'Content-Type': 'application/json', "Accept": "*/*"}
    # res = requests.post(p_httpPushUrl, data=json.dumps(data_str), headers=headers)
    headers = {'Content-Type': 'application/x-www-form-urlencoded', "Accept": "*/*"}
    res = requests.post(p_httpPushUrl, data=data_str, headers=headers)

    # try:
    #     res = requests.post(p_httpPushUrl, json=json.dumps(data_str), headers=headers, timeout=10)
    #     print(f"状态码: {res.status_code}")
    #     print(f"响应内容: {res.text}")
    # except requests.exceptions.RequestException as e:
    #     print(f"请求异常: {e}")

    print(data_str)
    print("=============================")
    print(res)


if __name__ == '__main__':
    main()

