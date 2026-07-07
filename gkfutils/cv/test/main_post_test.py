import os
import cv2
import base64
import requests
import json
import numpy as np


def main():
    # p_httpPushUrl = "http://10.58.42.29:48080/admin-api/iot/err-record/imageAlertAlarm"
    p_httpPushUrl = "http://10.10.11.222:8080/v1/guo_xun/cv_belt_tear/postDetection"
    # p_httpPushUrl = "http://10.10.11.222:8080/v1/guo_xun/cv_belt_tear/postConfig"
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

    # data_str = {
    #     "processedImage": img_base64,
    #     "beforeImage": imgNoBbxBase64, 
    #     "location": "121",
    #     "robotIP": "10.58.128.30",
    #     # "alarmBbox": str(bbx),
    #     # "alarmTime": alarm_time
    # }

    data_str = {
        "imgDet": img_base64
    }

    # 国巡GINS
    headers = {'Content-Type': 'application/json', "Accept": "*/*"}
    res = requests.post(p_httpPushUrl, data=json.dumps(data_str), headers=headers)

    # 新疆托辊掉落检测
    # headers = {'Content-Type': 'application/x-www-form-urlencoded', "Accept": "*/*"}
    # res = requests.post(p_httpPushUrl, data=data_str, headers=headers)

    # try:
    #     res = requests.post(p_httpPushUrl, json=json.dumps(data_str), headers=headers, timeout=10)
    #     print(f"状态码: {res.status_code}")
    #     print(f"响应内容: {res.text}")
    # except requests.exceptions.RequestException as e:
    #     print(f"请求异常: {e}")

    print(data_str)
    print("=============================")
    print(res)


def main2():
    p_httpPushUrl = "http://10.10.11.225:8080/v1/guo_xun/cv_belt_tear/postDetection"
    # p_httpPushUrl = "http://10.10.11.222:8080/v1/guo_xun/cv_belt_tear/postConfig"
    # img_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\Test\A\ScreenShot_2025-12-31_103804_037.png"
    img_path = r"G:\Gosion\projects\000.Test_Data\images\zidane.jpg"
    # img_path = r"G:\Gosion\projects\000.Test_Data\images\bus.jpg"
    # img_path = r"G:\Gosion\projects\010.BeltTornDetectionR3\data\test\_home_sy_MVviewer_pictures_Pic_2026_03_06_141328_6_2.jpg"
    img1 = cv2.imread(img_path)

    _, buffer = cv2.imencode(".jpg", img1)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    data_str = {
        # "jsonObject": {"imgDet": img_base64, "videoUrl": "http://10.10.11.29:8091/gx_resourceUrl/image/upload/system_4701bbf6-c3d8-42c3-9cf7-814fbc501fc3.mp4"}
        "jsonObject": {"imgDet": img_base64, "videoUrl": "/home/orangepi/wujiahu/data/17751080350572039574937311850498.mp4"}
        # "jsonObject": {"imgDet": img_base64}
    }

    # 国巡GINS
    headers = {'Content-Type': 'application/json', "Accept": "*/*"}
    res = requests.post(p_httpPushUrl, data=json.dumps(data_str), headers=headers)

    # print(data_str)
    print(res)
    jsonRes = res._content.decode().replace("false", "False").replace("true", "True")
    jsonRes = eval(jsonRes)
    print(jsonRes)
    showImg = jsonRes["param"]["showImg"]
    showImg = cv2.imdecode(np.frombuffer(base64.b64decode(showImg), np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("res", showImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main3():
    p_httpPushUrl = "http://10.10.11.222:8080/v1/guo_xun/cv_belt_tear/postConfig"

    config_path = r"G:\Gosion\projects\010.BeltTornDetectionR3\code\Latest\BeltTearDetection_CPP\config\config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 👉 直接转成 Python 字典/列表

    # 国巡GINS
    headers = {'Content-Type': 'application/json', "Accept": "*/*"}
    res = requests.post(p_httpPushUrl, data=json.dumps(data), headers=headers)
    

    print(data)
    print(res)

if __name__ == '__main__':
    main2()
    # main3()

