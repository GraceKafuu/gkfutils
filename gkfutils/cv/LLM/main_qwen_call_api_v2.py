import json
import os
from dashscope import MultiModalConversation
import base64
import mimetypes
import dashscope
import requests
import numpy as np
import time


# ---用于 Base64 编码 ---
# 格式为 data:{mime_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")


def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")
    

def download_image(image_url, save_path='output.png'):
    try:
        response = requests.get(image_url, stream=True, timeout=300)  # 设置超时
        response.raise_for_status()  # 如果HTTP状态码不是200，则引发异常
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"图像已成功下载到: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"图像下载失败: {e}")


def main_url_download(image_url, savePath, fname, i):
    # image_url = "https://dashscope-result-hz.oss-cn-hangzhou.aliyuncs.com/7d/bb/20260108/31eda923/26c478bc-1ad9-4af7-875e-08e54d379df6-1.png?Expires=1768460008&OSSAccessKeyId=LTAI5tKPD3TMqf2Lna1fASuh&Signature=t1bFV2znkQx2rKIF1RuSX9PFpOs%3D"
    rdm_num = str(np.random.random()).replace("-", "").replace(".", "")
    download_image(image_url, save_path='{}/{}_{}_{}.png'.format(savePath, fname, i, rdm_num))


def main():
    # 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    # 获取图像的 Base64 编码
    # 调用编码函数，请将 "/path/to/your/image.png" 替换为您的本地图片文件路径，否则无法运行
    
    data_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\Qwen_Image_Edit\20260305\images"
    file_list = sorted(os.listdir(data_path))

    save_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\Qwen_Image_Edit\20260305\images_qwen-image-edit-max-2026-01-16_output"
    os.makedirs(save_path, exist_ok=True)
    
    img_path1 = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\XianChang\TuogunChaixie\2ucrz2f7svoi2x4ou1m9wme8q_0.jpg"
    img_path2 = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\XianChang\TuogunChaixie\mmexport1772515403009.jpg"
    img_path3 = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\XianChang\TuogunChaixie\mmexport1772515404546.jpg"
    img_path4 = r"G:\Gosion\projects\009.TuoGun_Det\resources\videos\BrowserRecording\20260304\Small_frames\1772515946702_youdiaoluo\1772515946702_output_000000027.jpg"
    image_drop1 = encode_file(img_path1)
    image_drop2 = encode_file(img_path2)
    image_drop3 = encode_file(img_path3)
    image_drop4 = encode_file(img_path4)

    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
    api_key = ""

    for f in file_list[90:]:
        fname = os.path.splitext(f)[0]
        f_abs_path = os.path.join(data_path, f)

        image = encode_file(f_abs_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image},
                    {"image": image_drop1},
                    {"image": image_drop4},
                    {"text": "按照图2,图3的方式模拟将图1的托辊掉落至皮带面上，也就是将图1中竖的和斜的黑黑的物体模拟掉落在皮带面上，要求生成后的图毫无违和感，生成部位物体模糊化，物体清晰度及整体感觉和图1差不多。要求是更改图1, 生成更改之后的图。"}
                ]
            }
        ]

        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"image": image},
        #             {"image": image_drop1},
        #             {"image": image_drop2},
        #             {"text": "参考图2和图3编辑图1，要求按照图2和图3的方式模拟将图1的托辊掉落至皮带面上，要求编辑后图1毫无违和感，保持图1原有的清晰度。"}
        #         ]
        #     }
        # ]

        # qwen-image-2.0系列、qwen-image-edit-max、qwen-image-edit-plus系列支持输出1-6张图片，此处以2张为例
        response = MultiModalConversation.call(
            api_key=api_key,
            model="qwen-image-2.0-pro",
            messages=messages,
            stream=False,
            n=1,
            watermark=False,
            negative_prompt=" ",
            prompt_extend=True,
            size="1536*1024",
        )

        if response.status_code == 200:
            # 如需查看完整响应，请取消下行注释
            # print(json.dumps(response, ensure_ascii=False))
            # for i, content in enumerate(response.output.choices[0].message.content):
            #     print(f"输出图像{i+1}的URL:{content['image']}")
            for i, content in enumerate(response.output.choices[0].message.content):
                print(f"输出图像{i+1}的URL:{content['image']}")
                main_url_download(content['image'], save_path, fname, i)
        else:
            print(f"HTTP返回码：{response.status_code}")
            print(f"错误码：{response.code}")
            print(f"错误信息：{response.message}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")

        
if __name__ == '__main__':
    main()