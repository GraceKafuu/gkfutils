import json
import os
from dashscope import MultiModalConversation
import dashscope
import base64
import mimetypes
import requests
import numpy as np
import time



def main_url():
    # 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    # 模型支持输入1-3张图片
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/thtclx/input1.png"},
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/iclsnx/input2.png"},
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/gborgw/input3.png"},
                {"text": "图1中的女生穿着图2中的黑色裙子按图3的姿势坐下"}
            ]
        }
    ]

    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
    api_key = ""

    # qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
    response = MultiModalConversation.call(
        api_key=api_key,
        model="qwen-image-edit-plus",
        messages=messages,
        stream=False,
        n=2,
        watermark=False,
        negative_prompt="低质量",
        prompt_extend=True,
        # 仅当输出图像数量n=1时支持设置size参数，否则会报错
        # size="1024*2048",
    )

    if response.status_code == 200:
        # 如需查看完整响应，请取消下行注释
        # print(json.dumps(response, ensure_ascii=False))
        for i, content in enumerate(response.output.choices[0].message.content):
            print(f"输出图像{i+1}的URL:{content['image']}")
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")
        
        
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
    
            
def main_base64():
    # 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    # ---用于 Base64 编码 ---
    # 格式为 data:{mime_type};base64,{base64_data}

    # 获取图像的 Base64 编码
    # 调用编码函数，请将 "/path/to/your/image.png" 替换为您的本地图片文件路径，否则无法运行
    # image = encode_file(r"G:\Gosion\projects\009.TuoGun_Det\images\bbx\20251220_144110_output_000000002.jpg")

    d = "20251221_merged"
    data_path = r"G:\Gosion\projects\009.TuoGun_Det\videos\frames\{}".format(d)
    file_list = sorted(os.listdir(data_path))

    save_path = "output/qwen-image-edit-plus/20250109_{}".format(d)
    os.makedirs(save_path, exist_ok=True)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = os.path.join(data_path, f)

        image = encode_file(f_abs_path)
        # image = encode_file(r"G:\Gosion\projects\009.TuoGun_Det\videos\frames\20251220\10.58.136.147_552\20251220_152615\20251220_152615_output_000000003.jpg")

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image},
                    {"text": "随机选择一个图中工业输送皮带下方的黑色圆柱物体脱落掉在地面上（注意：原来在皮带下方的黑色圆柱物体需要擦除。）"}
                ]
            }
        ]

        # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        # 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
        api_key = ""

        # qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
        response = MultiModalConversation.call(
            api_key=api_key,
            model="qwen-image-edit-plus",
            messages=messages,
            stream=False,
            n=6,
            watermark=False,
            negative_prompt="低质量，低分辨率，低画质，画面过饱和，无细节，过度光滑，画面具有AI感。构图混乱。物体扭曲。",
            prompt_extend=True,
            # 仅当输出图像数量n=1时支持设置size参数，否则会报错
            # size="2048*1024",
        )
        
        
        if response.status_code == 200:
            # 如需查看完整响应，请取消下行注释
            # print(json.dumps(response, ensure_ascii=False))
            for i, content in enumerate(response.output.choices[0].message.content):
                print(f"输出图像{i+1}的URL:{content['image']}")
                main_url_download(content['image'], save_path, fname, i)
        else:
            print(f"HTTP返回码：{response.status_code}")
            print(f"错误码：{response.code}")
            print(f"错误信息：{response.message}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")


        # time.sleep(.1)

    
if __name__ == '__main__':
    pass
    # main_url()
    # main_base64()
    # main_url_download()