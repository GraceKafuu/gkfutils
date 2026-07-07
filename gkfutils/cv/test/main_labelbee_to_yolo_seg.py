import os
import json
import shutil
from pathlib import Path

# ====================== 配置项（直接改这里）======================
IMAGE_DIR = r"G:\Gosion\projects\012.JurongDianchang\data\belt_deviate\v3_0\images"          # 原图文件夹
JSON_DIR = r"G:\Gosion\projects\012.JurongDianchang\data\belt_deviate\v3_0\jsons"             # 标签文件夹
OUTPUT_ROOT = r"G:\Gosion\projects\012.JurongDianchang\data\belt_deviate\v3_0_converted"      # 输出根目录
# =================================================================

def labelbee2yolo(json_path, img_w, img_h):
    """
    单个 LabelBee JSON → YOLO 分割标签行
    返回：yolo格式字符串（空则表示无有效标注）
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 读取 step_1 → result 里的 polygon
        if "step_1" not in data or "result" not in data["step_1"]:
            return None

        results = data["step_1"]["result"]
        if not results:
            return None

        yolo_lines = []
        for res in results:
            # 过滤无效标注
            if not res.get("valid", True):
                continue

            point_list = res.get("pointList", [])
            if len(point_list) < 3:  # 多边形至少3个点
                continue

            # 类别 ID（你的 attribute 是 "1"）
            class_id = int(res.get("attribute", "0")) - 1

            # 归一化坐标
            coords = []
            for p in point_list:
                x = p["x"] / img_w
                y = p["y"] / img_h
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")

            # 拼接 YOLO 行
            line = f"{class_id} {' '.join(coords)}"
            yolo_lines.append(line)

        return "\n".join(yolo_lines) if yolo_lines else None

    except Exception as e:
        print(f"解析失败 {json_path}: {e}")
        return None


def convert_folder():
    # 创建输出目录
    out_img_dir = Path(OUTPUT_ROOT) / "images"
    out_lab_dir = Path(OUTPUT_ROOT) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有 JSON
    json_files = list(Path(JSON_DIR).glob("*.json"))
    print(f"找到 {len(json_files)} 个标签文件")

    success = 0
    skip = 0

    for json_file in json_files:
        # 图片名（和 json 同名）
        img_name = json_file.stem
        img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        img_path = None

        for ext in img_exts:
            temp_path = Path(IMAGE_DIR) / (img_name)
            if temp_path.exists():
                img_path = temp_path
                break

        if not img_path:
            print(f"跳过：未找到图片 {img_name}")
            skip += 1
            continue

        # 读取图片宽高
        with open(json_file, encoding='utf-8') as f:
            jdata = json.load(f)
        img_w = jdata["width"]
        img_h = jdata["height"]

        # 转换标签
        yolo_label = labelbee2yolo(json_file, img_w, img_h)

        if not yolo_label:
            print(f"跳过：无有效标注 {json_file.name}")
            skip += 1
            continue

        # ====================== 输出文件 ======================
        # 1. 复制图片
        shutil.copy(img_path, out_img_dir / img_path.name)

        # 2. 保存 TXT 标签
        lab_out_path = out_lab_dir / (os.path.splitext(img_name)[0] + ".txt")
        with open(lab_out_path, "w", encoding="utf-8") as f:
            f.write(yolo_label + "\n")

        success += 1
        print(f"已转换: {json_file.name}")

    print(f"\n✅ 转换完成！成功: {success}  跳过: {skip}")
    print(f"图片输出: {out_img_dir}")
    print(f"标签输出: {out_lab_dir}")


if __name__ == "__main__":
    convert_folder()