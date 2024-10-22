gkf means "GraceKafuu", gkfutils is my personal python package and my study to make a .whl file.


<details open>
<summary>Install</summary>
Pip install the gkfutils package.
  
[![PyPI - Version](https://img.shields.io/pypi/v/gkfutils?logo=pypi&logoColor=white)](https://pypi.org/project/gkfutils/)

```bash
pip install gkfutils
```

<details open>
<summary>Examples</summary>

```bash
import gkfutils

print(gkfutils.__version__)

# 1.生成一个txt文件，内容是路径下文件的绝对路径/相对路径。（路径下不包含子目录）
gkfutils.utils.gen_file_list(data_path="", abspath=True)

# 2.提取视频帧
gkfutils.cv.utils.extract_one_gif_frames(gif_path="")
gkfutils.cv.utils.extract_one_video_frames(video_path="", gap=5)
gkfutils.cv.utils.extract_videos_frames(base_path="", gap=5, save_path="")

# 3.


```
