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
gkfutils.utils.gen_data_txt_list(data_path="data_path", one_dir_flag=True)  # Will generate a .txt file that contains file_abs_path of files in data_path.

```
