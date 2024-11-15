import setuptools

# 遇到的问题：
# 1.加上encoding="utf-8"后，打包的whl上传至pypi失败。
# 2.假如不加encoding="utf-8"，则打包不成功。所以直接去除了long_description。

# with open("README.md", "r", encoding="utf-8") as fh:
# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="gkfutils",
    version="1.1.6",
    author="GraceKafuu",
    author_email="gracekafuu@gmail.com",
    description="GraceKafuu utils",
    # long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
