from setuptools import setup, find_packages

setup(
    name="TranSQ",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.0.0",
    license="MIT",
    description="TranSQ: Transformer-based Semantic Query for Medical Report Generation",
    author="Kong Ming",
    author_email="zjukongming@zju.edu.cn",
    url="https://github.com/zjukongming/TranSQ",
    keywords=["medical report generation"],
    install_requires=["torch", "pytorch_lightning"],
)
