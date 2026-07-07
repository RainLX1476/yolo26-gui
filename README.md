# yolo26-gui

26 暑期学校实习成果：基于 YOLO26 的交通标识检测 GUI 项目。

这个仓库目前采用“脚本优先”的轻量结构，适合在 Windows 上直接运行主程序和工具脚本。

## 目录结构

```text
yolo26-gui/
├─ gui/
│  ├─ __init__.py
│  └─ main_window.py
├─ scripts/
│  ├─ model_convert.py
│  └─ predict.py
├─ test_scripts/
│  └─ run_predict_visualize.py
├─ tests/
│  └─ test_app.py
├─ main.py
├─ .gitignore
├─ requirements.txt
├─ requirements-dev.txt
├─ pytest.ini
└─ README.md
```

## 运行方式

推荐使用 Python `3.10+`。

```powershell
py main.py
```

如果你希望先安装依赖，再运行主程序：

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py main.py
```

## 依赖安装

当前项目中的模型转换与单张图像推理脚本统一使用 `pip` 安装依赖。

### 核心依赖

- `ultralytics`
- `onnxruntime-directml`
- `opencv-python`
- `numpy`
- `PySide6`

其中：

- `model_convert.py` 依赖 `ultralytics`
- `predict.py` 依赖 `onnxruntime-directml`、`opencv-python`、`numpy`
- `onnxruntime-directml` 安装后，代码中的导入名仍然是 `onnxruntime`

### 安装命令

推荐先升级 `pip`，再安装运行依赖：

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

如果还需要运行测试：

```powershell
py -m pip install -r requirements-dev.txt
```

## 脚本说明

### 1. PT 转 ONNX

脚本路径：`scripts/model_convert.py`

脚本内部负责命令行参数解析，对外核心函数为 `convert_pt_to_onnx(...)`。

示例：

```powershell
py scripts/model_convert.py --pt best.pt --onnx best.onnx
```

可选参数示例：

```powershell
py scripts/model_convert.py --pt best.pt --onnx best.onnx --imgsz 640 --opset 12 --dynamic --simplify
```

### 2. ONNX 单张图像推理

脚本路径：`scripts/predict.py`

该脚本默认使用 `ONNX Runtime DirectML` 后端，适合 Windows 环境下调用 ONNX 模型进行单张图像检测。

脚本当前只保留尽可能简单的能力：

- 脚本内部加载 ONNX 模型
- 对外核心函数只负责接收“模型句柄 + 图像”
- 输出一个或多个检测结果

示例：

```powershell
py scripts/predict.py --onnx best.onnx --image test.jpg
```

可选参数示例：

```powershell
py scripts/predict.py --onnx best.onnx --image test.jpg --conf 0.25
```

## 开发建议

- GUI 主入口统一从 `main.py` 启动
- 如果后续接入模型，可新增顶层 `core/`、`gui/pages/` 等普通模块目录
- 如果后续需要打包成 `exe`，再补充 PyInstaller 配置即可
- `tests/` 先保留最小结构，后续加功能时可以同步补测试
