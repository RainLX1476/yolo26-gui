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
│  ├─ predict.py
│  ├─ check_dataset.py
│  └─ evaluate_dataset.py
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

当前项目中的四个脚本统一使用 `pip` 安装依赖。

### 核心依赖

- `ultralytics`
- `onnxruntime-directml`
- `opencv-python`
- `numpy`
- `PySide6`

其中：

- `model_convert.py` 依赖 `ultralytics`
- `predict.py` 依赖 `onnxruntime-directml`、`opencv-python`、`numpy`
- `check_dataset.py` 依赖 `opencv-python`、`numpy`
- `evaluate_dataset.py` 依赖 `onnxruntime-directml`、`opencv-python`、`numpy`
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

### 1. `scripts/model_convert.py`

作用：

- 将 YOLO 的 `.pt` 模型导出为 `.onnx`
- 适合把训练结果转换成后续 GUI、推理脚本、评测脚本可直接使用的 ONNX 模型

核心函数：

- `convert_pt_to_onnx(pt_path, onnx_path=None, imgsz=640, opset=12, dynamic=False, simplify=False, half=False)`

命令行用法：

```powershell
py scripts/model_convert.py --pt best.pt --onnx best.onnx
```

常用可选参数：

- `--imgsz 640`：输入尺寸，传 1 个值表示正方形，传 2 个值表示高和宽
- `--opset 12`：指定 ONNX opset 版本
- `--dynamic`：导出动态输入尺寸模型
- `--simplify`：简化导出的 ONNX 图
- `--half`：导出为 FP16 精度

示例：

```powershell
py scripts/model_convert.py --pt best.pt --onnx best.onnx --imgsz 640 --opset 12 --dynamic --simplify
```

命令行输出：

- 终端打印导出的 ONNX 文件路径

### 2. `scripts/predict.py`

作用：

- 使用 ONNX Runtime 对单张图像执行目标检测
- 支持传入图像路径，也支持在函数调用时直接传入 `numpy.ndarray`

核心函数：

- `predict_image(model, image, conf_threshold=0.25)`

相关内部函数：

- `_load_onnx_model(model_path)`：加载 ONNX 模型

命令行用法：

```powershell
py scripts/predict.py --onnx best.onnx --image test.jpg
```

常用可选参数：

- `--conf 0.25`：置信度阈值

示例：

```powershell
py scripts/predict.py --onnx best.onnx --image test.jpg --conf 0.25
```

命令行输出：

- 终端打印 JSON 格式的检测结果列表
- 每个结果包含 `class_id`、`confidence`、`box`

### 3. `scripts/check_dataset.py`

作用：

- 检查 `evaluate/input` 下的 YOLO 格式评测集是否完整、可用
- 统计各类别图片数量、各类别检测框数量、总图片数、总框数、平均每图框数
- 检查图片和标注是否一一对应
- 检查图片是否损坏
- 检查标注类别 ID 是否越界
- 统计小目标比例，小目标定义为标注框 `w < 32` 且 `h < 32` 像素

核心函数：

- `check_dataset(input_dir="evaluate/input")`

函数调用返回：

- 返回结构化字典，包含基础统计结果
- 返回 `histogram_data`，供上层自行绘制“每类图片数”和“每类检测框数”直方图

命令行用法：

```powershell
py scripts/check_dataset.py --input evaluate/input
```

命令行行为：

- 在终端输出完整检查结果文本
- 将结果保存到 `temp/`
- 使用 `opencv` 弹出两张直方图窗口

命令行输出文件：

- `temp/check_dataset_summary.json`
- `temp/check_dataset_report.txt`
- `temp/check_dataset_per_class_images.png`
- `temp/check_dataset_per_class_boxes.png`

### 4. `scripts/evaluate_dataset.py`

作用：

- 使用 ONNX 模型对整个评测集执行检测评测
- 按图像统计预测结果、TP、FP、FN
- 计算整体 `Precision`、`Recall`、`mAP50`、误检率、漏检率
- 输出每个类别的 `AP50 / Precision / Recall`
- 保存评测可视化结果，便于排查误检和漏检

核心函数：

- `evaluate_dataset(model_path, input_dir="evaluate/input", output_dir="evaluate/output", conf_threshold=0.25, iou_threshold=0.5, classes_path=None)`

命令行用法：

```powershell
py scripts/evaluate_dataset.py --onnx best.onnx
```

常用可选参数：

- `--input evaluate/input`：评测集目录
- `--output evaluate/output`：评测输出目录
- `--conf 0.25`：推理置信度阈值
- `--iou 0.5`：TP 判定的 IoU 阈值
- `--classes classes.txt`：可选类别名文件，支持 `classes.txt / names.txt / dataset.yaml`

示例：

```powershell
py scripts/evaluate_dataset.py --onnx best.onnx --input evaluate/input --output evaluate/output --conf 0.25 --iou 0.5
```

命令行输出：

- 终端打印整体评测摘要
- 终端打印各类别 `AP50 / Precision / Recall / GT / Prediction`
- 终端打印误检最多类别和漏检最多类别

评测输出文件：

- `evaluate/output/records/summary.json`
- `evaluate/output/records/per_image_results.json`
- `evaluate/output/visualizations/`：所有图像的可视化结果
- `evaluate/output/errors/`：存在误检或漏检的图像可视化结果

## 开发建议

- GUI 主入口统一从 `main.py` 启动
- 如果后续接入模型，可新增顶层 `core/`、`gui/pages/` 等普通模块目录
- 如果后续需要打包成 `exe`，再补充 PyInstaller 配置即可
- `tests/` 先保留最小结构，后续加功能时可以同步补测试
