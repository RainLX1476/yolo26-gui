# 1.项目简介
随着智能交通系统和自动驾驶技术的快速发展，道路交通信号标志的准确识别已成为保障行车安全的关键技术之一 —— 它不仅是车辆感知周边交通环境的核心环节，也是后续决策控制模块的重要输入。本项目基于 Ultralytics 最新的 YOLO26 目标检测框架，构建了一套端到端的实时交通标志牌识别系统，能够精准捕获不同场景、不同光照条件下的交通标识，为自动驾驶安全辅助驾驶、交通违章抓拍、自适应信号控制等应用提供技术支撑

相比前代 YOLO 系列模型，YOLO26 的技术优势恰好匹配交通标识识别的特殊需求 —— 其轻量化的检测头设计，能在不牺牲推理速度的前提下强化对交通标识这类中小目标的特征提取能力；原生端到端无 NMS 的架构优化，更是直接简化了部署后的阈值调优流程，进一步降低了系统的部署成本。
![img_1.png](img_1.png)

**场景适配价值**：通过 TT100K 公开数据集加测定制式的评测方案，系统能有效覆盖不同天气、不同光照强度、不同拍摄角度和观测距离下的标识识别需求，确保在真实场景下的性能适配。



# 2.环境配置
1. ***安装nvidia显卡驱动*** 

首先要在设备管理器中查看你的显卡型号，在这里可以看到显卡型号为RTX 3060。
![img_2.png](img_2.png)

NVIDIA 驱动下载：https://www.nvidia.cn/Download/index.aspx?lang=c n
下载对应你的英伟达显卡驱动。

下载之后就是简单的下一步执行直到完成。然后，在cmd中输入执行：nvidia-smi

如果输出右图所示的显卡信息，说明你的驱动安装成功。
注：图中的 CUDA Version是当前Driver版本能支持的最高的CUDA版本
![img_3.png](img_3.png)

2. ***安装cuda和cudnn***

CUDA用的是11.8版本 cuda下载链接: https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
下载后得到文件：cuda_11.8.0_522.06_windows.exe 执行该文件进行安装。

cudnn下载地址：https://developer.nvidia.com/cudnn 。
注意：cudnn版本要和cuda版本匹配。下载后得到文件：cudnn-windows-x86_64-8.9.0.131_cuda11-archive.zip

安装cuDNN ：复制cudnn文件 对于cudnn直接将其解开压缩包，然后需要将bin,include,lib中的文件复制粘贴到cuda的文件夹下C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
注意：对整个文件夹bin,include,lib中内容选中后进行复制粘贴

 CUDA安装测试 ：最后测试cuda是否配置成功，打开cmd执行：nvcc -V 即可看到cuda的信息
![img_4.png](img_4.png)

3. ***安装gpu版torch、torchvision***

安装pytorch： 
创建环境名为yolo26的虚拟环境:conda create -n yolo26 python=3.10 -y

安装成功后激活yolo26环境： conda activate yolo26

在所创建的yolo环境下安装pytorch和torchvision,  执行命令：
GPU版（确保支持的最大cuda版本大于11.7）：

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

CPU版本： pip install torch==1.13.1 torchvision==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

在yolo26环境中执行：
import torch
torch.cuda.is_available() 
返回True，说明GPU版本的pytorch安装好了，返回False，说明CPU版本的pytorch安装好了，报错说明没有安装好

4. ***下载YOLO26并安装***

安装Git软件， git下载地址：https://git-scm.com/downloads

克隆项目到本地（如d:） 项目网址:  https://github.com/ultralytics/ultralytics

在 Git CMD窗口中执行:git clone https://github.com/ultralytics/ultralytics
注意：优先使用老师提供的代码

在yolo26虚拟环境下执行以下命令安装YOLO26：
conda activate yolo26
pip install ultralytics



# 3.数据集准备
本项目采用 TT100K（Tsinghua-Tencent 100K）作为模型训练的基础数据集 —— 这是由清华 - 腾讯联合实验室发布的公开交通标志检测数据集，也是当前行业内公认的交通标识识别训练基准数据集之一。该数据集的原始采集场景为国内真实城市道路和高速公路，数据集内的图像覆盖了白天、黑夜、雨天、雾天等不同 illuminance（光照强度）、天气条件下的交通标识，同时包含了不同拍摄角度、不同观测距离下的标识样本 —— 这恰好匹配了本项目需要支持的多种真实应用场景的需求。

由于官方 TT100K 数据集的标注格式为 COCO JSON 格式，而 YOLO26 模型的训练要求输入为 YOLO TXT 格式的标注文件，因此在启动训练前，必须先对数据集的格式进行转换；同时，为了验证模型的泛化能力，项目额外制作了一个独立的评测集 —— 该评测集不参与模型的任何训练过程，仅用于验证模型的实际识别性能。

1. ***生成标注文件，存放在Annotations文件夹***

安装jinjia2, 执行
pip install Jinja2 pillow

使用python程序tt100k_to_voc_test.py和tt100k_to_voc_train.py将
TT100K数据集转换成PASCAL VOC的xml标记文件。
执行:
python tt100k_to_voc_test.py
python tt100k_to_voc_train.py
执行完成后，xxx/TT100K/VOCdevkit/VOC2007/Annotations文件中会生成xml格式的标注文件

2. ***生成图片数据，存放在JPEGImages文件夹***

VOCdevkit\VOC2007下面有两个文件夹：Annotations和JPEGImages
JPEGImages--放所有的训练和测试图片；
Annotations--放所有的xml标注文件；

再拷贝图片文件至 VOCdevkit\VOC2007\JPEGImages：
window系统：
手动拷贝TT100K/test/文件夹下所有的图片到xxx/VOCdevkit/VOC2007/JPEGImages
手动拷贝TT100K/train/文件夹下所有的图片到xxx/VOCdevkit/VOC2007/JPEGImages
注： 换成自己的路径

3. ***拷贝数据集到YOLO26项目中***

将VOCdevkit数据集文件夹拷贝到xxx/ultralytics/ultralytics
testfiles.zip (下载到xxx/ultralytics/ultralytics目录下并解压)
prepare_data.py (下载到xxx/ultralytics/ultralytics目录下)

拷贝完成后的目录结构如右图：![img_5.png](img_5.png)

4. ***划分训练集和验证集***

xxx/ultralytics/ultralytics路径下，执行python脚本：
python prepare_data.py

注意：
在VOCdevkit目录下生成了images和labels文件夹
    images文件夹下有train和val文件夹，分别放置训练集和验证集图片；
    labels文件夹有train和val文件夹，分别放置训练集和验证集标签（yolo格式）；

在xxx/ultralytics/ultralytics下生成了两个文件yolov8_train.txt和yolov8_val.txt。
     yolov8_train.txt和yolov8_val.txt分别给出了训练图片文件和验证图片文件的列表，含有每个图片的路径和文件名。

转换后的数据集如右图：

![img_6.png](img_6.png)

5. ***修改配置文件***

5.1 将VOC-tt100k.yaml文件拷贝到xxx/ultralytics/ultralytics/cfg/datasets，并将文件中的path修改成自己的路径

5.2 修改文件xxx/ultralytics/ultralytics/cfg/default.yaml
注意：为避免交通标志中的左转和右转标志的混淆，不做左右翻转的数据增强

fliplr: 0.0 # (float) image flip left-right (probability)

### 制作评测集

为了客观验证模型的实际泛化能力和识别精度，项目额外构建了一个独立的评测集 —— 该评测集不参与模型的任何训练或调优过程，完全用于验证模型的实际业务性能。评测集的构建标准，与工业级目标检测任务的评测要求完全对齐，其内容覆盖了训练集样本中占比较少的小目标交通标志、以及不同天气 / 光照 / 角度下的极端场景，能客观反映模型在实际业务场景下的泛化能力。

#### 采集与筛选图像

评测集的图像采集遵循以下两个要求：

1. **场景独立性**：所有图像均来自 TT100K 的测试子集、或实际道路监控中采集的未参与过训练的新图像，图像场景需覆盖训练集中占比较低的极端场景 —— 例如夜间低光照、雨天反光、雾天能见度低、以及大 / 小角度倾斜下的交通标识样本；

2. **样本分布均衡**：评测集的样本需在各类别间均衡分布 —— 每个类别的交通标识样本数量，应与该类别在训练集中的样本占比保持一致，避免因某类样本过多或过少，导致评测结果出现偏差。

#### 使用 LabelImg 标注图像

项目采用 LabelImg 作为评测集的标注工具 —— 这是一款开源的图形化图像标注工具，能生成 YOLO 格式的标注文件，完全适配本项目的训练需求。标注的完整流程如下：


1. **安装 LabelImg**：通过 pip 包管理工具直接安装，命令为`pip install labelImg`；

2. **启动工具**：在终端输入`labelImg`命令，即可启动标注工具界面；

3. **配置标注环境**：点击界面顶部的 “Open Dir” 按钮，选择存放评测集原始图像的目录；再点击 “Change Save Dir” 按钮，选择专门用于存放标注文件的目录 —— 强烈建议将图像文件和标注文件分开存放，后续整理数据集目录时，将标注文件放入对应的 labels 目录下；

4. **设置标注格式**：在界面顶部的 “View” 下拉菜单中，勾选 “Auto Save mode” 选项 —— 开启自动保存功能，可避免因忘记保存而丢失标注进度；再在 “Format” 下拉菜单中，选择 “YOLO” 格式 —— 这是保证标注文件能被 YOLO26 读取的关键步骤；

5. **配置预定义类别**：打开 LabelImg 安装目录下的`data/predefined_classes.txt`文件，将里面的默认类别替换为项目实际使用的 21 类交通标识名称 —— 标注时的类别输入框，会自动读取该文件中定义的类别列表，避免手动输入类别的拼写错误；

6. **开始标注**：在图像上按住鼠标左键拖动，选中需要标注的交通标识区域，松开鼠标后会弹出类别选择对话框；在对话框中选择与该标识对应的类别名称后，按 Enter 键确认；标注完成后，按 Ctrl+S 快捷键保存标注结果，然后通过键盘的 D 键或界面的 “Next Image” 按钮，切换到下一张图像继续标注。

#### 验证标注结果

在完成所有评测集图像的标注后，需先对标注结果进行验证，再将其投入实际评测环节。验证的核心逻辑是：检查每个标注文件的格式是否符合 YOLO 规范、每个标注框的坐标是否在合理的范围内 —— 如果标注文件的格式不正确，或坐标数值超出了 0～1 的合理区间，会导致后续评测过程中出现数据加载错误。

为简化验证流程，项目在`scripts/`目录下提供了`verify_annotations.py`验证脚本，执行该脚本即可自动完成所有标注文件的格式验证。验证完成后，将评测集的图像文件和标注文件，分别整理到与训练集结构对应的`images/test/`和`labels/test/`目录下，替换原有的测试集文件，或在训练配置文件中，单独指定评测集的路径。



# 4.训练方法


# 5.推理方法



# 6.评测方法

## yolo26-gui

26 暑期学校实习成果：基于 YOLO26 的交通标识检测 GUI 项目。

这个仓库目前采用“脚本优先”的轻量结构，适合在 Windows 上直接运行主程序和工具脚本。

当前 GUI 已实现三个主功能：

- 单图预测
- 视频预测
- 模型评估

## 目录结构

```text
yolo26-gui/
├─ evaluate/
│  ├─ input/
│  ├─ input_video/
│  ├─ output/
│  └─ output_video/
├─ gui/
│  ├─ __init__.py
│  ├─ backend_worker.py
│  └─ main_window.py
├─ models/
├─ scripts/
│  ├─ model_convert.py
│  ├─ predict.py
│  ├─ check_dataset.py
│  ├─ evaluate_dataset.py
│  └─ predict_video.py
├─ temp/
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

GUI 启动后可直接使用：

- 顶部模型菜单加载 `.onnx` 或 `.pt` 模型
- 单图预测页选择图片并查看 GT / 预测框
- 视频预测页加载视频、播放、导出预测视频
- 模型评估页执行数据集检查和评测

如果你希望先安装依赖，再运行主程序：

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py main.py
```

## 依赖安装

当前项目中的五个脚本统一使用 `pip` 安装依赖。

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
- `predict_video.py` 依赖 `onnxruntime-directml`、`opencv-python`、`numpy`
- `onnxruntime-directml` 安装后，代码中的导入名仍然是 `onnxruntime`
- `PySide6` 用于 GUI

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
- 当前实现会先在 `temp/model_convert/` 下生成临时导出文件，再移动到目标位置，避免覆盖现有 `models/*.onnx`

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
- 兼容 `txt` 与 `xml` 标注；若同名 `txt` 不存在，会自动尝试同名 `xml`
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
- 支持读取 `txt` 标注；若同名 `txt` 不存在，会自动尝试 Pascal VOC `xml`
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

### 5. `scripts/predict_video.py`

作用：

- 对一段交通标志牌视频执行逐帧检测
- 将检测框、类别名称、置信度绘制到视频帧上
- 在显示窗口中实时叠加 FPS
- 导出带检测结果的输出视频

核心函数：

- `iter_video_predictions(model, video_path, conf_threshold=0.25, classes_path=None)`
- `predict_video(model_path, video_path="evaluate/input_video", output_path="evaluate/output_video", conf_threshold=0.25, classes_path=None, show=True)`
- `render_detections(frame, detections, class_names=None, fps=None)`

说明：

- CLI 模式下可直接读取一个视频并保存结果
- 函数层已经拆成“逐帧结果生成 + 渲染”两层，后续 GUI 页面可以直接复用 `iter_video_predictions(...)` 获取每一帧的原图、检测结果、渲染结果和 FPS
- 另外提供 `VideoPredictionSession`，支持 `seek_frame(...)` 和 `seek_time(...)`，便于后续 GUI 实现播放进度条拖动和跳转

命令行用法：

```powershell
py scripts/predict_video.py --onnx models/best.onnx
```

常用可选参数：

- `--input evaluate/input_video`：输入视频路径，或包含单个视频的目录
- `--output evaluate/output_video`：输出视频路径，或输出目录
- `--classes evaluate/input/classes.txt`：可选类别名文件
- `--conf 0.25`：置信度阈值
- `--no-show`：只保存输出视频，不弹实时显示窗口

示例：

```powershell
py scripts/predict_video.py --onnx models/best.onnx --input evaluate/input_video/demo.mp4 --output evaluate/output_video --conf 0.25
```

命令行输出：

- 终端打印输入视频、输出视频、分辨率、输入 FPS、总帧数、已处理帧数、平均推理 FPS
- 默认弹出实时检测窗口，按 `q` 或 `Esc` 可提前结束

输出文件：

- 默认输出到 `evaluate/output_video/<输入视频名>_pred.mp4`

## GUI 说明

### 模型加载

- 支持直接加载 `.onnx`
- 支持选择 `.pt`，GUI 会先调用转换脚本
- `.pt` 转换结果会先输出到 `temp/`，再重命名后移动到 `models/` 目录

### 单图预测

- 选择图片后会立即加载到左右两个预览框
- 支持显示基准框、显示预测框
- 支持滚轮缩放、按钮缩放、鼠标拖拽平移

### 视频预测

- 支持加载视频、播放 / 暂停、拖动进度条
- 可切换“播放时启用预测”
- 支持导出预测后视频到 `evaluate/output_video/`

### 模型评估

- “检查数据集”会输出数据集质量报告
- “开始评估”会输出 Precision、Recall、mAP50、误检率、漏检率等摘要
- 评估结果默认保存到 `evaluate/output/records/summary.json` 与 `evaluate/output/records/per_image_results.json`

## 日志

- GUI 运行时日志默认写入 `temp/gui_runtime.log`
- 如果出现“模型评估后再单图预测崩溃”等问题，可以优先查看这个日志文件

## 开发建议

- GUI 主入口统一从 `main.py` 启动
- 如果后续接入模型，可新增顶层 `core/`、`gui/pages/` 等普通模块目录
- 如果后续需要打包成 `exe`，再补充 PyInstaller 配置即可
- `tests/` 先保留最小结构，后续加功能时可以同步补测试


# 7.可视化界面使用方法

# 8.ONNX转换与推理方法

# 9.项目成员分工

# 10.实验结果展示

# 11.常见问题


