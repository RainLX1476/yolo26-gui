# yolo26-gui

26 暑期学校实习成果：基于 YOLO26 的交通标识检测 GUI 项目。

这个仓库目前采用“轻量工程化”结构，适合在 Windows 上以脚本方式直接运行，同时保留后续扩展成较完整项目的空间。

## 目录结构

```text
yolo26-gui/
├─ src/
│  └─ yolo26_gui/
│     ├─ __init__.py
│     ├─ __main__.py
│     └─ app.py
├─ tests/
│  └─ test_app.py
├─ .gitignore
├─ pyproject.toml
└─ README.md
```

## 运行方式

推荐使用 Python `3.10+`。

```powershell
py -m yolo26_gui
```

如果你希望按包方式导入，建议先安装当前项目：

```powershell
py -m pip install -e .
```

## 开发建议

- 把实际 GUI 入口逻辑补到 `src/yolo26_gui/app.py`
- 如果后续接入模型，可新增 `src/yolo26_gui/inference/`
- 如果后续需要打包成 `exe`，再补充 PyInstaller 配置即可
- `tests/` 先保留最小结构，后续加功能时可以同步补测试
