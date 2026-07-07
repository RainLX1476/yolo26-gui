from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
	QApplication,
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QMenu,
	QTabWidget,
	QToolButton,
	QVBoxLayout,
	QWidget,
)

from gui.backend_worker import BackendWorker


def _create_placeholder_page(title: str, description: str) -> QWidget:
	"""创建暂不包含业务功能的占位页面。"""
	page = QWidget()
	layout = QVBoxLayout(page)

	title_label = QLabel(title)
	title_label.setStyleSheet("font-size: 22px; font-weight: 600;")

	description_label = QLabel(description)
	description_label.setWordWrap(True)
	description_label.setStyleSheet("font-size: 14px; color: #555;")

	layout.addWidget(title_label)
	layout.addWidget(description_label)
	layout.addStretch()
	return page


class MainWindow(QMainWindow):
	"""主窗口。"""

	load_model_requested = Signal(str)
	unload_model_requested = Signal()
	switch_model_requested = Signal(str)

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("YOLO26 GUI")
		self.resize(800, 480)
		self.model_status = "idle"
		self.current_model_path: Path | None = None
		self._init_backend()

		central_widget = QWidget()
		central_layout = QVBoxLayout(central_widget)

		status_layout = QHBoxLayout()
		status_title = QLabel("模型状态")
		status_title.setStyleSheet("font-size: 14px; font-weight: 600;")

		self.model_status_dot = QLabel()
		self.model_status_dot.setFixedSize(12, 12)

		self.model_status_label = QLabel()
		self.model_status_label.setStyleSheet("font-size: 14px; color: #444;")

		self.model_action_button = QToolButton()
		self.model_action_button.setPopupMode(QToolButton.InstantPopup)
		self.model_action_button.setMinimumWidth(130)
		self.model_action_menu = QMenu(self)
		self.model_action_button.setMenu(self.model_action_menu)

		status_layout.addWidget(status_title)
		status_layout.addSpacing(8)
		status_layout.addWidget(self.model_status_dot)
		status_layout.addWidget(self.model_status_label)
		status_layout.addStretch()
		status_layout.addWidget(self.model_action_button)

		tab_widget = QTabWidget()
		tab_widget.addTab(
			_create_placeholder_page(
				"图片识别",
				"该页面用于后续接入单张图片的目标检测与结果展示功能。",
			),
			"图片识别",
		)
		tab_widget.addTab(
			_create_placeholder_page(
				"视频识别",
				"该页面用于后续接入视频文件或摄像头的目标检测功能。",
			),
			"视频识别",
		)
		tab_widget.addTab(
			_create_placeholder_page(
				"模型评测",
				"该页面用于后续接入模型精度、速度和相关评测结果展示功能。",
			),
			"模型评测",
		)

		central_layout.addLayout(status_layout)
		central_layout.addWidget(tab_widget)
		self.setCentralWidget(central_widget)

		self.set_model_status("idle")

	def _init_backend(self) -> None:
		"""初始化后台处理线程。"""
		self.backend_thread = QThread(self)
		self.backend_worker = BackendWorker()
		self.backend_worker.moveToThread(self.backend_thread)

		self.load_model_requested.connect(self.backend_worker.load_model)
		self.unload_model_requested.connect(self.backend_worker.unload_model)
		self.switch_model_requested.connect(self.backend_worker.switch_model)
		self.backend_worker.status_changed.connect(self.set_model_status)
		self.backend_worker.model_changed.connect(self._on_model_changed)

		self.backend_thread.start()

	def _models_dir(self) -> Path:
		"""返回项目根目录下的模型目录。"""
		return Path(__file__).resolve().parents[1] / "models"

	def _find_model_files(self) -> list[Path]:
		"""查找 `models` 目录下可选的模型文件。"""
		models_dir = self._models_dir()
		if not models_dir.exists():
			return []

		model_files: list[Path] = []
		for pattern in ("*.onnx", "*.pt"):
			model_files.extend(models_dir.glob(pattern))
		return sorted(model_files)

	def _request_load_model(self, model_path: str | Path, *, switch: bool = False) -> None:
		"""向后台线程发送模型加载或切换命令。"""
		model_file = str(Path(model_path).expanduser().resolve())
		if switch and self.current_model_path is not None:
			self.switch_model_requested.emit(model_file)
			return
		self.load_model_requested.emit(model_file)

	def _select_other_model(self) -> None:
		"""打开系统文件选择框，选择其他位置的模型。"""
		start_dir = str(self._models_dir())
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"选择模型文件",
			start_dir,
			"模型文件 (*.onnx *.pt);;所有文件 (*.*)",
		)
		if file_path:
			self._request_load_model(file_path, switch=self.current_model_path is not None)

	def _on_model_changed(self, model_path: object) -> None:
		"""接收后台线程发回的当前模型路径。"""
		if model_path:
			self.current_model_path = Path(str(model_path))
		else:
			self.current_model_path = None
		self._refresh_model_menu(self.model_status)

	def _refresh_model_menu(self, status: str) -> None:
		"""根据当前状态刷新右侧下拉菜单。"""
		self.model_action_menu.clear()
		model_files = self._find_model_files()

		if status == "loaded" and self.current_model_path is not None:
			current_action = QAction(f"当前模型: {self.current_model_path.name}", self)
			current_action.setEnabled(False)
			self.model_action_menu.addAction(current_action)
			self.model_action_menu.addSeparator()

			unload_action = QAction("卸载模型", self)
			unload_action.triggered.connect(lambda checked=False: self.unload_model_requested.emit())
			self.model_action_menu.addAction(unload_action)

			if model_files:
				switch_menu = self.model_action_menu.addMenu("切换模型")
				for model_path in model_files:
					action = QAction(model_path.name, self)
					action.triggered.connect(
						lambda checked=False, path=model_path: self._request_load_model(path, switch=True)
					)
					switch_menu.addAction(action)

			other_action = QAction("选择其他位置的模型...", self)
			other_action.triggered.connect(self._select_other_model)
			self.model_action_menu.addAction(other_action)
			return

		if model_files:
			for model_path in model_files:
				action = QAction(model_path.name, self)
				action.triggered.connect(
					lambda checked=False, path=model_path: self._request_load_model(path)
				)
				self.model_action_menu.addAction(action)
			self.model_action_menu.addSeparator()

		other_action = QAction("选择其他位置的模型...", self)
		other_action.triggered.connect(self._select_other_model)
		self.model_action_menu.addAction(other_action)

	def set_model_status(self, status: str, message: str | None = None) -> None:
		"""更新模型加载状态指示器。"""
		self.model_status = status
		status_styles = {
			"idle": ("#9AA0A6", "未加载模型", "加载模型"),
			"loading": ("#F4B400", "模型加载中...", "加载中..."),
			"loaded": ("#34A853", "模型已加载", "模型操作"),
			"error": ("#D93025", "模型加载失败", "加载模型"),
		}
		color, default_message, button_text = status_styles.get(
			status,
			status_styles["idle"],
		)
		self.model_status_dot.setStyleSheet(
			f"background-color: {color}; border-radius: 6px;"
		)
		self.model_status_label.setText(message or default_message)
		self.model_action_button.setText(button_text)
		self.model_action_button.setEnabled(status != "loading")
		self._refresh_model_menu(status)

	def closeEvent(self, event) -> None:
		"""窗口关闭时停止后台线程。"""
		self.backend_thread.quit()
		self.backend_thread.wait()
		super().closeEvent(event)


def main() -> None:
	"""GUI 应用程序入口。"""
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec())
