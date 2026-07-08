from __future__ import annotations

import json
import logging
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QDialog,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QMenu,
	QMessageBox,
	QPlainTextEdit,
	QPushButton,
	QSizePolicy,
	QSlider,
	QSplitter,
	QTabWidget,
	QToolButton,
	QVBoxLayout,
	QWidget,
)

from gui.backend_worker import BackendWorker
from scripts.evaluate_dataset import _class_name, _find_label_path, _load_class_names, _load_ground_truths

LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


LOG_PATH = _repo_root() / "temp" / "gui_runtime.log"


def _format_seconds(seconds: float) -> str:
	total_seconds = max(int(seconds), 0)
	minutes, secs = divmod(total_seconds, 60)
	hours, minutes = divmod(minutes, 60)
	if hours:
		return f"{hours:02d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:02d}:{secs:02d}"


def _setup_logging() -> Path:
	LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO)

	if not any(
		isinstance(handler, logging.FileHandler)
		and Path(getattr(handler, "baseFilename", "")).resolve() == LOG_PATH.resolve()
		for handler in root_logger.handlers
	):
		file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
		file_handler.setFormatter(
			logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
		)
		root_logger.addHandler(file_handler)

	return LOG_PATH


def _log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
	if issubclass(exc_type, KeyboardInterrupt):
		sys.__excepthook__(exc_type, exc_value, exc_traceback)
		return
	LOGGER.error(
		"Unhandled exception:\n%s",
		"".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
	)
	sys.__excepthook__(exc_type, exc_value, exc_traceback)


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
	LOGGER.error(
		"Unhandled thread exception in %s:\n%s",
		args.thread.name if args.thread else "unknown-thread",
		"".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
	)


class ImagePreviewLabel(QLabel):
	"""支持自动缩放显示 QPixmap 的预览标签。"""

	def __init__(self, placeholder: str) -> None:
		super().__init__(placeholder)
		self.setAlignment(Qt.AlignCenter)
		self.setMinimumSize(180, 120)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.setStyleSheet(
			"background: #f5f7fa; border: 1px solid #d7dce3; border-radius: 8px; color: #5f6b7a;"
		)
		self.setWordWrap(True)
		self._pixmap: QPixmap | None = None
		self._zoom_factor = 1.0
		self._drag_last_pos = None
		self._pan_offset = (0, 0)

	def set_preview_pixmap(self, pixmap: QPixmap | None) -> None:
		self._pixmap = pixmap
		self._refresh()

	def clear_preview(self, text: str) -> None:
		self._pixmap = None
		self._zoom_factor = 1.0
		self._drag_last_pos = None
		self._pan_offset = (0, 0)
		self.clear()
		self.setText(text)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		self._refresh()

	def wheelEvent(self, event) -> None:
		if self._pixmap is None:
			super().wheelEvent(event)
			return
		delta = event.angleDelta().y()
		if delta > 0:
			self.zoom_in()
		elif delta < 0:
			self.zoom_out()
		event.accept()

	def zoom_in(self) -> None:
		self._zoom_factor = min(self._zoom_factor * 1.15, 8.0)
		self._refresh()

	def zoom_out(self) -> None:
		self._zoom_factor = max(self._zoom_factor / 1.15, 0.2)
		self._refresh()

	def reset_zoom(self) -> None:
		self._zoom_factor = 1.0
		self._pan_offset = (0, 0)
		self._refresh()

	def mousePressEvent(self, event) -> None:
		if self._pixmap is not None and event.button() == Qt.LeftButton:
			self._drag_last_pos = event.position()
			self.setCursor(Qt.ClosedHandCursor)
			event.accept()
			return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event) -> None:
		if self._pixmap is not None and self._drag_last_pos is not None:
			current_pos = event.position()
			delta = current_pos - self._drag_last_pos
			offset_x = int(self._pan_offset[0] + delta.x())
			offset_y = int(self._pan_offset[1] + delta.y())
			self._pan_offset = (offset_x, offset_y)
			self._drag_last_pos = current_pos
			self._refresh()
			event.accept()
			return
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event) -> None:
		if event.button() == Qt.LeftButton and self._drag_last_pos is not None:
			self._drag_last_pos = None
			self.setCursor(Qt.ArrowCursor)
			event.accept()
			return
		super().mouseReleaseEvent(event)

	def _refresh(self) -> None:
		if self._pixmap is None:
			return
		target_size = self.size() * self._zoom_factor
		scaled = self._pixmap.scaled(
			target_size,
			Qt.KeepAspectRatio,
			Qt.SmoothTransformation,
		)
		canvas = QPixmap(self.size())
		canvas.fill(Qt.transparent)
		painter_x = (self.width() - scaled.width()) // 2 + self._pan_offset[0]
		painter_y = (self.height() - scaled.height()) // 2 + self._pan_offset[1]
		from PySide6.QtGui import QPainter

		painter = QPainter(canvas)
		painter.drawPixmap(painter_x, painter_y, scaled)
		painter.end()
		self.setPixmap(canvas)


def _cv_to_pixmap(image_bgr) -> QPixmap:
	rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	height, width, channels = rgb_image.shape
	bytes_per_line = channels * width
	image = QImage(
		rgb_image.data,
		width,
		height,
		bytes_per_line,
		QImage.Format_RGB888,
	)
	return QPixmap.fromImage(image.copy())


def _draw_labeled_box(
	image,
	box: list[float],
	text: str,
	color: tuple[int, int, int],
) -> None:
	x1, y1, x2, y2 = [int(round(value)) for value in box]
	cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
	label_y = max(18, y1 - 6)
	cv2.putText(
		image,
		text,
		(x1, label_y),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		color,
		2,
		cv2.LINE_AA,
	)


class HistogramDialog(QDialog):
	"""展示数据集检查直方图的弹窗。"""

	def __init__(self, title: str, image_path: str | Path, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.setWindowTitle(title)
		self.resize(960, 720)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(10, 10, 10, 10)

		self.preview_label = ImagePreviewLabel("直方图加载失败")
		layout.addWidget(self.preview_label, 1)

		pixmap = QPixmap(str(image_path))
		if pixmap.isNull():
			self.preview_label.clear_preview(f"无法加载直方图:\n{image_path}")
		else:
			self.preview_label.set_preview_pixmap(pixmap)


def _create_zoom_controls(target_label: ImagePreviewLabel) -> QWidget:
	container = QWidget()
	layout = QHBoxLayout(container)
	layout.setContentsMargins(0, 0, 0, 0)
	layout.setSpacing(6)

	zoom_out_button = QPushButton("-")
	zoom_out_button.setFixedWidth(32)
	zoom_out_button.clicked.connect(target_label.zoom_out)

	reset_button = QPushButton("1:1")
	reset_button.setFixedWidth(44)
	reset_button.clicked.connect(target_label.reset_zoom)

	zoom_in_button = QPushButton("+")
	zoom_in_button.setFixedWidth(32)
	zoom_in_button.clicked.connect(target_label.zoom_in)

	hint_label = QLabel("滚轮缩放")
	hint_label.setStyleSheet("color: #6b7280;")

	layout.addWidget(zoom_out_button)
	layout.addWidget(reset_button)
	layout.addWidget(zoom_in_button)
	layout.addWidget(hint_label)
	layout.addStretch()
	return container


class MainWindow(QMainWindow):
	"""YOLO2 交通标识牌识别 GUI。"""

	load_model_requested = Signal(str)
	unload_model_requested = Signal()
	switch_model_requested = Signal(str)

	predict_image_requested = Signal(str, float)
	check_dataset_requested = Signal(str)
	evaluate_dataset_requested = Signal(str, str, float, float, object)

	open_video_requested = Signal(str, float, object)
	close_video_requested = Signal()
	read_video_frame_requested = Signal(bool)
	seek_video_frame_requested = Signal(int)
	export_video_prediction_requested = Signal(str, str, float, object)

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("YOLO26 GUI")
		self.resize(800, 480)

		self.model_status = "idle"
		self.current_model_path: Path | None = None
		self.runtime_model_path: Path | None = None
		self.current_image_path: Path | None = None
		self.current_image_data = None
		self.current_ground_truths: list[Any] = []
		self.current_image_class_names: dict[int, str] = {}
		self.current_image_predictions: list[dict[str, Any]] = []

		self.video_metadata: dict[str, Any] | None = None
		self.video_frame_pending = False
		self.video_slider_active = False
		self.histogram_dialogs: list[HistogramDialog] = []

		self._init_backend()
		self._build_ui()
		self._wire_backend_signals()
		self.set_model_status("idle")
		self._fill_default_paths()

	def _build_ui(self) -> None:
		central_widget = QWidget()
		central_layout = QVBoxLayout(central_widget)
		central_layout.setContentsMargins(10, 10, 10, 10)
		central_layout.setSpacing(8)

		central_layout.addLayout(self._build_header())

		self.tab_widget = QTabWidget()
		self.tab_widget.addTab(self._build_image_tab(), "单图预测")
		self.tab_widget.addTab(self._build_video_tab(), "视频预测")
		self.tab_widget.addTab(self._build_dataset_tab(), "模型评估")
		central_layout.addWidget(self.tab_widget)

		self.setCentralWidget(central_widget)
		self.statusBar().showMessage("准备就绪")

	def _build_header(self) -> QHBoxLayout:
		status_layout = QHBoxLayout()
		status_title = QLabel("模型状态")
		status_title.setStyleSheet("font-size: 14px; font-weight: 600;")

		self.model_status_dot = QLabel()
		self.model_status_dot.setFixedSize(12, 12)

		self.model_status_label = QLabel()
		self.model_status_label.setStyleSheet("font-size: 14px; color: #444;")

		self.model_detail_label = QLabel("当前未加载模型")
		self.model_detail_label.setStyleSheet("font-size: 12px; color: #6b7280;")

		self.model_action_button = QToolButton()
		self.model_action_button.setPopupMode(QToolButton.InstantPopup)
		self.model_action_button.setMinimumWidth(110)
		self.model_action_menu = QMenu(self)
		self.model_action_button.setMenu(self.model_action_menu)

		model_text_layout = QVBoxLayout()
		model_text_layout.setSpacing(2)
		model_text_layout.addWidget(self.model_status_label)
		model_text_layout.addWidget(self.model_detail_label)

		status_layout.addWidget(status_title)
		status_layout.addSpacing(8)
		status_layout.addWidget(self.model_status_dot)
		status_layout.addLayout(model_text_layout)
		status_layout.addStretch()
		status_layout.addWidget(self.model_action_button)
		return status_layout

	def _build_image_tab(self) -> QWidget:
		page = QWidget()
		layout = QVBoxLayout(page)
		layout.setSpacing(8)

		control_box = QGroupBox("图片输入")
		control_layout = QGridLayout(control_box)
		control_layout.setColumnStretch(1, 1)

		self.image_path_edit = QLineEdit()
		self.image_path_edit.setPlaceholderText("选择待预测图片")
		image_browse_button = QPushButton("浏览图片")
		image_browse_button.clicked.connect(self._browse_image_file)

		self.image_conf_spin = QDoubleSpinBox()
		self.image_conf_spin.setRange(0.01, 1.0)
		self.image_conf_spin.setSingleStep(0.05)
		self.image_conf_spin.setValue(0.25)

		self.image_predict_button = QPushButton("开始预测")
		self.image_predict_button.clicked.connect(self._predict_image)

		control_layout.addWidget(QLabel("图片路径"), 0, 0)
		control_layout.addWidget(self.image_path_edit, 0, 1)
		control_layout.addWidget(image_browse_button, 0, 2)
		control_layout.addWidget(QLabel("置信度阈值"), 1, 0)
		control_layout.addWidget(self.image_conf_spin, 1, 1)
		control_layout.addWidget(self.image_predict_button, 1, 2)

		image_options_layout = QHBoxLayout()
		self.left_show_gt_checkbox = QCheckBox("左侧显示基准框")
		self.left_show_gt_checkbox.setChecked(False)
		self.left_show_gt_checkbox.toggled.connect(self._refresh_image_previews)
		self.right_show_gt_checkbox = QCheckBox("右侧显示基准框")
		self.right_show_gt_checkbox.setChecked(False)
		self.right_show_gt_checkbox.setEnabled(False)
		self.right_show_gt_checkbox.toggled.connect(self._refresh_image_previews)
		self.right_show_pred_checkbox = QCheckBox("右侧显示预测框")
		self.right_show_pred_checkbox.setChecked(False)
		self.right_show_pred_checkbox.setEnabled(False)
		self.right_show_pred_checkbox.toggled.connect(self._refresh_image_previews)
		image_options_layout.addWidget(self.left_show_gt_checkbox)
		image_options_layout.addStretch()
		image_options_layout.addWidget(self.right_show_gt_checkbox)
		image_options_layout.addWidget(self.right_show_pred_checkbox)

		preview_widget = QWidget()
		preview_layout = QHBoxLayout(preview_widget)
		preview_layout.setContentsMargins(0, 0, 0, 0)
		self.original_image_label = ImagePreviewLabel("原图预览")
		self.predicted_image_label = ImagePreviewLabel("预测结果预览")

		left_preview_panel = QWidget()
		left_preview_layout = QVBoxLayout(left_preview_panel)
		left_preview_layout.setContentsMargins(0, 0, 0, 0)
		left_preview_layout.setSpacing(6)
		left_preview_layout.addWidget(self.original_image_label, 1)
		left_preview_layout.addWidget(_create_zoom_controls(self.original_image_label))

		right_preview_panel = QWidget()
		right_preview_layout = QVBoxLayout(right_preview_panel)
		right_preview_layout.setContentsMargins(0, 0, 0, 0)
		right_preview_layout.setSpacing(6)
		right_preview_layout.addWidget(self.predicted_image_label, 1)
		right_preview_layout.addWidget(_create_zoom_controls(self.predicted_image_label))

		preview_layout.addWidget(left_preview_panel, 1)
		preview_layout.addWidget(right_preview_panel, 1)

		self.image_result_text = QPlainTextEdit()
		self.image_result_text.setReadOnly(True)
		line_height = self.image_result_text.fontMetrics().lineSpacing()
		min_result_height = line_height * 6 + 24
		self.image_result_text.setMinimumHeight(min_result_height)
		self.image_result_text.setPlaceholderText("检测结果会显示在这里，可拖动上方分隔条调整输出框大小")

		image_splitter = QSplitter(Qt.Vertical)
		image_splitter.addWidget(preview_widget)
		image_splitter.addWidget(self.image_result_text)
		image_splitter.setChildrenCollapsible(False)
		image_splitter.setStretchFactor(0, 4)
		image_splitter.setStretchFactor(1, 1)
		image_splitter.setSizes([1, min_result_height])

		layout.addWidget(control_box)
		layout.addLayout(image_options_layout)
		layout.addWidget(image_splitter, 1)
		return page

	def _build_video_tab(self) -> QWidget:
		page = QWidget()
		layout = QVBoxLayout(page)
		layout.setSpacing(8)

		control_box = QGroupBox("视频输入与播放")
		control_layout = QGridLayout(control_box)
		control_layout.setColumnStretch(1, 1)

		self.video_path_edit = QLineEdit()
		self.video_path_edit.setPlaceholderText("选择待播放/预测的视频")
		video_browse_button = QPushButton("浏览视频")
		video_browse_button.clicked.connect(self._browse_video_file)

		self.video_conf_spin = QDoubleSpinBox()
		self.video_conf_spin.setRange(0.01, 1.0)
		self.video_conf_spin.setSingleStep(0.05)
		self.video_conf_spin.setValue(0.25)

		self.video_predict_checkbox = QCheckBox("播放时启用预测")
		self.video_predict_checkbox.setChecked(True)

		self.video_open_button = QPushButton("加载视频")
		self.video_open_button.clicked.connect(self._open_video)
		self.video_play_button = QPushButton("播放")
		self.video_play_button.clicked.connect(self._toggle_video_playback)
		self.video_play_button.setEnabled(False)
		self.video_export_button = QPushButton("导出预测视频")
		self.video_export_button.clicked.connect(self._export_video_prediction)
		self.video_export_button.setEnabled(False)

		control_layout.addWidget(QLabel("视频路径"), 0, 0)
		control_layout.addWidget(self.video_path_edit, 0, 1)
		control_layout.addWidget(video_browse_button, 0, 2)
		control_layout.addWidget(QLabel("置信度阈值"), 1, 0)
		control_layout.addWidget(self.video_conf_spin, 1, 1)
		control_layout.addWidget(self.video_predict_checkbox, 1, 2)
		control_layout.addWidget(self.video_open_button, 2, 0)
		control_layout.addWidget(self.video_export_button, 2, 2)

		self.video_preview_label = ImagePreviewLabel("加载视频后会在这里显示首帧/播放画面")
		self.video_preview_label.setMinimumHeight(170)

		self.video_slider = QSlider(Qt.Horizontal)
		self.video_slider.setEnabled(False)
		self.video_slider.sliderPressed.connect(self._on_video_slider_pressed)
		self.video_slider.sliderReleased.connect(self._on_video_slider_released)

		slider_layout = QHBoxLayout()
		slider_layout.addWidget(self.video_play_button)
		slider_layout.addWidget(self.video_slider, 1)

		info_layout = QHBoxLayout()
		self.video_time_label = QLabel("00:00 / 00:00")
		self.video_meta_label = QLabel("未加载视频")
		self.video_meta_label.setStyleSheet("color: #4b5563;")
		info_layout.addWidget(self.video_time_label)
		info_layout.addStretch()
		info_layout.addWidget(self.video_meta_label)

		video_status_widget = QWidget()
		video_status_layout = QHBoxLayout(video_status_widget)
		video_status_layout.setContentsMargins(0, 0, 0, 0)
		video_status_layout.setSpacing(14)

		self.video_status_mode_label = QLabel("状态: 未加载")
		self.video_status_frame_label = QLabel("帧: -")
		self.video_status_detect_label = QLabel("检测: -")
		self.video_status_fps_label = QLabel("FPS: -")
		self.video_status_extra_label = QLabel("输出: -")
		self.video_status_extra_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		for label in (
			self.video_status_mode_label,
			self.video_status_frame_label,
			self.video_status_detect_label,
			self.video_status_fps_label,
			self.video_status_extra_label,
		):
			label.setStyleSheet("color: #4b5563;")
			video_status_layout.addWidget(label)

		layout.addWidget(control_box)
		layout.addWidget(self.video_preview_label, 1)
		layout.addLayout(slider_layout)
		layout.addLayout(info_layout)
		layout.addWidget(video_status_widget)
		return page

	def _build_dataset_tab(self) -> QWidget:
		page = QWidget()
		layout = QVBoxLayout(page)
		layout.setSpacing(8)

		control_box = QGroupBox("模型评估参数")
		form_layout = QFormLayout(control_box)

		self.dataset_dir_edit = QLineEdit()
		self.dataset_dir_edit.setPlaceholderText("选择评测数据集目录")
		dataset_browse_button = QPushButton("浏览目录")
		dataset_browse_button.clicked.connect(self._browse_dataset_dir)
		dataset_row = self._with_button(self.dataset_dir_edit, dataset_browse_button)

		self.dataset_output_dir_edit = QLineEdit()
		self.dataset_output_dir_edit.setPlaceholderText("评测结果输出目录")
		output_browse_button = QPushButton("浏览目录")
		output_browse_button.clicked.connect(self._browse_dataset_output_dir)
		output_row = self._with_button(self.dataset_output_dir_edit, output_browse_button)

		self.dataset_classes_edit = QLineEdit()
		self.dataset_classes_edit.setPlaceholderText("可选：classes.txt / names.txt / dataset.yaml")
		classes_browse_button = QPushButton("浏览文件")
		classes_browse_button.clicked.connect(self._browse_dataset_classes_file)
		classes_row = self._with_button(self.dataset_classes_edit, classes_browse_button)

		self.eval_conf_spin = QDoubleSpinBox()
		self.eval_conf_spin.setRange(0.01, 1.0)
		self.eval_conf_spin.setSingleStep(0.05)
		self.eval_conf_spin.setValue(0.25)

		self.eval_iou_spin = QDoubleSpinBox()
		self.eval_iou_spin.setRange(0.1, 1.0)
		self.eval_iou_spin.setSingleStep(0.05)
		self.eval_iou_spin.setValue(0.5)

		form_layout.addRow("数据集目录", dataset_row)
		form_layout.addRow("输出目录", output_row)
		form_layout.addRow("类别文件", classes_row)
		form_layout.addRow("置信度阈值", self.eval_conf_spin)
		form_layout.addRow("IoU 阈值", self.eval_iou_spin)

		button_layout = QHBoxLayout()
		self.dataset_check_button = QPushButton("检查数据集")
		self.dataset_check_button.clicked.connect(self._check_dataset)
		self.dataset_evaluate_button = QPushButton("开始评估")
		self.dataset_evaluate_button.clicked.connect(self._evaluate_dataset)
		button_layout.addWidget(self.dataset_check_button)
		button_layout.addWidget(self.dataset_evaluate_button)
		button_layout.addStretch()

		self.dataset_summary_text = QPlainTextEdit()
		self.dataset_summary_text.setReadOnly(True)
		self.dataset_summary_text.setMinimumHeight(100)
		self.dataset_summary_text.setPlaceholderText("数据集检查报告与模型评估摘要会显示在这里")

		self.dataset_json_text = QPlainTextEdit()
		self.dataset_json_text.setReadOnly(True)
		self.dataset_json_text.setMinimumHeight(100)
		self.dataset_json_text.setPlaceholderText("结构化结果会显示在这里")

		json_box = QGroupBox("结构化结果")
		json_layout = QVBoxLayout(json_box)
		json_layout.addWidget(self.dataset_json_text)

		layout.addWidget(control_box)
		layout.addLayout(button_layout)
		layout.addWidget(self.dataset_summary_text, 1)
		layout.addWidget(json_box, 1)
		return page

	def _with_button(self, widget: QWidget, button: QPushButton) -> QWidget:
		container = QWidget()
		layout = QHBoxLayout(container)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(widget, 1)
		layout.addWidget(button)
		return container

	def _wire_backend_signals(self) -> None:
		self.backend_worker.status_changed.connect(self.set_model_status)
		self.backend_worker.model_changed.connect(self._on_model_changed)
		self.backend_worker.error_occurred.connect(self._on_backend_error)
		self.backend_worker.image_prediction_ready.connect(self._on_image_prediction_ready)
		self.backend_worker.dataset_check_ready.connect(self._on_dataset_check_ready)
		self.backend_worker.dataset_evaluation_ready.connect(self._on_dataset_evaluation_ready)
		self.backend_worker.video_loaded.connect(self._on_video_loaded)
		self.backend_worker.video_frame_ready.connect(self._on_video_frame_ready)
		self.backend_worker.video_playback_finished.connect(self._on_video_playback_finished)
		self.backend_worker.video_export_ready.connect(self._on_video_export_ready)
		self.backend_worker.video_closed.connect(self._on_video_closed)

	def _set_video_status(
		self,
		*,
		mode: str | None = None,
		frame: str | None = None,
		detections: str | None = None,
		fps: str | None = None,
		extra: str | None = None,
	) -> None:
		if mode is not None:
			self.video_status_mode_label.setText(f"状态: {mode}")
		if frame is not None:
			self.video_status_frame_label.setText(f"帧: {frame}")
		if detections is not None:
			self.video_status_detect_label.setText(f"检测: {detections}")
		if fps is not None:
			self.video_status_fps_label.setText(f"FPS: {fps}")
		if extra is not None:
			self.video_status_extra_label.setText(f"输出: {extra}")

	def _init_backend(self) -> None:
		self.backend_thread = QThread(self)
		self.backend_worker = BackendWorker()
		self.backend_worker.moveToThread(self.backend_thread)

		self.load_model_requested.connect(self.backend_worker.load_model)
		self.unload_model_requested.connect(self.backend_worker.unload_model)
		self.switch_model_requested.connect(self.backend_worker.switch_model)
		self.predict_image_requested.connect(self.backend_worker.predict_image_file)
		self.check_dataset_requested.connect(self.backend_worker.check_dataset_dir)
		self.evaluate_dataset_requested.connect(self.backend_worker.evaluate_dataset_dir)
		self.open_video_requested.connect(self.backend_worker.open_video)
		self.close_video_requested.connect(self.backend_worker.close_video)
		self.read_video_frame_requested.connect(self.backend_worker.read_video_frame)
		self.seek_video_frame_requested.connect(self.backend_worker.seek_video_frame)
		self.export_video_prediction_requested.connect(self.backend_worker.export_video_prediction)

		self.backend_thread.start()

		self.video_timer = QTimer(self)
		self.video_timer.timeout.connect(self._request_next_video_frame)

	def _fill_default_paths(self) -> None:
		root = _repo_root()
		self.dataset_dir_edit.setText(str(root / "evaluate" / "input"))
		self.dataset_output_dir_edit.setText(str(root / "evaluate" / "output"))

		default_image = root / "evaluate" / "input" / "1.jpg"
		if default_image.exists():
			self.image_path_edit.setText(str(default_image))

		default_video = root / "evaluate" / "input_video" / "1.mp4"
		if default_video.exists():
			self.video_path_edit.setText(str(default_video))

	def _models_dir(self) -> Path:
		return _repo_root() / "models"

	def _find_model_files(self) -> list[Path]:
		models_dir = self._models_dir()
		if not models_dir.exists():
			return []
		model_files: list[Path] = []
		for pattern in ("*.onnx", "*.pt"):
			model_files.extend(models_dir.glob(pattern))
		return sorted(model_files)

	def _request_load_model(self, model_path: str | Path, *, switch: bool = False) -> None:
		model_file = str(Path(model_path).expanduser().resolve())
		if switch and self.current_model_path is not None:
			self.switch_model_requested.emit(model_file)
		else:
			self.load_model_requested.emit(model_file)

	def _select_other_model(self) -> None:
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"选择模型文件",
			str(self._models_dir()),
			"模型文件 (*.onnx *.pt);;所有文件 (*.*)",
		)
		if file_path:
			self._request_load_model(file_path, switch=self.current_model_path is not None)

	def _on_model_changed(self, payload: object) -> None:
		if isinstance(payload, dict):
			model_path = payload.get("model_path")
			runtime_model_path = payload.get("runtime_model_path")
			self.current_model_path = Path(model_path) if model_path else None
			self.runtime_model_path = Path(runtime_model_path) if runtime_model_path else None
		else:
			self.current_model_path = None
			self.runtime_model_path = None

		if self.current_model_path is None:
			self.model_detail_label.setText("当前未加载模型")
		elif self.runtime_model_path and self.runtime_model_path != self.current_model_path:
			self.model_detail_label.setText(
				f"源模型: {self.current_model_path.name} | 运行时: {self.runtime_model_path.name}"
			)
		else:
			self.model_detail_label.setText(f"当前模型: {self.current_model_path.name}")
		self._refresh_model_menu(self.model_status)

	def _refresh_model_menu(self, status: str) -> None:
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

			self.model_action_menu.addSeparator()

		if model_files:
			for model_path in model_files:
				action = QAction(model_path.name, self)
				action.triggered.connect(
					lambda checked=False, path=model_path: self._request_load_model(
						path,
						switch=status == "loaded",
					)
				)
				self.model_action_menu.addAction(action)
			self.model_action_menu.addSeparator()

		other_action = QAction("选择其他位置的模型...", self)
		other_action.triggered.connect(self._select_other_model)
		self.model_action_menu.addAction(other_action)

	def set_model_status(self, status: str, message: str | None = None) -> None:
		self.model_status = status
		status_styles = {
			"idle": ("#9AA0A6", "未加载模型", "加载模型"),
			"loading": ("#F4B400", "模型加载中...", "加载中..."),
			"loaded": ("#34A853", "模型已加载", "模型操作"),
			"error": ("#D93025", "模型加载失败", "加载模型"),
		}
		color, default_message, button_text = status_styles.get(status, status_styles["idle"])
		self.model_status_dot.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
		self.model_status_label.setText(message or default_message)
		self.model_action_button.setText(button_text)
		self.model_action_button.setEnabled(status != "loading")
		self._refresh_model_menu(status)
		self.statusBar().showMessage(message or default_message, 5000)

	def _browse_image_file(self) -> None:
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"选择图片",
			str(_repo_root() / "evaluate" / "input"),
			"图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*.*)",
		)
		if file_path:
			self.image_path_edit.setText(file_path)
			self._load_selected_image(file_path)

	def _detect_dataset_root_for_image(self, image_path: Path) -> Path:
		candidate_roots = [
			_repo_root() / "evaluate" / "input",
			image_path.parent,
		]
		for candidate in candidate_roots:
			try:
				image_path.relative_to(candidate.resolve())
				return candidate.resolve()
			except ValueError:
				continue
		return image_path.parent

	def _load_selected_image(self, image_path: str | Path) -> None:
		image_file = Path(image_path).expanduser().resolve()
		image = cv2.imread(str(image_file))
		if image is None:
			self.original_image_label.clear_preview("图片加载失败")
			self.predicted_image_label.clear_preview("图片加载失败")
			self.current_image_path = None
			self.current_image_data = None
			self.current_ground_truths = []
			self.current_image_predictions = []
			self.image_result_text.setPlainText(f"无法读取图像文件: {image_file}")
			return

		dataset_root = self._detect_dataset_root_for_image(image_file)
		label_path = _find_label_path(image_file, dataset_root)
		class_names = _load_class_names(dataset_root, None)
		ground_truths = _load_ground_truths(label_path, image.shape[:2], class_names)

		self.current_image_path = image_file
		self.current_image_data = image
		self.current_ground_truths = ground_truths
		self.current_image_class_names = class_names
		self.current_image_predictions = []
		self.right_show_gt_checkbox.setEnabled(False)
		self.right_show_pred_checkbox.setEnabled(False)
		self.right_show_gt_checkbox.setChecked(False)
		self.right_show_pred_checkbox.setChecked(False)
		self._refresh_image_previews()

		lines = [
			f"图片: {image_file.name}",
			f"标注文件: {label_path.name if label_path.exists() else '未找到'}",
			f"基准框数量: {len(ground_truths)}",
			"",
			"已加载图片，可直接勾选查看基准框，或点击“开始预测”生成预测框。",
		]
		self.image_result_text.setPlainText("\n".join(lines))

	def _render_image_with_options(
		self,
		*,
		show_ground_truths: bool,
		show_predictions: bool,
	):
		if self.current_image_data is None:
			return None

		canvas = self.current_image_data.copy()
		if show_ground_truths:
			for gt in self.current_ground_truths:
				_draw_labeled_box(
					canvas,
					gt.box,
					f"GT {_class_name(gt.class_id, self.current_image_class_names)}",
					(0, 200, 0),
				)

		if show_predictions:
			for prediction in self.current_image_predictions:
				_draw_labeled_box(
					canvas,
					list(map(float, prediction["box"])),
					f"Pred {_class_name(int(prediction['class_id']), self.current_image_class_names)} {float(prediction['confidence']):.2f}",
					(0, 0, 255),
				)
		return canvas

	def _refresh_image_previews(self) -> None:
		if self.current_image_data is None:
			return

		left_image = self._render_image_with_options(
			show_ground_truths=self.left_show_gt_checkbox.isChecked(),
			show_predictions=False,
		)
		right_image = self._render_image_with_options(
			show_ground_truths=self.right_show_gt_checkbox.isChecked(),
			show_predictions=self.right_show_pred_checkbox.isChecked(),
		)

		if left_image is not None:
			self.original_image_label.set_preview_pixmap(_cv_to_pixmap(left_image))
		if right_image is not None:
			self.predicted_image_label.set_preview_pixmap(_cv_to_pixmap(right_image))

	def _predict_image(self) -> None:
		image_path = self.image_path_edit.text().strip()
		if not image_path:
			self._show_warning("请先选择图片文件")
			return
		if self.current_image_path is None or str(self.current_image_path) != str(Path(image_path).expanduser().resolve()):
			self._load_selected_image(image_path)
		self.image_predict_button.setEnabled(False)
		self.statusBar().showMessage("正在执行图片预测...")
		self.predict_image_requested.emit(image_path, self.image_conf_spin.value())

	def _on_image_prediction_ready(self, payload: object) -> None:
		self.image_predict_button.setEnabled(True)
		if not isinstance(payload, dict):
			return

		self.current_image_predictions = list(payload.get("detections", []))
		if self.current_image_path is None:
			self.current_image_path = Path(payload["image_path"])
		if self.current_image_data is None:
			self.current_image_data = payload["original_image"]
		self.right_show_gt_checkbox.setEnabled(True)
		self.right_show_pred_checkbox.setEnabled(True)
		self.right_show_pred_checkbox.setChecked(True)
		self._refresh_image_previews()

		detections = payload.get("detections", [])
		lines = [
			f"图片: {Path(payload['image_path']).name}",
			f"阈值: {float(payload['confidence_threshold']):.2f}",
			f"基准框数量: {len(self.current_ground_truths)}",
			f"检测数量: {len(detections)}",
			"",
		]
		if detections:
			for index, detection in enumerate(detections, start=1):
				lines.append(
					f"{index}. class_id={detection['class_id']} conf={float(detection['confidence']):.3f} "
					f"box={list(map(lambda value: round(float(value), 1), detection['box']))}"
				)
		else:
			lines.append("未检测到目标。")
		self.image_result_text.setPlainText("\n".join(lines))
		self.statusBar().showMessage("图片预测完成", 4000)

	def _browse_video_file(self) -> None:
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"选择视频",
			str(_repo_root() / "evaluate" / "input_video"),
			"视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;所有文件 (*.*)",
		)
		if file_path:
			self.video_path_edit.setText(file_path)

	def _open_video(self) -> None:
		video_path = self.video_path_edit.text().strip()
		if not video_path:
			self._show_warning("请先选择视频文件")
			return
		self.video_timer.stop()
		self.video_frame_pending = False
		self.video_open_button.setEnabled(False)
		self.video_preview_label.clear_preview("正在加载视频...")
		self._set_video_status(mode="加载中", frame="-", detections="-", fps="-", extra="-")
		self.statusBar().showMessage("正在加载视频...")
		self.open_video_requested.emit(
			video_path,
			self.video_conf_spin.value(),
			None,
		)

	def _on_video_loaded(self, payload: object) -> None:
		self.video_open_button.setEnabled(True)
		self.video_export_button.setEnabled(True)
		self.video_play_button.setEnabled(True)
		self.video_play_button.setText("播放")
		self.video_metadata = payload if isinstance(payload, dict) else None
		self.video_slider.setEnabled(True)
		self.video_slider.setValue(0)

		if self.video_metadata:
			frame_count = int(self.video_metadata.get("frame_count", 0))
			self.video_slider.setRange(0, max(frame_count - 1, 0))
			self.video_meta_label.setText(
				f"{self.video_metadata['frame_width']}x{self.video_metadata['frame_height']} | "
				f"FPS {float(self.video_metadata['input_fps']):.2f} | "
				f"共 {frame_count} 帧"
			)
			self.video_time_label.setText(
				f"00:00 / {_format_seconds(float(self.video_metadata['duration_sec']))}"
			)
		self._set_video_status(
			mode="已加载",
			frame="0",
			detections="0",
			fps="-",
			extra=Path(str(payload["video_path"])).name,
		)
		self.statusBar().showMessage("视频加载完成", 4000)
		self._request_single_video_frame()

	def _toggle_video_playback(self) -> None:
		if self.video_metadata is None:
			self._show_warning("请先加载视频")
			return
		if self.video_timer.isActive():
			self.video_timer.stop()
			self.video_play_button.setText("播放")
			return

		interval = 33
		if self.video_metadata:
			fps = float(self.video_metadata.get("input_fps", 0.0))
			if fps > 0:
				interval = max(1, int(round(1000 / fps)))
		self.video_timer.start(interval)
		self.video_play_button.setText("暂停")
		self._request_next_video_frame()

	def _request_single_video_frame(self) -> None:
		self.video_timer.stop()
		self.video_play_button.setText("播放")
		self._request_next_video_frame()

	def _request_next_video_frame(self) -> None:
		if self.video_frame_pending or self.video_metadata is None:
			return
		self.video_frame_pending = True
		self.read_video_frame_requested.emit(self.video_predict_checkbox.isChecked())

	def _on_video_frame_ready(self, payload: object) -> None:
		self.video_frame_pending = False
		if not isinstance(payload, dict):
			return

		self.video_preview_label.set_preview_pixmap(_cv_to_pixmap(payload["rendered_frame"]))
		frame_index = int(payload.get("frame_index", 0))
		timestamp_sec = float(payload.get("timestamp_sec", 0.0))
		video_metadata = payload.get("video_metadata") or self.video_metadata or {}
		duration_sec = float(video_metadata.get("duration_sec", 0.0))

		if not self.video_slider_active:
			self.video_slider.setValue(frame_index)
		self.video_time_label.setText(
			f"{_format_seconds(timestamp_sec)} / {_format_seconds(duration_sec)}"
		)

		detections = payload.get("detections", [])
		mode_text = "预测中" if payload.get("predict_enabled") else "仅播放"
		fps_text = f"{float(payload['fps']):.2f}" if payload.get("fps") is not None else "-"
		self._set_video_status(
			mode=mode_text,
			frame=f"{frame_index}",
			detections=str(len(detections)),
			fps=fps_text,
			extra=f"{timestamp_sec:.2f}s",
		)

	def _on_video_playback_finished(self) -> None:
		self.video_frame_pending = False
		self.video_timer.stop()
		self.video_play_button.setText("播放")
		self._set_video_status(mode="播放完成")
		self.statusBar().showMessage("视频播放完成", 4000)

	def _on_video_slider_pressed(self) -> None:
		self.video_slider_active = True

	def _on_video_slider_released(self) -> None:
		self.video_slider_active = False
		self.video_timer.stop()
		self.video_play_button.setText("播放")
		self.video_frame_pending = False
		self.seek_video_frame_requested.emit(self.video_slider.value())
		self._request_single_video_frame()

	def _export_video_prediction(self) -> None:
		video_path = self.video_path_edit.text().strip()
		if not video_path:
			self._show_warning("请先选择视频文件")
			return
		output_dir = _repo_root() / "evaluate" / "output_video"
		self.video_export_button.setEnabled(False)
		self.statusBar().showMessage("正在导出预测视频...")
		self.export_video_prediction_requested.emit(
			video_path,
			str(output_dir),
			self.video_conf_spin.value(),
			None,
		)

	def _on_video_export_ready(self, payload: object) -> None:
		self.video_export_button.setEnabled(True)
		if not isinstance(payload, dict):
			return
		self._set_video_status(
			mode="导出完成",
			frame=str(payload["processed_frames"]),
			detections="-",
			fps=f"{float(payload['average_fps']):.2f}",
			extra=Path(payload["output_video_path"]).name,
		)
		self.statusBar().showMessage("预测视频导出完成", 5000)

	def _browse_dataset_dir(self) -> None:
		dir_path = QFileDialog.getExistingDirectory(
			self,
			"选择数据集目录",
			self.dataset_dir_edit.text() or str(_repo_root() / "evaluate" / "input"),
		)
		if dir_path:
			self.dataset_dir_edit.setText(dir_path)

	def _browse_dataset_output_dir(self) -> None:
		dir_path = QFileDialog.getExistingDirectory(
			self,
			"选择输出目录",
			self.dataset_output_dir_edit.text() or str(_repo_root() / "evaluate" / "output"),
		)
		if dir_path:
			self.dataset_output_dir_edit.setText(dir_path)

	def _browse_dataset_classes_file(self) -> None:
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"选择类别文件",
			self.dataset_dir_edit.text() or str(_repo_root() / "evaluate" / "input"),
			"类别文件 (*.txt *.yaml *.yml);;所有文件 (*.*)",
		)
		if file_path:
			self.dataset_classes_edit.setText(file_path)

	def _check_dataset(self) -> None:
		dataset_dir = self.dataset_dir_edit.text().strip()
		if not dataset_dir:
			self._show_warning("请先选择数据集目录")
			return
		self.dataset_check_button.setEnabled(False)
		self.statusBar().showMessage("正在检查数据集...")
		self.check_dataset_requested.emit(dataset_dir)

	def _on_dataset_check_ready(self, payload: object) -> None:
		self.dataset_check_button.setEnabled(True)
		if not isinstance(payload, dict):
			return
		self.dataset_summary_text.setPlainText(payload.get("report_text", ""))
		self.dataset_json_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
		self._show_dataset_histograms(payload)
		self.statusBar().showMessage("数据集检查完成", 5000)

	def _show_dataset_histograms(self, payload: dict[str, Any]) -> None:
		self.histogram_dialogs = []
		dialog_specs = [
			("每类图片数量直方图", payload.get("image_histogram_path")),
			("每类检测框数量直方图", payload.get("box_histogram_path")),
		]
		for title, image_path in dialog_specs:
			if not image_path:
				continue
			dialog = HistogramDialog(title, image_path, self)
			dialog.show()
			dialog.raise_()
			dialog.activateWindow()
			self.histogram_dialogs.append(dialog)

	def _evaluate_dataset(self) -> None:
		dataset_dir = self.dataset_dir_edit.text().strip()
		output_dir = self.dataset_output_dir_edit.text().strip()
		if not dataset_dir or not output_dir:
			self._show_warning("请先填写数据集目录和输出目录")
			return
		self.dataset_evaluate_button.setEnabled(False)
		self.statusBar().showMessage("正在进行模型评估...")
		self.evaluate_dataset_requested.emit(
			dataset_dir,
			output_dir,
			self.eval_conf_spin.value(),
			self.eval_iou_spin.value(),
			self.dataset_classes_edit.text().strip() or None,
		)

	def _on_dataset_evaluation_ready(self, payload: object) -> None:
		self.dataset_evaluate_button.setEnabled(True)
		if not isinstance(payload, dict):
			return

		lines = [
			f"模型: {payload['model_path']}",
			f"评测集目录: {payload['input_dir']}",
			f"输出目录: {payload['output_dir']}",
			f"图像数量: {payload['image_count']}",
			f"Precision: {float(payload['precision']):.4f}",
			f"Recall: {float(payload['recall']):.4f}",
			f"80% 召回达标: {'是' if payload['recall_reached_80_percent'] else '否'}",
			f"mAP50: {float(payload['map50']):.4f}",
			f"误检率: {float(payload['false_detection_rate']):.4f}",
			f"漏检率: {float(payload['miss_rate']):.4f}",
		]

		per_class_metrics = payload.get("per_class_metrics", [])
		if per_class_metrics:
			lines.extend(
				[
					"",
					"各类别 AP50:",
					f"{'Class':<20} {'ID':>4} {'AP50':>8} {'Precision':>10} {'Recall':>8} {'GT':>6} {'Prediction':>10}",
					"-" * 78,
				]
			)
			for metric in per_class_metrics:
				lines.append(
					f"{metric['class_name']:<20.20} "
					f"{metric['class_id']:>4} "
					f"{metric['ap50']:>8.4f} "
					f"{metric['precision']:>10.4f} "
					f"{metric['recall']:>8.4f} "
					f"{metric['ground_truth_count']:>6} "
					f"{metric['prediction_count']:>10}"
				)

		most_fp = payload.get("most_false_positive_class")
		most_fn = payload.get("most_false_negative_class")
		lines.append("")
		if most_fp:
			lines.append(
				f"误检最多类别: {most_fp['class_name']} (id={most_fp['class_id']}, count={most_fp['count']})"
			)
		else:
			lines.append("误检最多类别: 无")
		if most_fn:
			lines.append(
				f"漏检最多类别: {most_fn['class_name']} (id={most_fn['class_id']}, count={most_fn['count']})"
			)
		else:
			lines.append("漏检最多类别: 无")

		lines.extend(
			[
				"",
				"输出路径提示:",
				f"正确样本叠加框目录: {Path(payload['output_dir']) / 'visualizations' / 'combined'}",
				f"正确样本预测框目录: {Path(payload['output_dir']) / 'visualizations' / 'predictions_only'}",
				f"正确样本基准框目录: {Path(payload['output_dir']) / 'visualizations' / 'ground_truths_only'}",
				f"误检样本叠加框目录: {Path(payload['output_dir']) / 'errors' / 'false_positives' / 'combined'}",
				f"误检样本预测框目录: {Path(payload['output_dir']) / 'errors' / 'false_positives' / 'predictions_only'}",
				f"误检样本基准框目录: {Path(payload['output_dir']) / 'errors' / 'false_positives' / 'ground_truths_only'}",
				f"漏检样本叠加框目录: {Path(payload['output_dir']) / 'errors' / 'false_negatives' / 'combined'}",
				f"漏检样本预测框目录: {Path(payload['output_dir']) / 'errors' / 'false_negatives' / 'predictions_only'}",
				f"漏检样本基准框目录: {Path(payload['output_dir']) / 'errors' / 'false_negatives' / 'ground_truths_only'}",
				f"记录目录: {Path(payload['output_dir']) / 'records'}",
			]
		)

		self.dataset_summary_text.setPlainText("\n".join(lines))
		self.dataset_json_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
		self.statusBar().showMessage("模型评估完成", 5000)

	def _on_video_closed(self) -> None:
		self.video_metadata = None
		self.video_timer.stop()
		self.video_play_button.setText("播放")
		self.video_play_button.setEnabled(False)
		self.video_export_button.setEnabled(False)
		self.video_slider.setEnabled(False)
		self.video_preview_label.clear_preview("视频已关闭")
		self._set_video_status(mode="未加载", frame="-", detections="-", fps="-", extra="-")

	def _on_backend_error(self, task: str, message: str) -> None:
		if task == "predict_image":
			self.image_predict_button.setEnabled(True)
		elif task == "check_dataset":
			self.dataset_check_button.setEnabled(True)
		elif task == "evaluate_dataset":
			self.dataset_evaluate_button.setEnabled(True)
		elif task == "open_video":
			self.video_open_button.setEnabled(True)
			self.video_preview_label.clear_preview("视频加载失败")
			self._set_video_status(mode="加载失败")
		elif task == "export_video_prediction":
			self.video_export_button.setEnabled(True)
			self._set_video_status(mode="导出失败")
		elif task == "read_video_frame":
			self.video_frame_pending = False
			self.video_timer.stop()
			self.video_play_button.setText("播放")
			self._set_video_status(mode="播放失败")

		self.statusBar().showMessage(f"{task} 失败: {message}", 6000)
		self._show_warning(f"{task} 失败\n{message}\n\n日志文件: {LOG_PATH}")

	def _show_warning(self, message: str) -> None:
		QMessageBox.warning(self, "提示", message)

	def closeEvent(self, event) -> None:
		self.video_timer.stop()
		self.close_video_requested.emit()
		self.backend_thread.quit()
		self.backend_thread.wait()
		super().closeEvent(event)


def main() -> None:
	_setup_logging()
	sys.excepthook = _log_uncaught_exception
	threading.excepthook = _log_thread_exception
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec())
