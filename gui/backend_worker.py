from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import onnxruntime as ort
from PySide6.QtCore import QObject, Signal, Slot

from scripts.check_dataset import _format_report, _save_cli_outputs, check_dataset
from scripts.evaluate_dataset import evaluate_dataset
from scripts.predict import _load_onnx_model, predict_image
from scripts.predict_video import (
	_load_class_names,
	_read_video_metadata,
	_resolve_input_video,
	predict_video,
	render_detections,
)

logger = logging.getLogger(__name__)


class BackendWorker(QObject):
	"""处理模型、预测、视频和评测相关命令的后台工作对象。"""

	status_changed = Signal(str, str)
	model_changed = Signal(object)
	command_finished = Signal(str, bool, str)
	error_occurred = Signal(str, str)

	image_prediction_ready = Signal(object)
	dataset_check_ready = Signal(object)
	dataset_evaluation_ready = Signal(object)

	video_loaded = Signal(object)
	video_closed = Signal()
	video_frame_ready = Signal(object)
	video_export_ready = Signal(object)
	video_playback_finished = Signal()

	def __init__(self) -> None:
		super().__init__()
		self.current_model_path: Path | None = None
		self.runtime_model_path: Path | None = None
		self.current_model_handle: ort.InferenceSession | None = None

		self.video_file: Path | None = None
		self.video_capture: cv2.VideoCapture | None = None
		self.video_metadata: dict[str, Any] | None = None
		self.video_class_names: dict[int, str] = {}
		self.video_conf_threshold = 0.25
		self.video_smoothed_fps = 0.0

	def _emit_error(self, task: str, exc: Exception) -> None:
		message = str(exc)
		logger.exception("Backend task failed: %s", task, exc_info=exc)
		self.error_occurred.emit(task, message)
		self.command_finished.emit(task, False, message)

	def _release_video(self) -> None:
		if self.video_capture is not None:
			self.video_capture.release()
		self.video_capture = None
		self.video_file = None
		self.video_metadata = None
		self.video_class_names = {}
		self.video_smoothed_fps = 0.0

	def _require_model(self) -> ort.InferenceSession:
		if self.current_model_handle is None:
			raise RuntimeError("请先加载模型")
		return self.current_model_handle

	def _models_dir(self) -> Path:
		models_dir = Path(__file__).resolve().parents[1] / "models"
		models_dir.mkdir(parents=True, exist_ok=True)
		return models_dir

	def _build_converted_model_path(self, pt_file: Path) -> Path:
		"""为 `.pt` 转出的 ONNX 生成稳定输出路径。"""
		models_dir = self._models_dir()
		base_name = f"{pt_file.stem}_converted.onnx"
		candidate = models_dir / base_name
		if candidate.resolve() != pt_file.with_suffix(".onnx").resolve():
			return candidate
		return models_dir / f"{pt_file.stem}_from_pt.onnx"

	def _load_model_file(self, model_file: Path) -> tuple[ort.InferenceSession, Path]:
		if model_file.suffix.lower() == ".onnx":
			runtime_path = model_file
		elif model_file.suffix.lower() == ".pt":
			from scripts.model_convert import convert_pt_to_onnx

			runtime_path = self._build_converted_model_path(model_file)
			runtime_path = convert_pt_to_onnx(
				pt_path=model_file,
				onnx_path=runtime_path,
			)
		else:
			raise ValueError("当前仅支持加载 .onnx 或 .pt 模型")

		return _load_onnx_model(runtime_path), Path(runtime_path).resolve()

	@Slot(str)
	def load_model(self, model_path: str) -> None:
		model_file = Path(model_path).expanduser().resolve()
		if model_file.suffix.lower() == ".pt":
			self.status_changed.emit("loading", f"正在转换并加载模型: {model_file.name}")
		else:
			self.status_changed.emit("loading", f"模型加载中: {model_file.name}")
		try:
			model_handle, runtime_path = self._load_model_file(model_file)
			self.current_model_path = model_file
			self.runtime_model_path = runtime_path
			self.current_model_handle = model_handle

			if model_file == runtime_path:
				message = f"已加载模型: {model_file.name}"
			else:
				message = f"已加载模型: {model_file.name}，已转换到 models/{runtime_path.name}"

			self.status_changed.emit("loaded", message)
			self.model_changed.emit(
				{
					"model_path": str(model_file),
					"runtime_model_path": str(runtime_path),
				}
			)
			self.command_finished.emit("load_model", True, str(model_file))
		except Exception as exc:
			self.current_model_path = None
			self.runtime_model_path = None
			self.current_model_handle = None
			self.status_changed.emit("error", f"模型加载失败: {exc}")
			self.model_changed.emit(None)
			self._emit_error("load_model", exc)

	@Slot()
	def unload_model(self) -> None:
		model_name = self.current_model_path.name if self.current_model_path else ""
		self.current_model_handle = None
		self.current_model_path = None
		self.runtime_model_path = None
		self.status_changed.emit("idle", "未加载模型")
		self.model_changed.emit(None)
		self.command_finished.emit("unload_model", True, model_name)

	@Slot(str)
	def switch_model(self, model_path: str) -> None:
		self.load_model(model_path)
		self.command_finished.emit("switch_model", self.current_model_path is not None, model_path)

	@Slot(str, float)
	def predict_image_file(self, image_path: str, conf_threshold: float) -> None:
		try:
			model = self._require_model()
			image_file = Path(image_path).expanduser().resolve()
			image = cv2.imread(str(image_file))
			if image is None:
				raise FileNotFoundError(f"无法读取图像文件: {image_file}")

			detections = predict_image(model, image, conf_threshold=conf_threshold)
			rendered = render_detections(image, detections)
			self.image_prediction_ready.emit(
				{
					"image_path": str(image_file),
					"detections": detections,
					"rendered_image": rendered,
					"original_image": image,
					"confidence_threshold": conf_threshold,
				}
			)
			self.command_finished.emit("predict_image", True, str(image_file))
		except Exception as exc:
			self._emit_error("predict_image", exc)

	@Slot(str)
	def check_dataset_dir(self, dataset_dir: str) -> None:
		try:
			result = check_dataset(dataset_dir)
			output_paths = _save_cli_outputs(result)
			payload = {
				**result,
				**output_paths,
				"report_text": _format_report({**result, **output_paths}),
			}
			self.dataset_check_ready.emit(payload)
			self.command_finished.emit("check_dataset", True, str(dataset_dir))
		except Exception as exc:
			self._emit_error("check_dataset", exc)

	@Slot(str, str, float, float, object)
	def evaluate_dataset_dir(
		self,
		dataset_dir: str,
		output_dir: str,
		conf_threshold: float,
		iou_threshold: float,
		classes_path: object,
	) -> None:
		try:
			model_path = self.current_model_path or self.runtime_model_path
			if model_path is None:
				raise RuntimeError("请先加载模型")

			result = evaluate_dataset(
				model_path=str(self.runtime_model_path or model_path),
				input_dir=dataset_dir,
				output_dir=output_dir,
				conf_threshold=conf_threshold,
				iou_threshold=iou_threshold,
				classes_path=str(classes_path) if classes_path else None,
			)
			self.dataset_evaluation_ready.emit(result)
			self.command_finished.emit("evaluate_dataset", True, str(dataset_dir))
		except Exception as exc:
			self._emit_error("evaluate_dataset", exc)

	@Slot(str, float, object)
	def open_video(self, video_path: str, conf_threshold: float, classes_path: object) -> None:
		try:
			self._release_video()
			video_file = _resolve_input_video(video_path)
			capture = cv2.VideoCapture(str(video_file))
			if not capture.isOpened():
				raise FileNotFoundError(f"无法打开视频文件: {video_file}")

			metadata = _read_video_metadata(video_file)
			self.video_file = video_file
			self.video_capture = capture
			self.video_metadata = {
				"video_path": metadata.video_path,
				"frame_width": metadata.frame_width,
				"frame_height": metadata.frame_height,
				"input_fps": metadata.input_fps,
				"frame_count": metadata.frame_count,
				"duration_sec": metadata.duration_sec,
			}
			self.video_class_names = _load_class_names(video_file, str(classes_path) if classes_path else None)
			self.video_conf_threshold = conf_threshold
			self.video_smoothed_fps = 0.0

			self.video_loaded.emit(
				{
					**self.video_metadata,
					"class_names": self.video_class_names,
				}
			)
			self.command_finished.emit("open_video", True, str(video_file))
		except Exception as exc:
			self._release_video()
			self._emit_error("open_video", exc)

	@Slot()
	def close_video(self) -> None:
		self._release_video()
		self.video_closed.emit()
		self.command_finished.emit("close_video", True, "")

	@Slot(bool)
	def read_video_frame(self, predict_enabled: bool) -> None:
		try:
			if self.video_capture is None or self.video_file is None or self.video_metadata is None:
				raise RuntimeError("请先加载视频")

			ok, frame = self.video_capture.read()
			if not ok:
				self.video_playback_finished.emit()
				self.command_finished.emit("read_video_frame", True, "finished")
				return

			frame_index = max(int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
			timestamp_sec = self.video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
			detections: list[dict[str, Any]] = []
			rendered_frame = frame
			fps_value: float | None = None

			if predict_enabled:
				model = self._require_model()
				frame_start = time.perf_counter()
				detections = predict_image(
					model=model,
					image=frame,
					conf_threshold=self.video_conf_threshold,
				)
				frame_elapsed = max(time.perf_counter() - frame_start, 1e-6)
				current_fps = 1.0 / frame_elapsed
				self.video_smoothed_fps = (
					current_fps
					if self.video_smoothed_fps == 0.0
					else self.video_smoothed_fps * 0.9 + current_fps * 0.1
				)
				fps_value = self.video_smoothed_fps
				rendered_frame = render_detections(
					frame,
					detections,
					class_names=self.video_class_names,
					fps=fps_value,
				)
			else:
				self.video_smoothed_fps = 0.0

			self.video_frame_ready.emit(
				{
					"frame_index": frame_index,
					"timestamp_sec": timestamp_sec,
					"frame": frame,
					"rendered_frame": rendered_frame,
					"detections": detections,
					"fps": fps_value,
					"predict_enabled": predict_enabled,
					"class_names": self.video_class_names,
					"video_metadata": self.video_metadata,
				}
			)
			self.command_finished.emit("read_video_frame", True, str(frame_index))
		except Exception as exc:
			self._emit_error("read_video_frame", exc)

	@Slot(int)
	def seek_video_frame(self, frame_index: int) -> None:
		try:
			if self.video_capture is None or self.video_metadata is None:
				raise RuntimeError("请先加载视频")
			max_frame_index = max(int(self.video_metadata["frame_count"]) - 1, 0)
			clamped_index = min(max(frame_index, 0), max_frame_index)
			self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_index)
			self.video_smoothed_fps = 0.0
			self.command_finished.emit("seek_video_frame", True, str(clamped_index))
		except Exception as exc:
			self._emit_error("seek_video_frame", exc)

	@Slot(str, str, float, object)
	def export_video_prediction(
		self,
		video_path: str,
		output_path: str,
		conf_threshold: float,
		classes_path: object,
	) -> None:
		try:
			model_path = self.runtime_model_path or self.current_model_path
			if model_path is None:
				raise RuntimeError("请先加载模型")

			summary = predict_video(
				model_path=str(model_path),
				video_path=video_path,
				output_path=output_path,
				conf_threshold=conf_threshold,
				classes_path=str(classes_path) if classes_path else None,
				show=False,
			)
			self.video_export_ready.emit(summary)
			self.command_finished.emit("export_video_prediction", True, summary["output_video_path"])
		except Exception as exc:
			self._emit_error("export_video_prediction", exc)
