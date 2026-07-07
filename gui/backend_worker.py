from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from PySide6.QtCore import QObject, Signal, Slot


def _get_runtime_providers() -> list[str]:
	"""根据当前环境返回可用的 ONNX Runtime 后端。"""
	available_providers = set(ort.get_available_providers())
	preferred_providers = [
		"DmlExecutionProvider",
		"CUDAExecutionProvider",
		"CPUExecutionProvider",
	]
	return [
		provider for provider in preferred_providers if provider in available_providers
	] or ["CPUExecutionProvider"]


class BackendWorker(QObject):
	"""处理模型相关命令的后台工作对象。"""

	status_changed = Signal(str, str)
	model_changed = Signal(object)
	command_finished = Signal(str, bool, str)

	def __init__(self) -> None:
		super().__init__()
		self.current_model_path: Path | None = None
		self.current_model_handle: ort.InferenceSession | None = None

	@Slot(str)
	def load_model(self, model_path: str) -> None:
		"""加载指定模型。"""
		model_file = Path(model_path).expanduser().resolve()
		self.status_changed.emit("loading", f"模型加载中: {model_file.name}")
		try:
			if model_file.suffix.lower() != ".onnx":
				raise ValueError("当前仅支持直接加载 .onnx 模型")

			model_handle = ort.InferenceSession(
				str(model_file),
				providers=_get_runtime_providers(),
			)
			self.current_model_path = model_file
			self.current_model_handle = model_handle
			self.status_changed.emit("loaded", f"已加载模型: {model_file.name}")
			self.model_changed.emit(str(model_file))
			self.command_finished.emit("load_model", True, str(model_file))
		except Exception as exc:
			self.current_model_path = None
			self.current_model_handle = None
			self.status_changed.emit("error", f"模型加载失败: {exc}")
			self.model_changed.emit(None)
			self.command_finished.emit("load_model", False, str(exc))

	@Slot()
	def unload_model(self) -> None:
		"""卸载当前模型。"""
		model_name = self.current_model_path.name if self.current_model_path else ""
		self.current_model_handle = None
		self.current_model_path = None
		self.status_changed.emit("idle", "未加载模型")
		self.model_changed.emit(None)
		self.command_finished.emit("unload_model", True, model_name)

	@Slot(str)
	def switch_model(self, model_path: str) -> None:
		"""切换到新的模型。"""
		self.load_model(model_path)
		self.command_finished.emit("switch_model", self.current_model_path is not None, model_path)
