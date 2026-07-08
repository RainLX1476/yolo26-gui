from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import onnxruntime as ort

try:
	from scripts.predict import _load_onnx_model, predict_image
except ModuleNotFoundError:
	from predict import _load_onnx_model, predict_image

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

__all__ = [
	"VideoPredictionSession",
	"iter_video_predictions",
	"predict_video",
	"render_detections",
]


@dataclass
class VideoMetadata:
	video_path: str
	frame_width: int
	frame_height: int
	input_fps: float
	frame_count: int
	duration_sec: float


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="对视频执行交通标志牌检测并导出结果视频")
	parser.add_argument("--onnx", required=True, help="待推理的 ONNX 模型路径")
	parser.add_argument(
		"--input",
		default="evaluate/input_video",
		help="输入视频路径或包含单个视频的目录，默认 evaluate/input_video",
	)
	parser.add_argument(
		"--output",
		default="evaluate/output_video",
		help="输出视频路径或输出目录，默认 evaluate/output_video",
	)
	parser.add_argument(
		"--classes",
		help="可选类别名文件，默认自动查找 input/classes.txt 或 evaluate/input/classes.txt",
	)
	parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="仅保存输出视频，不显示实时窗口",
	)
	return parser.parse_args()


def _ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def _find_videos(input_path: Path) -> list[Path]:
	if input_path.is_file():
		return [input_path]
	return sorted(
		path
		for path in input_path.rglob("*")
		if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
	)


def _resolve_input_video(input_path: str | Path) -> Path:
	video_input = Path(input_path).expanduser().resolve()
	if not video_input.exists():
		raise FileNotFoundError(f"输入视频路径不存在: {video_input}")

	videos = _find_videos(video_input)
	if not videos:
		raise FileNotFoundError(f"未找到视频文件: {video_input}")
	if len(videos) > 1:
		raise ValueError(
			f"找到多个视频文件，请明确指定其中一个: {[str(path) for path in videos]}"
		)
	return videos[0]


def _resolve_output_video(output_path: str | Path, input_video: Path) -> Path:
	output = Path(output_path).expanduser().resolve()
	if output.suffix.lower() in VIDEO_SUFFIXES:
		_ensure_dir(output.parent)
		return output

	_ensure_dir(output)
	return output / f"{input_video.stem}_pred.mp4"


def _load_class_names(video_path: Path, classes_arg: str | None) -> dict[int, str]:
	candidate_paths: list[Path] = []
	if classes_arg:
		candidate_paths.append(Path(classes_arg).expanduser().resolve())

	candidate_paths.extend(
		[
			video_path.parent / "classes.txt",
			video_path.parent.parent / "classes.txt",
			video_path.parents[1] / "input" / "classes.txt" if len(video_path.parents) > 1 else video_path.parent / "classes.txt",
		]
	)

	seen: set[Path] = set()
	for candidate in candidate_paths:
		if candidate in seen:
			continue
		seen.add(candidate)
		if not candidate.exists():
			continue
		names = [
			line.strip()
			for line in candidate.read_text(encoding="utf-8").splitlines()
			if line.strip()
		]
		return {index: name for index, name in enumerate(names)}
	return {}


def _class_name(class_id: int, class_names: dict[int, str]) -> str:
	return class_names.get(class_id, f"class_{class_id}")


def _open_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
	fourcc_candidates = ("mp4v", "avc1", "XVID")
	valid_fps = fps if fps > 0 else 25.0

	for fourcc_name in fourcc_candidates:
		writer = cv2.VideoWriter(
			str(output_path),
			cv2.VideoWriter_fourcc(*fourcc_name),
			valid_fps,
			(width, height),
		)
		if writer.isOpened():
			return writer
		writer.release()

	raise RuntimeError(f"无法创建输出视频: {output_path}")


def _read_video_metadata(video_file: Path) -> VideoMetadata:
	capture = cv2.VideoCapture(str(video_file))
	if not capture.isOpened():
		raise FileNotFoundError(f"无法打开视频文件: {video_file}")

	frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	input_fps = float(capture.get(cv2.CAP_PROP_FPS))
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	capture.release()

	valid_fps = input_fps if input_fps > 0 else 25.0
	duration_sec = frame_count / valid_fps if frame_count > 0 else 0.0
	return VideoMetadata(
		video_path=str(video_file),
		frame_width=frame_width,
		frame_height=frame_height,
		input_fps=valid_fps,
		frame_count=frame_count,
		duration_sec=duration_sec,
	)


def render_detections(
	frame: np.ndarray,
	detections: list[dict[str, object]],
	class_names: dict[int, str] | None = None,
	fps: float | None = None,
) -> np.ndarray:
	class_names = class_names or {}
	canvas = frame.copy()

	for detection in detections:
		class_id = int(detection["class_id"])
		confidence = float(detection["confidence"])
		x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
		label = f"{_class_name(class_id, class_names)} {confidence:.2f}"

		cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
		(text_width, text_height), baseline = cv2.getTextSize(
			label,
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			2,
		)
		label_top = max(0, y1 - text_height - baseline - 8)
		label_bottom = label_top + text_height + baseline + 8
		label_right = min(canvas.shape[1], x1 + text_width + 10)
		cv2.rectangle(canvas, (x1, label_top), (label_right, label_bottom), (0, 220, 0), -1)
		cv2.putText(
			canvas,
			label,
			(x1 + 5, label_bottom - baseline - 4),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(20, 20, 20),
			2,
			cv2.LINE_AA,
		)

	if fps is not None:
		fps_text = f"FPS: {fps:.2f}"
		cv2.putText(
			canvas,
			fps_text,
			(20, 36),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.9,
			(0, 255, 255),
			2,
			cv2.LINE_AA,
		)

	return canvas


class VideoPredictionSession:
	"""支持顺序播放和随机跳转的视频推理会话。

	后续 GUI 可直接复用该对象来实现播放、暂停、拖动进度条、跳帧等能力。
	"""

	def __init__(
		self,
		model: ort.InferenceSession,
		video_path: str | Path,
		conf_threshold: float = 0.25,
		classes_path: str | None = None,
	) -> None:
		self.model = model
		self.video_file = _resolve_input_video(video_path)
		self.conf_threshold = conf_threshold
		self.class_names = _load_class_names(self.video_file, classes_path)
		self.metadata = _read_video_metadata(self.video_file)
		self.capture = cv2.VideoCapture(str(self.video_file))
		if not self.capture.isOpened():
			raise FileNotFoundError(f"无法打开视频文件: {self.video_file}")
		self.smoothed_fps = 0.0

	def close(self) -> None:
		if self.capture.isOpened():
			self.capture.release()

	def seek_frame(self, frame_index: int) -> int:
		max_frame_index = max(self.metadata.frame_count - 1, 0)
		clamped_index = min(max(frame_index, 0), max_frame_index)
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_index)
		self.smoothed_fps = 0.0
		return clamped_index

	def seek_time(self, timestamp_sec: float) -> int:
		if self.metadata.input_fps <= 0:
			return self.seek_frame(0)
		target_frame = int(round(max(timestamp_sec, 0.0) * self.metadata.input_fps))
		return self.seek_frame(target_frame)

	def tell_frame(self) -> int:
		return int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

	def tell_time(self) -> float:
		return self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

	def read_next(self) -> dict[str, Any] | None:
		ok, frame = self.capture.read()
		if not ok:
			return None

		frame_index = max(int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)
		frame_start = time.perf_counter()
		detections = predict_image(
			model=self.model,
			image=frame,
			conf_threshold=self.conf_threshold,
		)
		frame_elapsed = max(time.perf_counter() - frame_start, 1e-6)
		current_fps = 1.0 / frame_elapsed
		self.smoothed_fps = (
			current_fps
			if self.smoothed_fps == 0.0
			else self.smoothed_fps * 0.9 + current_fps * 0.1
		)

		return {
			"frame_index": frame_index,
			"frame": frame,
			"rendered_frame": render_detections(
				frame,
				detections,
				class_names=self.class_names,
				fps=self.smoothed_fps,
			),
			"detections": detections,
			"fps": self.smoothed_fps,
			"timestamp_sec": self.tell_time(),
			"class_names": self.class_names,
			"video_metadata": {
				"video_path": self.metadata.video_path,
				"frame_width": self.metadata.frame_width,
				"frame_height": self.metadata.frame_height,
				"input_fps": self.metadata.input_fps,
				"frame_count": self.metadata.frame_count,
				"duration_sec": self.metadata.duration_sec,
			},
		}


def iter_video_predictions(
	model: ort.InferenceSession,
	video_path: str | Path,
	conf_threshold: float = 0.25,
	classes_path: str | None = None,
) -> Iterator[dict[str, Any]]:
	session = VideoPredictionSession(
		model=model,
		video_path=video_path,
		conf_threshold=conf_threshold,
		classes_path=classes_path,
	)
	frame_index = 0

	try:
		while True:
			result = session.read_next()
			if result is None:
				break
			yield result
			frame_index += 1
	finally:
		session.close()

		if frame_index == 0:
			raise ValueError(f"视频中没有可读取的帧: {session.video_file}")


def predict_video(
	model_path: str | Path,
	video_path: str | Path = "evaluate/input_video",
	output_path: str | Path = "evaluate/output_video",
	conf_threshold: float = 0.25,
	classes_path: str | None = None,
	show: bool = True,
	window_name: str = "Video Detection",
) -> dict[str, Any]:
	video_file = _resolve_input_video(video_path)
	output_file = _resolve_output_video(output_path, video_file)
	model = _load_onnx_model(model_path)
	class_names = _load_class_names(video_file, classes_path)
	metadata = _read_video_metadata(video_file)

	writer = _open_video_writer(
		output_file,
		metadata.frame_width,
		metadata.frame_height,
		metadata.input_fps,
	)
	start_time = time.perf_counter()
	processed_frames = 0
	last_fps = 0.0
	frame_generator = iter_video_predictions(
		model=model,
		video_path=video_file,
		conf_threshold=conf_threshold,
		classes_path=classes_path,
	)

	try:
		for result in frame_generator:
			rendered_frame = result["rendered_frame"]
			writer.write(rendered_frame)
			processed_frames += 1
			last_fps = float(result["fps"])

			if show:
				cv2.imshow(window_name, rendered_frame)
				key = cv2.waitKey(1) & 0xFF
				if key in {27, ord("q"), ord("Q")}:
					break
	finally:
		frame_generator.close()
		writer.release()
		if show:
			cv2.destroyAllWindows()

	total_elapsed = max(time.perf_counter() - start_time, 1e-6)
	average_fps = processed_frames / total_elapsed

	return {
		"model_path": str(Path(model_path).expanduser().resolve()),
		"input_video_path": str(video_file),
		"output_video_path": str(output_file),
		"frame_width": metadata.frame_width,
		"frame_height": metadata.frame_height,
		"input_fps": metadata.input_fps,
		"frame_count": metadata.frame_count,
		"duration_sec": metadata.duration_sec,
		"processed_frames": processed_frames,
		"average_fps": float(average_fps),
		"last_smoothed_fps": float(last_fps),
		"class_names": class_names,
	}


def _print_summary(summary: dict[str, Any]) -> None:
	print(f"模型: {summary['model_path']}")
	print(f"输入视频: {summary['input_video_path']}")
	print(f"输出视频: {summary['output_video_path']}")
	print(f"视频尺寸: {summary['frame_width']}x{summary['frame_height']}")
	print(f"输入 FPS: {summary['input_fps']:.2f}")
	print(f"总帧数: {summary['frame_count']}")
	print(f"已处理帧数: {summary['processed_frames']}")
	print(f"平均推理 FPS: {summary['average_fps']:.2f}")


def main() -> None:
	args = _parse_args()
	summary = predict_video(
		model_path=args.onnx,
		video_path=args.input,
		output_path=args.output,
		conf_threshold=args.conf,
		classes_path=args.classes,
		show=not args.no_show,
	)
	_print_summary(summary)


if __name__ == "__main__":
	main()
