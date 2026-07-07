from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

__all__ = ["predict_image"]


def _load_onnx_model(model_path: str | Path) -> ort.InferenceSession:
	"""加载 ONNX 模型，返回可直接复用的模型句柄。"""
	model_file = Path(model_path).expanduser().resolve()
	return ort.InferenceSession(
		str(model_file),
		providers=["DmlExecutionProvider", "CPUExecutionProvider"],
	)


def _read_image(image: str | Path | np.ndarray) -> np.ndarray:
	"""读取图像；支持传入路径或已经加载好的图像数组。"""
	if isinstance(image, np.ndarray):
		return image

	image_file = Path(image).expanduser().resolve()
	image_data = cv2.imread(str(image_file))
	if image_data is None:
		raise FileNotFoundError(f"无法读取图像文件: {image_file}")
	return image_data


def _get_input_size(model: ort.InferenceSession) -> tuple[int, int]:
	"""从模型输入信息中读取高宽。"""
	input_shape = model.get_inputs()[0].shape
	input_height = int(input_shape[2])
	input_width = int(input_shape[3])
	return input_height, input_width


def _letterbox(
	image: np.ndarray,
	input_size: tuple[int, int],
	color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
	"""按比例缩放并补边，减少输入形变。"""
	input_height, input_width = input_size
	image_height, image_width = image.shape[:2]
	scale = min(input_width / image_width, input_height / image_height)

	resized_width = int(round(image_width * scale))
	resized_height = int(round(image_height * scale))
	resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

	pad_width = input_width - resized_width
	pad_height = input_height - resized_height
	pad_left = pad_width / 2
	pad_top = pad_height / 2

	top = int(round(pad_top - 0.1))
	bottom = int(round(pad_top + 0.1))
	left = int(round(pad_left - 0.1))
	right = int(round(pad_left + 0.1))

	letterboxed_image = cv2.copyMakeBorder(
		resized_image,
		top,
		bottom,
		left,
		right,
		cv2.BORDER_CONSTANT,
		value=color,
	)
	return letterboxed_image, scale, (pad_left, pad_top)


def _preprocess_image(
	image: np.ndarray,
	input_size: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[float, float]]:
	"""将图像整理为模型可用的输入张量。"""
	letterboxed_image, scale, pad = _letterbox(image, input_size)
	rgb_image = cv2.cvtColor(letterboxed_image, cv2.COLOR_BGR2RGB)
	input_tensor = rgb_image.astype(np.float32) / 255.0
	input_tensor = np.transpose(input_tensor, (2, 0, 1))
	return np.expand_dims(input_tensor, axis=0), scale, pad


def _normalize_predictions(output: np.ndarray) -> np.ndarray:
	"""将模型输出统一整理为 `num_boxes x num_attrs` 结构。"""
	predictions = np.squeeze(output)
	if predictions.ndim == 1:
		predictions = np.expand_dims(predictions, axis=0)
	if predictions.ndim != 2:
		raise ValueError(f"暂不支持的输出形状: {output.shape}")
	if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 128:
		predictions = predictions.T
	return predictions


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
	"""将中心点宽高框转换为左上右下框。"""
	converted = boxes.copy()
	converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
	converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
	converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
	converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
	return converted


def _scale_boxes(
	boxes: np.ndarray,
	image_shape: tuple[int, int],
	scale: float,
	pad: tuple[float, float],
) -> np.ndarray:
	"""将输入尺寸上的检测框还原到原图尺寸。"""
	scaled_boxes = boxes.copy()
	scaled_boxes[:, [0, 2]] -= pad[0]
	scaled_boxes[:, [1, 3]] -= pad[1]
	scaled_boxes /= scale

	image_height, image_width = image_shape
	scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, image_width)
	scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, image_height)
	return scaled_boxes


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
	"""计算一个框与多个框的 IoU。"""
	x1 = np.maximum(box[0], boxes[:, 0])
	y1 = np.maximum(box[1], boxes[:, 1])
	x2 = np.minimum(box[2], boxes[:, 2])
	y2 = np.minimum(box[3], boxes[:, 3])

	inter_width = np.maximum(0.0, x2 - x1)
	inter_height = np.maximum(0.0, y2 - y1)
	intersection = inter_width * inter_height

	box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
	boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
	union = box_area + boxes_area - intersection
	return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
	"""执行非极大值抑制。"""
	order = scores.argsort()[::-1]
	keep: list[int] = []

	while order.size > 0:
		current = int(order[0])
		keep.append(current)
		if order.size == 1:
			break

		ious = _compute_iou(boxes[current], boxes[order[1:]])
		remaining = np.where(ious < iou_threshold)[0]
		order = order[remaining + 1]

	return keep


def _extract_scores(
	predictions: np.ndarray,
	with_objectness: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""从预测结果中提取框、类别和置信度。"""
	if predictions.shape[1] < 6:
		raise ValueError(f"输出属性维度不足，无法解析检测结果: {predictions.shape}")

	boxes_xywh = predictions[:, :4]
	if with_objectness:
		objectness = predictions[:, 4]
		class_scores = predictions[:, 5:]
		class_ids = np.argmax(class_scores, axis=1)
		confidences = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
	else:
		class_scores = predictions[:, 4:]
		class_ids = np.argmax(class_scores, axis=1)
		confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

	return boxes_xywh, class_ids, confidences


def _parse_detections(
	output: np.ndarray,
	image_shape: tuple[int, int],
	scale: float,
	pad: tuple[float, float],
	conf_threshold: float,
	iou_threshold: float,
	with_objectness: bool,
) -> list[dict[str, object]]:
	"""将模型原始输出解析为最终检测结果列表。

	后续如果需要适配新的模型输出格式，优先修改这个函数。
	"""
	predictions = _normalize_predictions(output)
	boxes_xywh, class_ids, confidences = _extract_scores(predictions, with_objectness)
	valid_mask = confidences >= conf_threshold
	if not np.any(valid_mask):
		return []

	boxes_xywh = boxes_xywh[valid_mask]
	class_ids = class_ids[valid_mask]
	confidences = confidences[valid_mask]

	boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
	boxes_xyxy = _scale_boxes(boxes_xyxy, image_shape, scale, pad)

	results: list[dict[str, object]] = []
	for class_id in np.unique(class_ids):
		class_mask = class_ids == class_id
		class_boxes = boxes_xyxy[class_mask]
		class_scores = confidences[class_mask]
		keep_indices = _nms(class_boxes, class_scores, iou_threshold)

		for keep_index in keep_indices:
			box = class_boxes[keep_index]
			results.append(
				{
					"class_id": int(class_id),
					"confidence": float(class_scores[keep_index]),
					"box": [
						float(box[0]),
						float(box[1]),
						float(box[2]),
						float(box[3]),
					],
				}
			)

	results.sort(key=lambda item: float(item["confidence"]), reverse=True)
	return results


def predict_image(
	model: ort.InferenceSession,
	image: str | Path | np.ndarray,
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.45,
	with_objectness: bool = False,
) -> list[dict[str, object]]:
	"""传入模型句柄和图像，返回检测结果列表。

	参数:
		model: 已加载好的 ONNX Runtime 模型句柄。
		image: 待检测图像，可以是图像路径，也可以是已读取的 `numpy.ndarray`。
		conf_threshold: 置信度阈值，低于该值的候选结果会被过滤。
		iou_threshold: 非极大值抑制的 IoU 阈值，用于去除重复检测框。
		with_objectness: 是否按包含 objectness 的输出格式解析模型结果。

	返回:
		检测结果列表。每个元素为一个字典，包含以下字段：
		- `class_id`: 检测到的类别编号。
		- `confidence`: 检测结果的置信度。
		- `box`: 检测框坐标，格式为 `[x1, y1, x2, y2]`。
	"""
	image_data = _read_image(image)
	input_tensor, scale, pad = _preprocess_image(image_data, _get_input_size(model))

	input_name = model.get_inputs()[0].name
	output_name = model.get_outputs()[0].name
	output = model.run([output_name], {input_name: input_tensor})[0]
	return _parse_detections(
		output=output,
		image_shape=image_data.shape[:2],
		scale=scale,
		pad=pad,
		conf_threshold=conf_threshold,
		iou_threshold=iou_threshold,
		with_objectness=with_objectness,
	)


def parse_args() -> argparse.Namespace:
	"""解析命令行参数。"""
	parser = argparse.ArgumentParser(description="使用 ONNX Runtime DirectML 后端执行单张图像检测")
	parser.add_argument("--onnx", required=True, help="输入的 ONNX 模型路径")
	parser.add_argument("--image", required=True, help="待检测图像路径")
	parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
	parser.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值，默认 0.45")
	parser.add_argument("--with-objectness", action="store_true", help="按带 objectness 的输出格式解析")
	return parser.parse_args()


def main() -> None:
	"""命令行入口。"""
	args = parse_args()
	model = _load_onnx_model(args.onnx)
	results = predict_image(
		model=model,
		image=args.image,
		conf_threshold=args.conf,
		iou_threshold=args.iou,
		with_objectness=args.with_objectness,
	)
	print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
