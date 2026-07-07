from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
	import yaml
except ImportError:  # pragma: no cover - yaml 由 ultralytics 依赖间接提供
	yaml = None

try:
	from scripts.predict import _load_onnx_model, predict_image
except ModuleNotFoundError:
	from predict import _load_onnx_model, predict_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ObjectAnnotation:
	class_id: int
	box: list[float]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="对评测集执行目标检测评测并保存可视化结果")
	parser.add_argument("--onnx", required=True, help="待评测的 ONNX 模型路径")
	parser.add_argument(
		"--input",
		default="evaluate/input",
		help="评测集目录，默认 evaluate/input",
	)
	parser.add_argument(
		"--output",
		default="evaluate/output",
		help="评测输出目录，默认 evaluate/output",
	)
	parser.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值，默认 0.25")
	parser.add_argument("--iou", type=float, default=0.5, help="TP 判定 IoU 阈值，默认 0.5")
	parser.add_argument(
		"--classes",
		help="可选类别名文件，支持 classes.txt / names.txt / dataset.yaml",
	)
	return parser.parse_args()


def _ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def _discover_images(input_dir: Path) -> list[Path]:
	return sorted(
		path
		for path in input_dir.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
	)


def _find_label_path(image_path: Path, input_dir: Path) -> Path:
	candidates = [image_path.with_suffix(".txt")]

	try:
		relative_path = image_path.relative_to(input_dir)
	except ValueError:
		relative_path = image_path.name

	if isinstance(relative_path, Path):
		if relative_path.parts and relative_path.parts[0] == "images":
			candidates.append(
				input_dir / "labels" / relative_path.relative_to("images")
			)
		candidates.append(input_dir / "labels" / relative_path)
		candidates.append(input_dir / relative_path.name)
	else:
		candidates.append(input_dir / Path(relative_path).name)

	seen: set[Path] = set()
	for candidate in candidates:
		txt_candidate = candidate.with_suffix(".txt")
		if txt_candidate in seen:
			continue
		seen.add(txt_candidate)
		if txt_candidate.exists():
			return txt_candidate
	return image_path.with_suffix(".txt")


def _load_class_names(input_dir: Path, classes_arg: str | None) -> dict[int, str]:
	candidate_paths: list[Path] = []
	if classes_arg:
		candidate_paths.append(Path(classes_arg).expanduser().resolve())
	candidate_paths.extend(
		[
			input_dir / "classes.txt",
			input_dir / "names.txt",
			input_dir / "dataset.yaml",
			input_dir / "data.yaml",
		]
	)

	for candidate in candidate_paths:
		if not candidate.exists():
			continue
		if candidate.suffix.lower() == ".txt":
			names = [
				line.strip()
				for line in candidate.read_text(encoding="utf-8").splitlines()
				if line.strip()
			]
			return {index: name for index, name in enumerate(names)}
		if candidate.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
			data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
			names = data.get("names", {})
			if isinstance(names, list):
				return {index: str(name) for index, name in enumerate(names)}
			if isinstance(names, dict):
				return {int(index): str(name) for index, name in names.items()}

	return {}


def _class_name(class_id: int, class_names: dict[int, str]) -> str:
	return class_names.get(class_id, f"class_{class_id}")


def _parse_label_line(parts: list[str], image_width: int, image_height: int) -> ObjectAnnotation:
	if len(parts) < 5:
		raise ValueError("标注行至少需要 5 个字段")

	class_id = int(float(parts[0]))
	values = [float(value) for value in parts[1:5]]

	if all(0.0 <= value <= 1.0 for value in values):
		center_x, center_y, box_width, box_height = values
		x1 = (center_x - box_width / 2) * image_width
		y1 = (center_y - box_height / 2) * image_height
		x2 = (center_x + box_width / 2) * image_width
		y2 = (center_y + box_height / 2) * image_height
	else:
		x1, y1, x2, y2 = values

	x1 = float(np.clip(x1, 0, image_width))
	y1 = float(np.clip(y1, 0, image_height))
	x2 = float(np.clip(x2, 0, image_width))
	y2 = float(np.clip(y2, 0, image_height))
	return ObjectAnnotation(class_id=class_id, box=[x1, y1, x2, y2])


def _load_ground_truths(label_path: Path, image_shape: tuple[int, int]) -> list[ObjectAnnotation]:
	image_height, image_width = image_shape
	if not label_path.exists():
		return []

	annotations: list[ObjectAnnotation] = []
	for raw_line in label_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line:
			continue
		annotations.append(_parse_label_line(line.split(), image_width, image_height))
	return annotations


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b
	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_width = max(0.0, inter_x2 - inter_x1)
	inter_height = max(0.0, inter_y2 - inter_y1)
	inter_area = inter_width * inter_height

	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area
	if union <= 0:
		return 0.0
	return inter_area / union


def _match_predictions(
	ground_truths: list[ObjectAnnotation],
	predictions: list[dict[str, Any]],
	iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
	gt_by_class: dict[int, list[tuple[int, ObjectAnnotation]]] = defaultdict(list)
	for gt_index, gt in enumerate(ground_truths):
		gt_by_class[gt.class_id].append((gt_index, gt))

	matched_gt_indices: set[int] = set()
	match_results: list[dict[str, Any]] = []
	false_positive_indices: list[int] = []

	for pred_index, prediction in enumerate(predictions):
		class_id = int(prediction["class_id"])
		best_iou = 0.0
		best_gt_index: int | None = None

		for gt_index, gt in gt_by_class.get(class_id, []):
			if gt_index in matched_gt_indices:
				continue
			iou = _compute_iou(prediction["box"], gt.box)
			if iou > best_iou:
				best_iou = iou
				best_gt_index = gt_index

		is_true_positive = best_gt_index is not None and best_iou >= iou_threshold
		if is_true_positive:
			matched_gt_indices.add(best_gt_index)
		else:
			false_positive_indices.append(pred_index)

		match_results.append(
			{
				"prediction_index": pred_index,
				"class_id": class_id,
				"confidence": float(prediction["confidence"]),
				"iou": float(best_iou),
				"matched_gt_index": best_gt_index,
				"is_true_positive": is_true_positive,
			}
		)

	false_negative_indices = [
		index for index in range(len(ground_truths)) if index not in matched_gt_indices
	]
	return match_results, false_positive_indices, false_negative_indices


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
	if recalls.size == 0 or precisions.size == 0:
		return 0.0

	augmented_recalls = np.concatenate(([0.0], recalls, [1.0]))
	augmented_precisions = np.concatenate(([0.0], precisions, [0.0]))

	for index in range(augmented_precisions.size - 1, 0, -1):
		augmented_precisions[index - 1] = max(
			augmented_precisions[index - 1],
			augmented_precisions[index],
		)

	indices = np.where(augmented_recalls[1:] != augmented_recalls[:-1])[0]
	return float(
		np.sum(
			(augmented_recalls[indices + 1] - augmented_recalls[indices])
			* augmented_precisions[indices + 1]
		)
	)


def _draw_box(
	image: np.ndarray,
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


def _save_visualization(
	image: np.ndarray,
	ground_truths: list[ObjectAnnotation],
	predictions: list[dict[str, Any]],
	class_names: dict[int, str],
	output_path: Path,
) -> None:
	canvas = image.copy()
	for gt in ground_truths:
		_draw_box(
			canvas,
			gt.box,
			f"GT {_class_name(gt.class_id, class_names)}",
			(0, 200, 0),
		)
	for prediction in predictions:
		_draw_box(
			canvas,
			list(map(float, prediction["box"])),
			f"Pred {_class_name(int(prediction['class_id']), class_names)} {float(prediction['confidence']):.2f}",
			(0, 0, 255),
		)
	_ensure_dir(output_path.parent)
	cv2.imwrite(str(output_path), canvas)


def _build_output_image_path(root_dir: Path, image_path: Path, input_root: Path) -> Path:
	try:
		relative_path = image_path.relative_to(input_root)
	except ValueError:
		relative_path = Path(image_path.name)
	return root_dir / relative_path


def evaluate_dataset(
	model_path: str | Path,
	input_dir: str | Path = "evaluate/input",
	output_dir: str | Path = "evaluate/output",
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.5,
	classes_path: str | None = None,
) -> dict[str, Any]:
	input_root = Path(input_dir).expanduser().resolve()
	output_root = Path(output_dir).expanduser().resolve()
	visual_dir = _ensure_dir(output_root / "visualizations")
	error_dir = _ensure_dir(output_root / "errors")
	record_dir = _ensure_dir(output_root / "records")

	if not input_root.exists():
		raise FileNotFoundError(f"评测输入目录不存在: {input_root}")

	image_paths = _discover_images(input_root)
	if not image_paths:
		raise FileNotFoundError(f"在目录中没有找到评测图像: {input_root}")

	class_names = _load_class_names(input_root, classes_path)
	model = _load_onnx_model(model_path)

	per_image_records: list[dict[str, Any]] = []
	prediction_records_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
	gt_count_by_class: Counter[int] = Counter()
	fp_count_by_class: Counter[int] = Counter()
	fn_count_by_class: Counter[int] = Counter()

	total_tp = 0
	total_fp = 0
	total_fn = 0

	for image_path in image_paths:
		image = cv2.imread(str(image_path))
		if image is None:
			raise FileNotFoundError(f"无法读取图像文件: {image_path}")

		label_path = _find_label_path(image_path, input_root)
		ground_truths = _load_ground_truths(label_path, image.shape[:2])
		for gt in ground_truths:
			gt_count_by_class[gt.class_id] += 1

		predictions = predict_image(
			model=model,
			image=image,
			conf_threshold=conf_threshold,
		)
		match_results, false_positive_indices, false_negative_indices = _match_predictions(
			ground_truths,
			predictions,
			iou_threshold,
		)

		for match in match_results:
			prediction_records_by_class[match["class_id"]].append(match)
			if match["is_true_positive"]:
				total_tp += 1
			else:
				total_fp += 1
				fp_count_by_class[match["class_id"]] += 1

		for false_negative_index in false_negative_indices:
			fn_class_id = ground_truths[false_negative_index].class_id
			total_fn += 1
			fn_count_by_class[fn_class_id] += 1

		is_error_image = bool(false_positive_indices or false_negative_indices)
		output_image_path = _build_output_image_path(visual_dir, image_path, input_root)
		_save_visualization(
			image=image,
			ground_truths=ground_truths,
			predictions=predictions,
			class_names=class_names,
			output_path=output_image_path,
		)
		if is_error_image:
			_save_visualization(
				image=image,
				ground_truths=ground_truths,
				predictions=predictions,
				class_names=class_names,
				output_path=_build_output_image_path(error_dir, image_path, input_root),
			)

		per_image_records.append(
			{
				"image_path": str(image_path),
				"label_path": str(label_path),
				"visualization_path": str(output_image_path),
				"error_visualization_path": (
					str(_build_output_image_path(error_dir, image_path, input_root))
					if is_error_image
					else None
				),
				"prediction_count": len(predictions),
				"ground_truth_count": len(ground_truths),
				"true_positive_count": sum(
					1 for match in match_results if match["is_true_positive"]
				),
				"false_positive_count": len(false_positive_indices),
				"false_negative_count": len(false_negative_indices),
				"is_error_image": is_error_image,
				"ground_truths": [
					{
						"class_id": gt.class_id,
						"class_name": _class_name(gt.class_id, class_names),
						"box": gt.box,
					}
					for gt in ground_truths
				],
				"predictions": [
					{
						"class_id": int(prediction["class_id"]),
						"class_name": _class_name(int(prediction["class_id"]), class_names),
						"confidence": float(prediction["confidence"]),
						"box": list(map(float, prediction["box"])),
					}
					for prediction in predictions
				],
				"matches": match_results,
			}
		)

	all_class_ids = sorted(
		set(class_names)
		| set(gt_count_by_class)
		| set(prediction_records_by_class)
		| set(fp_count_by_class)
		| set(fn_count_by_class)
	)

	per_class_metrics: list[dict[str, Any]] = []
	ap_values: list[float] = []
	for class_id in all_class_ids:
		records = sorted(
			prediction_records_by_class.get(class_id, []),
			key=lambda item: float(item["confidence"]),
			reverse=True,
		)
		gt_count = gt_count_by_class[class_id]
		if records:
			tp_array = np.array(
				[1.0 if record["is_true_positive"] else 0.0 for record in records],
				dtype=np.float64,
			)
			fp_array = 1.0 - tp_array
			cumulative_tp = np.cumsum(tp_array)
			cumulative_fp = np.cumsum(fp_array)
			recalls = cumulative_tp / max(gt_count, 1)
			precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
			ap = _compute_ap(recalls, precisions) if gt_count > 0 else 0.0
			final_precision = float(precisions[-1])
			final_recall = float(recalls[-1]) if gt_count > 0 else 0.0
		else:
			ap = 0.0
			final_precision = 0.0
			final_recall = 0.0

		if gt_count > 0:
			ap_values.append(ap)

		per_class_metrics.append(
			{
				"class_id": class_id,
				"class_name": _class_name(class_id, class_names),
				"ground_truth_count": gt_count,
				"prediction_count": len(records),
				"false_positive_count": fp_count_by_class[class_id],
				"false_negative_count": fn_count_by_class[class_id],
				"precision": final_precision,
				"recall": final_recall,
				"ap50": ap,
			}
		)

	precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
	recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
	false_detection_rate = total_fp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
	miss_rate = total_fn / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
	map50 = float(np.mean(ap_values)) if ap_values else 0.0

	most_fp_class_id = fp_count_by_class.most_common(1)[0][0] if fp_count_by_class else None
	most_fn_class_id = fn_count_by_class.most_common(1)[0][0] if fn_count_by_class else None

	summary = {
		"model_path": str(Path(model_path).expanduser().resolve()),
		"input_dir": str(input_root),
		"output_dir": str(output_root),
		"image_count": len(image_paths),
		"iou_threshold": iou_threshold,
		"confidence_threshold": conf_threshold,
		"true_positive": total_tp,
		"false_positive": total_fp,
		"false_negative": total_fn,
		"precision": precision,
		"recall": recall,
		"recall_reached_80_percent": recall >= 0.8,
		"false_detection_rate": false_detection_rate,
		"miss_rate": miss_rate,
		"map50": map50,
		"mean_ap_evaluated_classes": len(ap_values),
		"most_false_positive_class": (
			{
				"class_id": most_fp_class_id,
				"class_name": _class_name(most_fp_class_id, class_names),
				"count": fp_count_by_class[most_fp_class_id],
			}
			if most_fp_class_id is not None
			else None
		),
		"most_false_negative_class": (
			{
				"class_id": most_fn_class_id,
				"class_name": _class_name(most_fn_class_id, class_names),
				"count": fn_count_by_class[most_fn_class_id],
			}
			if most_fn_class_id is not None
			else None
		),
		"per_class_metrics": per_class_metrics,
	}

	(record_dir / "summary.json").write_text(
		json.dumps(summary, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	(record_dir / "per_image_results.json").write_text(
		json.dumps(per_image_records, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	return summary


def _print_summary(summary: dict[str, Any]) -> None:
	print(f"模型: {summary['model_path']}")
	print(f"评测集目录: {summary['input_dir']}")
	print(f"输出目录: {summary['output_dir']}")
	print(f"图像数量: {summary['image_count']}")
	print(f"Precision: {summary['precision']:.4f}")
	print(f"Recall: {summary['recall']:.4f}")
	print(f"Recall 是否达到 80%: {'是' if summary['recall_reached_80_percent'] else '否'}")
	print(f"mAP50: {summary['map50']:.4f}")
	print(f"误检率: {summary['false_detection_rate']:.4f}")
	print(f"漏检率: {summary['miss_rate']:.4f}")

	per_class_metrics = summary.get("per_class_metrics", [])
	if per_class_metrics:
		print("各类别 AP50:")
		header = (
			f"{'Class':<20} {'ID':>4} {'AP50':>8} {'Precision':>10} "
			f"{'Recall':>8} {'GT':>6} {'Prediction':>6}"
		)
		print(header)
		print("-" * len(header))
		for metric in per_class_metrics:
			print(
				f"{metric['class_name']:<20.20} "
				f"{metric['class_id']:>4} "
				f"{metric['ap50']:>8.4f} "
				f"{metric['precision']:>10.4f} "
				f"{metric['recall']:>8.4f} "
				f"{metric['ground_truth_count']:>6} "
				f"{metric['prediction_count']:>6}"
			)

	most_fp = summary["most_false_positive_class"]
	if most_fp is not None:
		print(
			f"误检最多类别: {most_fp['class_name']} (id={most_fp['class_id']}, count={most_fp['count']})"
		)
	else:
		print("误检最多类别: 无")

	most_fn = summary["most_false_negative_class"]
	if most_fn is not None:
		print(
			f"漏检最多类别: {most_fn['class_name']} (id={most_fn['class_id']}, count={most_fn['count']})"
		)
	else:
		print("漏检最多类别: 无")


def main() -> None:
	args = _parse_args()
	summary = evaluate_dataset(
		model_path=args.onnx,
		input_dir=args.input,
		output_dir=args.output,
		conf_threshold=args.conf,
		iou_threshold=args.iou,
		classes_path=args.classes,
	)
	_print_summary(summary)


if __name__ == "__main__":
	main()
