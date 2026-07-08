from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="检查 YOLO 格式评测集并弹窗展示结果")
	parser.add_argument(
		"--input",
		default="evaluate/input",
		help="数据集目录，默认 evaluate/input",
	)
	return parser.parse_args()


def _ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def _discover_files(input_dir: Path) -> tuple[list[Path], list[Path]]:
	image_paths = sorted(
		path
		for path in input_dir.rglob("*")
		if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
	)
	label_paths = sorted(
		path
		for path in input_dir.rglob("*")
		if path.is_file()
		and (
			(path.suffix.lower() == ".txt" and path.name != "classes.txt")
			or path.suffix.lower() == ".xml"
		)
	)
	return image_paths, label_paths


def _read_class_names(input_dir: Path) -> list[str]:
	classes_path = input_dir / "classes.txt"
	if not classes_path.exists():
		return []
	return [
		line.strip()
		for line in classes_path.read_text(encoding="utf-8").splitlines()
		if line.strip()
	]


def _relative_key(path: Path, root: Path) -> str:
	return str(path.relative_to(root).with_suffix("")).replace("\\", "/")


def _get_png_size(data: bytes) -> tuple[int, int]:
	if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
		raise ValueError("不是有效的 PNG 文件")
	width, height = struct.unpack(">II", data[16:24])
	return width, height


def _get_bmp_size(data: bytes) -> tuple[int, int]:
	if len(data) < 26 or data[:2] != b"BM":
		raise ValueError("不是有效的 BMP 文件")
	width, height = struct.unpack("<ii", data[18:26])
	return abs(width), abs(height)


def _get_jpeg_size(data: bytes) -> tuple[int, int]:
	if len(data) < 4 or data[:2] != b"\xff\xd8":
		raise ValueError("不是有效的 JPEG 文件")

	offset = 2
	while offset < len(data):
		if data[offset] != 0xFF:
			offset += 1
			continue
		while offset < len(data) and data[offset] == 0xFF:
			offset += 1
		if offset >= len(data):
			break

		marker = data[offset]
		offset += 1
		if marker in {0xD8, 0xD9}:
			continue
		if offset + 2 > len(data):
			break

		segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
		if segment_length < 2 or offset + segment_length > len(data):
			break

		if marker in {
			0xC0,
			0xC1,
			0xC2,
			0xC3,
			0xC5,
			0xC6,
			0xC7,
			0xC9,
			0xCA,
			0xCB,
			0xCD,
			0xCE,
			0xCF,
		}:
			if offset + 7 > len(data):
				break
			height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
			return width, height

		offset += segment_length

	raise ValueError("无法解析 JPEG 尺寸")


def _get_webp_size(data: bytes) -> tuple[int, int]:
	if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
		raise ValueError("不是有效的 WEBP 文件")

	chunk_type = data[12:16]
	if chunk_type == b"VP8 ":
		if len(data) < 30:
			raise ValueError("WEBP 文件长度不足")
		width, height = struct.unpack("<HH", data[26:30])
		return width & 0x3FFF, height & 0x3FFF
	if chunk_type == b"VP8L":
		if len(data) < 25:
			raise ValueError("WEBP 文件长度不足")
		b0, b1, b2, b3 = data[21:25]
		width = 1 + (((b1 & 0x3F) << 8) | b0)
		height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
		return width, height
	if chunk_type == b"VP8X":
		if len(data) < 30:
			raise ValueError("WEBP 文件长度不足")
		width = 1 + int.from_bytes(data[24:27], "little")
		height = 1 + int.from_bytes(data[27:30], "little")
		return width, height

	raise ValueError("不支持的 WEBP 子格式")


def _read_image_size_and_validate(image_path: Path) -> tuple[int, int]:
	data = image_path.read_bytes()
	if data.startswith(b"\x89PNG\r\n\x1a\n"):
		return _get_png_size(data)
	if data.startswith(b"\xff\xd8"):
		return _get_jpeg_size(data)
	if data.startswith(b"BM"):
		return _get_bmp_size(data)
	if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
		return _get_webp_size(data)
	raise ValueError(f"不支持或无法识别的图像格式: {image_path.name}")


def _parse_label_file(
	label_path: Path,
	image_width: int,
	image_height: int,
	num_classes: int,
) -> dict[str, Any]:
	records: list[dict[str, Any]] = []
	out_of_range_ids: list[int] = []
	invalid_lines: list[dict[str, Any]] = []

	for line_number, raw_line in enumerate(
		label_path.read_text(encoding="utf-8").splitlines(),
		start=1,
	):
		line = raw_line.strip()
		if not line:
			continue

		parts = line.split()
		if len(parts) < 5:
			invalid_lines.append(
				{
					"line_number": line_number,
					"content": line,
					"reason": "字段数量不足 5",
				}
			)
			continue

		try:
			class_id = int(float(parts[0]))
			box_width = float(parts[3])
			box_height = float(parts[4])
		except ValueError:
			invalid_lines.append(
				{
					"line_number": line_number,
					"content": line,
					"reason": "字段无法解析为数值",
				}
			)
			continue

		if num_classes > 0 and not 0 <= class_id < num_classes:
			out_of_range_ids.append(class_id)

		records.append(
			{
				"class_id": class_id,
				"width_px": abs(box_width * image_width),
				"height_px": abs(box_height * image_height),
			}
		)

	return {
		"records": records,
		"out_of_range_ids": out_of_range_ids,
		"invalid_lines": invalid_lines,
	}


def _parse_xml_label_file(
	label_path: Path,
	image_width: int,
	image_height: int,
	class_names: list[str],
	num_classes: int,
) -> dict[str, Any]:
	records: list[dict[str, Any]] = []
	out_of_range_ids: list[int] = []
	invalid_lines: list[dict[str, Any]] = []
	reverse_class_names = {name: index for index, name in enumerate(class_names)}

	root = ET.fromstring(label_path.read_text(encoding="utf-8"))
	for object_index, obj in enumerate(root.findall("object"), start=1):
		class_name = (obj.findtext("name") or "").strip()
		bndbox = obj.find("bndbox")
		if not class_name or bndbox is None:
			invalid_lines.append(
				{
					"line_number": object_index,
					"content": class_name or "<empty>",
					"reason": "XML object 缺少 name 或 bndbox",
				}
			)
			continue

		try:
			class_id = reverse_class_names[class_name]
		except KeyError:
			try:
				class_id = int(float(class_name))
			except ValueError:
				invalid_lines.append(
					{
						"line_number": object_index,
						"content": class_name,
						"reason": "XML 类别名未在 classes.txt 中定义",
					}
				)
				continue

		if num_classes > 0 and not 0 <= class_id < num_classes:
			out_of_range_ids.append(class_id)

		xmin = float(bndbox.findtext("xmin", "0"))
		ymin = float(bndbox.findtext("ymin", "0"))
		xmax = float(bndbox.findtext("xmax", "0"))
		ymax = float(bndbox.findtext("ymax", "0"))
		box_width = abs(np.clip(xmax, 0, image_width) - np.clip(xmin, 0, image_width))
		box_height = abs(np.clip(ymax, 0, image_height) - np.clip(ymin, 0, image_height))

		records.append(
			{
				"class_id": class_id,
				"width_px": float(box_width),
				"height_px": float(box_height),
			}
		)

	return {
		"records": records,
		"out_of_range_ids": out_of_range_ids,
		"invalid_lines": invalid_lines,
	}


def _build_class_items(counts: dict[int, int], class_names: list[str]) -> list[dict[str, Any]]:
	class_ids = set(counts.keys()) | set(range(len(class_names)))
	return [
		{
			"class_id": class_id,
			"class_name": class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}",
			"count": int(counts.get(class_id, 0)),
		}
		for class_id in sorted(class_ids)
	]


def _build_histogram_config(
	items: list[dict[str, Any]],
	title: str,
	x_label: str,
	y_label: str,
) -> dict[str, Any]:
	return {
		"title": title,
		"x_label": x_label,
		"y_label": y_label,
		"labels": [f"{item['class_name']}({item['class_id']})" for item in items],
		"values": [int(item["count"]) for item in items],
		"items": items,
	}


def _format_report(result: dict[str, Any]) -> str:
	lines = [
		"数据集检查结果",
		f"图片总数量: {result['total_images']}",
		f"检测框总数量: {result['total_boxes']}",
		f"平均每张图片的检测框数量: {result['average_boxes_per_image']:.4f}",
		f"损坏图片数量: {len(result['corrupted_images'])}",
		f"图片缺少标注文件数量: {len(result['images_without_labels'])}",
		f"标注缺少图片文件数量: {len(result['labels_without_images'])}",
		f"类别 ID 越界标注文件数量: {len(result['out_of_range_annotations'])}",
		f"无效标注行文件数量: {len(result['invalid_label_files'])}",
		(
			"小目标数量/比例: "
			f"{result['small_object_stats']['small_boxes']} / "
			f"{result['small_object_stats']['small_box_ratio']:.4f}"
		),
		"",
		"每个类别的图片数量",
	]

	for item in result["per_class_image_counts"]:
		lines.append(f"  [{item['class_id']}] {item['class_name']}: {item['count']}")

	lines.append("")
	lines.append("每个类别的检测框数量")
	for item in result["per_class_box_counts"]:
		lines.append(f"  [{item['class_id']}] {item['class_name']}: {item['count']}")

	if result["images_without_labels"]:
		lines.append("")
		lines.append("没有标注文件的图片")
		lines.extend(f"  {item}" for item in result["images_without_labels"])

	if result["labels_without_images"]:
		lines.append("")
		lines.append("没有图片文件的标注")
		lines.extend(f"  {item}" for item in result["labels_without_images"])

	if result["corrupted_images"]:
		lines.append("")
		lines.append("损坏图片")
		lines.extend(f"  {item}" for item in result["corrupted_images"])

	if result["out_of_range_annotations"]:
		lines.append("")
		lines.append("类别 ID 越界标注")
		for item in result["out_of_range_annotations"]:
			lines.append(
				f"  {item['label_path']}: invalid_class_ids={item['invalid_class_ids']}"
			)

	if result["invalid_label_files"]:
		lines.append("")
		lines.append("包含无效标注行的文件")
		for item in result["invalid_label_files"]:
			lines.append(f"  {item['label_path']}")
			for invalid_line in item["invalid_lines"]:
				lines.append(
					f"    line {invalid_line['line_number']}: {invalid_line['reason']} | {invalid_line['content']}"
				)

	return "\n".join(lines)


def _draw_histogram_image(histogram: dict[str, Any]) -> np.ndarray:
	items = histogram["items"]
	width = max(1200, 180 + len(items) * 70)
	height = 760
	margin_left = 100
	margin_right = 50
	margin_top = 80
	margin_bottom = 280
	plot_width = width - margin_left - margin_right
	plot_height = height - margin_top - margin_bottom
	max_value = max(histogram["values"], default=0)
	max_value = max(max_value, 1)

	canvas = np.full((height, width, 3), 255, dtype=np.uint8)
	axis_color = (90, 90, 90)
	bar_color = (74, 144, 226)
	text_color = (30, 30, 30)

	cv2.putText(
		canvas,
		histogram["title"],
		(40, 42),
		cv2.FONT_HERSHEY_SIMPLEX,
		1.0,
		text_color,
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		canvas,
		histogram["x_label"],
		(width // 2 - 30, height - 25),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		text_color,
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		canvas,
		histogram["y_label"],
		(20, margin_top - 20),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		text_color,
		2,
		cv2.LINE_AA,
	)

	cv2.line(
		canvas,
		(margin_left, margin_top),
		(margin_left, margin_top + plot_height),
		axis_color,
		2,
	)
	cv2.line(
		canvas,
		(margin_left, margin_top + plot_height),
		(margin_left + plot_width, margin_top + plot_height),
		axis_color,
		2,
	)

	for tick in range(6):
		value = int(round(max_value * tick / 5))
		y = margin_top + plot_height - int(plot_height * tick / 5)
		cv2.line(canvas, (margin_left - 8, y), (margin_left, y), axis_color, 2)
		cv2.putText(
			canvas,
			str(value),
			(20, y + 5),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			axis_color,
			1,
			cv2.LINE_AA,
		)

	if items:
		bar_step = plot_width / len(items)
		bar_width = max(16, int(bar_step * 0.58))
		for index, item in enumerate(items):
			bar_height = int((item["count"] / max_value) * max(plot_height - 10, 1))
			x_center = int(margin_left + (index + 0.5) * bar_step)
			x1 = x_center - bar_width // 2
			x2 = x_center + bar_width // 2
			y1 = margin_top + plot_height - bar_height
			y2 = margin_top + plot_height

			cv2.rectangle(canvas, (x1, y1), (x2, y2), bar_color, -1)
			cv2.putText(
				canvas,
				str(item["count"]),
				(max(x1 - 10, 0), max(y1 - 8, 20)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				text_color,
				1,
				cv2.LINE_AA,
			)

			label = f'{item["class_name"]}({item["class_id"]})'
			label_canvas = np.full((220, 60, 3), 255, dtype=np.uint8)
			cv2.putText(
				label_canvas,
				label,
				(2, 42),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(60, 60, 60),
				1,
				cv2.LINE_AA,
			)
			label_canvas = cv2.rotate(label_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
			label_height, label_width = label_canvas.shape[:2]
			x_start = max(0, min(width - label_width, x_center - label_width // 2))
			y_start = margin_top + plot_height + 12
			canvas[y_start : y_start + label_height, x_start : x_start + label_width] = label_canvas

	return canvas


def _show_histogram_windows(image_histogram: np.ndarray, box_histogram: np.ndarray) -> None:
	cv2.namedWindow("Per Class Image Histogram", cv2.WINDOW_NORMAL)
	cv2.namedWindow("Per Class Box Histogram", cv2.WINDOW_NORMAL)
	cv2.imshow("Per Class Image Histogram", image_histogram)
	cv2.imshow("Per Class Box Histogram", box_histogram)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def _save_cli_outputs(result: dict[str, Any]) -> dict[str, str]:
	temp_dir = _ensure_dir(Path("temp").resolve())
	summary_path = temp_dir / "check_dataset_summary.json"
	report_path = temp_dir / "check_dataset_report.txt"
	image_histogram_path = temp_dir / "check_dataset_per_class_images.png"
	box_histogram_path = temp_dir / "check_dataset_per_class_boxes.png"

	image_histogram = _draw_histogram_image(
		result["histogram_data"]["per_class_image_histogram"]
	)
	box_histogram = _draw_histogram_image(
		result["histogram_data"]["per_class_box_histogram"]
	)

	summary_path.write_text(
		json.dumps(result, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	report_path.write_text(_format_report(result), encoding="utf-8")
	cv2.imwrite(str(image_histogram_path), image_histogram)
	cv2.imwrite(str(box_histogram_path), box_histogram)
	return {
		"summary_json_path": str(summary_path),
		"report_text_path": str(report_path),
		"image_histogram_path": str(image_histogram_path),
		"box_histogram_path": str(box_histogram_path),
	}


def check_dataset(input_dir: str | Path = "evaluate/input") -> dict[str, Any]:
	input_path = Path(input_dir).expanduser().resolve()
	if not input_path.exists():
		raise FileNotFoundError(f"数据集目录不存在: {input_path}")

	class_names = _read_class_names(input_path)
	image_paths, label_paths = _discover_files(input_path)
	image_map = {_relative_key(path, input_path): path for path in image_paths}
	label_map = {_relative_key(path, input_path): path for path in label_paths}

	images_without_labels = sorted(
		str(path.relative_to(input_path))
		for key, path in image_map.items()
		if key not in label_map
	)
	labels_without_images = sorted(
		str(path.relative_to(input_path))
		for key, path in label_map.items()
		if key not in image_map
	)

	per_class_image_counts_raw: dict[int, int] = {}
	per_class_box_counts_raw: dict[int, int] = {}
	corrupted_images: list[str] = []
	out_of_range_annotations: list[dict[str, Any]] = []
	invalid_label_files: list[dict[str, Any]] = []
	total_boxes = 0
	small_boxes = 0

	for key, image_path in image_map.items():
		try:
			image_width, image_height = _read_image_size_and_validate(image_path)
		except Exception:
			corrupted_images.append(str(image_path.relative_to(input_path)))
			continue

		label_path = label_map.get(key)
		if label_path is None:
			continue

		if label_path.suffix.lower() == ".xml":
			parsed = _parse_xml_label_file(
				label_path,
				image_width,
				image_height,
				class_names,
				len(class_names),
			)
		else:
			parsed = _parse_label_file(label_path, image_width, image_height, len(class_names))
		records = parsed["records"]
		invalid_ids = parsed["out_of_range_ids"]
		invalid_lines = parsed["invalid_lines"]

		if invalid_ids:
			out_of_range_annotations.append(
				{
					"label_path": str(label_path.relative_to(input_path)),
					"invalid_class_ids": invalid_ids,
				}
			)

		if invalid_lines:
			invalid_label_files.append(
				{
					"label_path": str(label_path.relative_to(input_path)),
					"invalid_lines": invalid_lines,
				}
			)

		class_ids_in_image: set[int] = set()
		for record in records:
			class_id = int(record["class_id"])
			per_class_box_counts_raw[class_id] = per_class_box_counts_raw.get(class_id, 0) + 1
			class_ids_in_image.add(class_id)
			total_boxes += 1
			if record["width_px"] < 32 and record["height_px"] < 32:
				small_boxes += 1

		for class_id in class_ids_in_image:
			per_class_image_counts_raw[class_id] = per_class_image_counts_raw.get(class_id, 0) + 1

	per_class_image_counts = _build_class_items(per_class_image_counts_raw, class_names)
	per_class_box_counts = _build_class_items(per_class_box_counts_raw, class_names)
	total_images = len(image_paths)

	return {
		"input_dir": str(input_path),
		"total_images": total_images,
		"total_boxes": total_boxes,
		"average_boxes_per_image": float(total_boxes / total_images) if total_images else 0.0,
		"per_class_image_counts": per_class_image_counts,
		"per_class_box_counts": per_class_box_counts,
		"images_without_labels": images_without_labels,
		"labels_without_images": labels_without_images,
		"corrupted_images": corrupted_images,
		"out_of_range_annotations": out_of_range_annotations,
		"invalid_label_files": invalid_label_files,
		"small_object_stats": {
			"small_boxes": small_boxes,
			"total_boxes": total_boxes,
			"small_box_ratio": float(small_boxes / total_boxes) if total_boxes else 0.0,
		},
		"class_names": class_names,
		"histogram_data": {
			"per_class_image_histogram": _build_histogram_config(
				per_class_image_counts,
				"每个类别的图片数量",
				"类别",
				"图片数量",
			),
			"per_class_box_histogram": _build_histogram_config(
				per_class_box_counts,
				"每个类别的检测框数量",
				"类别",
				"检测框数量",
			),
		},
	}


def main() -> None:
	args = _parse_args()
	result = check_dataset(args.input)
	output_paths = _save_cli_outputs(result)
	result_with_paths = {**result, **output_paths}
	report_text = _format_report(result_with_paths)
	print(report_text)
	print()
	print(f"结果已保存到: {output_paths['summary_json_path']}")
	print(f"文本报告已保存到: {output_paths['report_text_path']}")
	print(f"图片数量直方图已保存到: {output_paths['image_histogram_path']}")
	print(f"检测框数量直方图已保存到: {output_paths['box_histogram_path']}")
	_show_histogram_windows(
		_draw_histogram_image(result["histogram_data"]["per_class_image_histogram"]),
		_draw_histogram_image(result["histogram_data"]["per_class_box_histogram"]),
	)


if __name__ == "__main__":
	main()
