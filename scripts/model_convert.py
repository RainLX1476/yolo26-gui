from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

__all__ = ["convert_pt_to_onnx"]


def convert_pt_to_onnx(
	pt_path: str | Path,
	onnx_path: str | Path | None = None,
	imgsz: int | tuple[int, int] = 640,
	opset: int = 12,
	dynamic: bool = False,
	simplify: bool = False,
	half: bool = False,
) -> Path:
	"""将 YOLO 的 `.pt` 模型转换为 `.onnx`。

	参数:
		pt_path: 输入的 `.pt` 模型路径。
		onnx_path: 输出的 `.onnx` 文件路径；未传入时使用导出默认路径。
		imgsz: 导出时使用的输入图像尺寸。
		opset: ONNX 的 opset 版本。
		dynamic: 是否启用动态输入尺寸。
		simplify: 是否简化导出的 ONNX 计算图。
		half: 是否导出为 FP16 精度。

	返回:
		生成后的 `.onnx` 文件路径。
	"""
	pt_file = Path(pt_path).expanduser().resolve()
	output_path = Path(onnx_path).expanduser().resolve() if onnx_path else None

	model = YOLO(str(pt_file))
	exported_path = Path(
		model.export(
			format="onnx",
			imgsz=imgsz,
			opset=opset,
			dynamic=dynamic,
			simplify=simplify,
			half=half,
		)
	).resolve()

	if output_path and exported_path != output_path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		exported_path.replace(output_path)
		return output_path

	return exported_path


def _parse_args() -> argparse.Namespace:
	"""解析命令行参数。"""
	parser = argparse.ArgumentParser(description="将 YOLO 的 .pt 模型转换为 .onnx 模型")
	parser.add_argument("--pt", required=True, help="输入的 .pt 模型路径")
	parser.add_argument("--onnx", help="输出的 .onnx 文件路径，默认使用导出结果原始路径")
	parser.add_argument(
		"--imgsz",
		type=int,
		nargs="+",
		default=[640],
		help="导出输入尺寸，传 1 个值表示正方形，传 2 个值表示高和宽",
	)
	parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本，默认 12")
	parser.add_argument("--dynamic", action="store_true", help="启用动态输入尺寸")
	parser.add_argument("--simplify", action="store_true", help="简化导出的 ONNX 计算图")
	parser.add_argument("--half", action="store_true", help="导出为 FP16 精度")
	return parser.parse_args()


def _normalize_imgsz(imgsz: list[int]) -> int | tuple[int, int]:
	"""将命令行传入的尺寸参数整理为导出函数所需格式。"""
	if len(imgsz) == 1:
		return imgsz[0]
	if len(imgsz) == 2:
		return imgsz[0], imgsz[1]
	raise ValueError("--imgsz 只能传入 1 个或 2 个整数")


def main() -> None:
	"""命令行入口。"""
	args = _parse_args()
	output_path = convert_pt_to_onnx(
		pt_path=args.pt,
		onnx_path=args.onnx,
		imgsz=_normalize_imgsz(args.imgsz),
		opset=args.opset,
		dynamic=args.dynamic,
		simplify=args.simplify,
		half=args.half,
	)
	print(f"ONNX 模型已导出到: {output_path}")


if __name__ == "__main__":
	main()
