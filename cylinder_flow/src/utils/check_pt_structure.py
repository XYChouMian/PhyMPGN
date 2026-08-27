import torch
from pathlib import Path
from typing import Any
import sys
import os

# 添加项目根目录到 sys.path，以便正确加载 pickle 数据
# 假设脚本位于 cylinder_flow/src/utils/ 下
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def format_shape(shape: Any) -> str:
    """格式化 shape"""
    if shape is None:
        return "None"
    if isinstance(shape, (list, tuple)):
        return f"({', '.join(str(s) for s in shape)})"
    return str(shape)


def format_dtype(dtype: Any) -> str:
    """格式化 dtype"""
    if dtype is None:
        return "None"
    return str(dtype)


def get_tensor_info(value: torch.Tensor, max_display: int = 10) -> str:
    """获取张量的信息字符串"""
    shape = format_shape(tuple(value.shape))
    dtype = format_dtype(value.dtype)
    size_mb = value.element_size() * value.nelement() / 1024 / 1024

    info = f"shape={shape}, dtype={dtype}, size={size_mb:.2f} MB"

    # 对于小张量，显示值
    if value.nelement() <= 1:
        if value.dim() == 0:
            info += f", value={value.item()}"
        else:
            info += f", value={value.tolist()}"
    elif value.nelement() <= max_display and value.dim() == 1:
        info += f", values={value.tolist()}"

    return info


def print_pt_structure(file_path: Path, verbose: bool = False):
    """
    递归打印 PT 文件的结构

    Args:
        file_path: PT 文件路径
        verbose: 是否显示详细统计信息（min/max/mean）
    """
    def print_indent(depth: int, is_last: list) -> str:
        """构建缩进"""
        indent = ""
        for i in range(depth):
            indent += "    " if is_last[i] else "│  "
        return indent

    def describe_value(key: str, value: Any, depth: int, is_last_item: bool, is_last: list):
        """描述一个值"""
        indent = print_indent(depth, is_last)
        branch = "└─ " if is_last_item else "├─ "

        # torch.Tensor
        if isinstance(value, torch.Tensor):
            info = get_tensor_info(value)
            extra = ""
            if verbose and value.is_floating_point() and value.nelement() > 0:
                extra = f", min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.float().mean().item():.4f}"
            print(f"{indent}{branch}[Tensor] {key}: {info}{extra}")

        # dict (嵌套字典)
        elif isinstance(value, dict):
            print(f"{indent}{branch}[Dict] {key} ({len(value)} keys)")
            describe_dict(value, depth + 1, is_last + [is_last_item])

        # list / tuple
        elif isinstance(value, (list, tuple)):
            type_name = "List" if isinstance(value, list) else "Tuple"
            print(f"{indent}{branch}[{type_name}] {key}: len={len(value)}")
            # 只显示前几个元素
            display_count = min(len(value), 3)
            for i in range(display_count):
                elem_is_last = (i == display_count - 1) and (display_count >= len(value))
                describe_value(f"[{i}]", value[i], depth + 1, elem_is_last, is_last + [is_last_item])
            if len(value) > display_count:
                indent_inner = print_indent(depth + 1, is_last + [is_last_item])
                print(f"{indent_inner}└─ ... ({len(value) - display_count} more)")

        # Data 对象 (torch_geometric)
        elif hasattr(value, '__dict__') and hasattr(value, 'keys'):
            try:
                keys = list(value.keys())
                print(f"{indent}{branch}[{value.__class__.__name__}] {key} ({len(keys)} attrs)")
                describe_dict(dict(value), depth + 1, is_last + [is_last_item])
            except Exception:
                print(f"{indent}{branch}[{value.__class__.__name__}] {key}: {repr(value)[:100]}")

        # 标量
        elif isinstance(value, (int, float, bool)):
            print(f"{indent}{branch}[{type(value).__name__}] {key}: {value}")

        # 字符串
        elif isinstance(value, str):
            display = value if len(value) <= 50 else value[:47] + "..."
            print(f"{indent}{branch}[str] {key}: \"{display}\"")

        # 其他
        else:
            print(f"{indent}{branch}[{type(value).__name__}] {key}: {repr(value)[:100]}")

    def describe_dict(d: dict, depth: int, is_last: list):
        """描述字典"""
        keys = list(d.keys())
        for idx, key in enumerate(keys):
            is_last_item = (idx == len(keys) - 1)
            describe_value(str(key), d[key], depth, is_last_item, is_last)

    print(f"📁 PT File: {file_path.name}")

    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return

    # 情况1: 元组 (data, slices) - PyG InMemoryDataset 格式
    if isinstance(data, tuple) and len(data) == 2:
        print(f"├─ [Tuple] len=2 (PyG InMemoryDataset format: data, slices)")
        print(f"│  ├─ [0] Data object:")
        if hasattr(data[0], 'keys'):
            describe_dict(dict(data[0]), 2, [False, False])
        else:
            print(f"│  │  └─ Type: {type(data[0]).__name__}")
        slices = data[1]
        if slices is None:
            print(f"│  └─ [1] Slices: None")
        elif isinstance(slices, dict):
            print(f"│  └─ [1] Slices dict ({len(slices)} keys):")
            for k, v in slices.items():
                v_info = get_tensor_info(v) if isinstance(v, torch.Tensor) else f"type={type(v).__name__}"
                print(f"│     ├─ {k}: {v_info}")
        else:
            print(f"│  └─ [1] Type: {type(slices).__name__}")

    # 情况2: 字典
    elif isinstance(data, dict):
        print(f"├─ [Dict] {len(data)} keys")
        describe_dict(data, 1, [True])

    # 情况3: Data 对象
    elif hasattr(data, 'keys') and hasattr(data, '__dict__'):
        print(f"├─ [{data.__class__.__name__}] ({len(list(data.keys()))} attrs)")
        describe_dict(dict(data), 1, [True])

    # 情况4: 单个张量
    elif isinstance(data, torch.Tensor):
        print(f"└─ [Tensor]: {get_tensor_info(data)}")

    # 情况5: 列表
    elif isinstance(data, list):
        print(f"└─ [List]: len={len(data)}")
        display_count = min(len(data), 3)
        for i in range(display_count):
            is_last_item = (i == display_count - 1) and (display_count >= len(data))
            describe_value(f"[{i}]", data[i], 1, is_last_item, [True])
        if len(data) > display_count:
            print(f"    └─ ... ({len(data) - display_count} more)")

    else:
        print(f"└─ [{type(data).__name__}]: {repr(data)[:200]}")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="检查 PT 文件结构")
    parser.add_argument('--dir', type=str, default=None,
                        help='指定文件夹路径（默认使用预设路径）')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示张量的详细统计信息（min/max/mean）')
    args = parser.parse_args()

    # 默认检查的文件夹
    default_dir = "/home/wqx/projects/PhyMPGN/data/2d_bl/processed"
    data_dir = Path(args.dir) if args.dir else Path(default_dir)

    print(f"📂 检查目录: {data_dir}\n")

    # 查找所有的 PT 文件
    pt_files = sorted(list(data_dir.glob("*.pt")))

    if not pt_files:
        print(f"❌ No PT files found in {data_dir}")
        return

    print(f"✅ Found {len(pt_files)} PT file(s) in {data_dir}\n")

    for pt_file in pt_files:
        file_size = pt_file.stat().st_size / 1024 / 1024
        print(f"{'='*60}")
        print(f"📄 {pt_file.name} ({file_size:.2f} MB)")
        print(f"{'='*60}")
        print_pt_structure(pt_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
