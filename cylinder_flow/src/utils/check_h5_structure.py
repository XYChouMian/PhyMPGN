import h5py
import numpy as np
from pathlib import Path


def print_h5_structure(file_path, max_depth=None):
    """
    递归打印HDF5文件的结构

    Args:
        file_path: HDF5文件路径
        max_depth: 最大递归深度，None表示不限制
    """
    def print_attrs(obj, depth, is_last):
        """打印对象的属性"""
        if len(obj.attrs) > 0:
            indent = ""
            for i in range(depth):
                indent += "    " if is_last[i] else "│   "

            for key, value in obj.attrs.items():
                if np.isscalar(value):
                    print(f"{indent}│   @{key} = {value}")
                else:
                    print(
                        f"{indent}│   @{key}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")

    def traverse(obj, depth=0, is_last=None):
        """递归遍历HDF5对象"""
        if is_last is None:
            is_last = []

        if max_depth is not None and depth > max_depth:
            return

        keys = list(obj.keys())

        for idx, key in enumerate(keys):
            item = obj[key]
            is_last_item = (idx == len(keys) - 1)

            # 构建缩进
            indent = ""
            for i in range(depth):
                indent += "    " if is_last[i] else "│   "

            branch = "└─ " if is_last_item else "├─ "

            if isinstance(item, h5py.Group):
                print(f"{indent}{branch}[Group] {key}/")
                print_attrs(item, depth, is_last + [is_last_item])
                traverse(item, depth + 1, is_last + [is_last_item])

            elif isinstance(item, h5py.Dataset):
                shape = item.shape
                dtype = item.dtype

                # 如果数据很小，直接显示值
                if item.size == 1:
                    value = item[()]
                    print(
                        f"{indent}{branch}[Dataset] {key}: shape={shape}, dtype={dtype}, value={value}")
                elif item.size <= 10 and len(shape) == 1:
                    values = item[:]
                    print(
                        f"{indent}{branch}[Dataset] {key}: shape={shape}, dtype={dtype}, values={values}")
                else:
                    print(
                        f"{indent}{branch}[Dataset] {key}: shape={shape}, dtype={dtype}")

                print_attrs(item, depth, is_last + [is_last_item])

    with h5py.File(file_path, 'r') as f:
        print(f"📁 HDF5 File: {file_path.name}")

        # 根节点属性
        if len(f.attrs) > 0:
            for key, value in f.attrs.items():
                if np.isscalar(value):
                    print(f"@{key} = {value}")
                else:
                    print(
                        f"@{key}: shape={np.array(value).shape}, dtype={np.array(value).dtype}")

        traverse(f, 0)


if __name__ == "__main__":
    # 数据集文件夹
    # data_dir = Path("/home/wqx/projects/PhyMPGN/data/wave")
    # data_dir = Path("/home/wqx/projects/PhyMPGN/data/2d_cf")
    # data_dir = Path("/home/wqx/projects/CFD-paradigm/data")
    data_dir = Path("/home/wqx/projects/PhyMPGN/data/2d_bl")
    # data_dir = Path("/mnt/d/study/2023-2024-1/FLUENT/1_Vortex_Streets/TP/2-different Re comparison/Re4000")

    # 查找所有的 HDF5 文件
    h5_files = list(data_dir.glob("*.h5"))

    if not h5_files:
        print(f"❌ No HDF5 files found in {data_dir}")
    else:
        print(f"\n✅ Found {len(h5_files)} HDF5 file(s) in {data_dir}\n")
        for idx, h5_file in enumerate(h5_files):
            print_h5_structure(h5_file)
            print()
