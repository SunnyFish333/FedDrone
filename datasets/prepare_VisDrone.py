import argparse
import os
import random
import shutil
import pandas as pd
from tqdm import tqdm
from datasets_utils import create_directories, archive_directories, get_distribution_dataframe, convert_bbox


def split_files(files: list, nclients: int) -> dict:
    """随机分配文件到客户端"""
    random.seed(42)  # 固定随机种子保证可复现
    random.shuffle(files)
    return {f'client{i + 1}': files[i::nclients] for i in range(nclients)}


def process_split(src_dir: str, dst_dir: str, subset: str,
                  nclients: int, yaml_path: str, is_train: bool) -> pd.DataFrame:
    """处理单个数据分割集"""
    print(f'Processing {subset} subset...')

    try:
        # 加载分布统计表
        dist_df = get_distribution_dataframe(yaml_path, nclients)
    except Exception as e:
        raise RuntimeError(f"加载YAML配置失败: {str(e)}") from e

    # 构建完整路径
    img_dir = os.path.join(src_dir, 'images', subset)
    ann_dir = os.path.join(src_dir, 'annotations', subset)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not img_files:
        raise RuntimeError(f"{subset}子集没有找到图像文件")

    # 分配文件到客户端或服务器
    split_map = split_files(img_files, nclients) if is_train else {'server': img_files}

    # 进度条配置
    progress_desc = f"{subset} -> clients" if is_train else f"{subset} -> server"

    with tqdm(total=len(img_files), desc=progress_desc) as pbar:
        for destination, files in split_map.items():
            for img_file in files:
                # 源路径
                img_src = os.path.join(img_dir, img_file)
                ann_src = os.path.join(ann_dir, img_file.replace('.jpg', '.txt'))

                # 目标路径
                img_dst = os.path.join(dst_dir, destination, 'images', img_file)
                label_dst = os.path.join(dst_dir, destination, 'labels',
                                         img_file.replace('.jpg', '.txt'))

                # 确保目标目录存在
                os.makedirs(os.path.dirname(img_dst), exist_ok=True)
                os.makedirs(os.path.dirname(label_dst), exist_ok=True)

                try:
                    # 复制图像文件
                    shutil.copy(img_src, img_dst)

                    # 转换标注文件
                    with open(ann_src, 'r') as src, open(label_dst, 'w') as dst:
                        for line in src:
                            parts = line.strip().split(',')
                            if len(parts) < 8:
                                continue

                            # 解析参数并验证
                            try:
                                x_min = float(parts[0])
                                y_min = float(parts[1])
                                w = float(parts[2])
                                h = float(parts[3])
                                cls_id = int(parts[5])
                                status = int(parts[6])
                            except (ValueError, IndexError) as e:
                                continue

                            # 过滤无效标注
                            if status == 0 or not (1 <= cls_id <= 10):
                                continue

                            # 坐标转换
                            x, y, bw, bh = convert_bbox(
                                bbox_left=x_min,
                                bbox_top=y_min,
                                bbox_right=x_min + w,
                                bbox_bottom=y_min + h,
                                img_width=1920,
                                img_height=1080
                            )

                            # 写入YOLO格式
                            dst.write(f"{cls_id - 1} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

                            # 更新统计信息
                            class_name = dist_df.index[cls_id]  # 根据YAML中的顺序获取类名
                            dist_df.loc[class_name, destination] += 1
                            dist_df.loc['Samples', destination] += 1

                    pbar.update(1)
                except Exception as e:
                    print(f"处理文件 {img_file} 失败: {str(e)}")
                    continue

    return dist_df


def main():
    parser = argparse.ArgumentParser(description='VisDrone联邦学习数据预处理工具')
    parser.add_argument('--src', type=str, default='./VisDrone2019',
                        help='原始数据集路径（必须包含images和annotations子目录）')
    parser.add_argument('--dst', type=str, default='./VisDrone',
                        help='输出目录路径（将自动创建client/server结构）')
    parser.add_argument('--data', type=str, default='../data/VisDrone.yaml',
                        help='YAML配置文件路径（默认：../data/VisDrone.yaml）')
    parser.add_argument('--nclients', type=int, default=10,
                        help='联邦客户端数量（默认：10）')
    parser.add_argument('--tar', action='store_true',
                        help='是否生成tar归档文件（默认：否）')
    args = parser.parse_args()

    # 输入验证
    if not os.path.exists(args.src):
        raise FileNotFoundError(f"原始数据集路径不存在: {args.src}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"YAML配置文件不存在: {args.data}")

    try:
        # 创建目录结构
        create_directories(args.dst, args.nclients)

        # 处理训练集和验证集
        print("=" * 50)
        train_dist = process_split(args.src, args.dst, 'train', args.nclients, args.data, True)
        print("=" * 50)
        val_dist = process_split(args.src, args.dst, 'val', args.nclients, args.data, False)

        # 合并统计结果
        total_dist = train_dist.add(val_dist, fill_value=0)
        stats_path = os.path.join(args.dst, 'objects_distribution.csv')
        total_dist.to_csv(stats_path)
        print(f"\n数据分布统计已保存至: {stats_path}")

        # 归档处理
        if args.tar:
            print("\n开始归档数据集...")
            archive_directories(args.dst, args.nclients)
            print("归档完成！")

        print("\n数据处理全部完成！")

    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
        raise


if __name__ == '__main__':
    main()

