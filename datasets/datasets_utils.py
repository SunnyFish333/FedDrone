import os
import pandas as pd
import tarfile
import yaml


def create_directories(target_path: str, nclients: int) -> None:
    #创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    #创建server子目录
    for subdict in ['', '/images', '/labels']:
        if not os.path.exists(f'{target_path}/server{subdict}'):
            os.makedirs(f'{target_path}/server{subdict}')
    #创建client子目录
    for k in range(1, nclients + 1):
        for subdict in ['', '/images', '/labels']:
            if not os.path.exists(f'{target_path}/client{k}{subdict}'):
                os.makedirs(f'{target_path}/client{k}{subdict}')


def archive_directories(target_path: str, nclients: int) -> None:
    #归档server目录
    server_path = os.path.join(target_path, 'server')
    tar_file_name = os.path.join(target_path, 'server.tar')
    with tarfile.open(tar_file_name, 'w') as tar_handle:
        tar_handle.add(server_path, arcname='server')
    #归档client目录
    for k in range(1, nclients + 1):
        client_path = os.path.join(target_path, f'client{k}')
        tar_file_name = os.path.join(target_path, f'client{k}.tar')
        with tarfile.open(tar_file_name, 'w') as tar_handle:
            tar_handle.add(client_path, arcname=f'client{k}')


def get_distribution_dataframe(data: str, nclients: int) -> pd.DataFrame:
    columns = ['server']
    with open(data, 'r') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    for k in range(1, nclients + 1):
        columns.append(f'client{k}')
    objects_distribution = pd.DataFrame(columns=columns, index=['Samples'] + data_dict['names'])
    objects_distribution.fillna(0, inplace=True)
    return objects_distribution


def convert_bbox(bbox_left: float, bbox_top: float, bbox_right: float, bbox_bottom: float, img_width: int,
                 img_height: int) -> tuple[float, float, float, float]:
    x = (bbox_left + bbox_right) / 2 / img_width
    y = (bbox_top + bbox_bottom) / 2 / img_height
    w = (bbox_right - bbox_left) / img_width
    h = (bbox_bottom - bbox_top) / img_height
    return x, y, w, h