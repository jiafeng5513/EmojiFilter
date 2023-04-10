from global_consts import *
import os
import shutil
import exifread


def get_label_from_filename(filename):
    label = -1
    for i in range(CLASSES_COUNT):
        if filename.find(CLASSES_FOLDER[i]) != -1:
            label = i
    return label


def mov(srcfile, dstpath):
    """
    move file srcfile to dstpath,
    :param srcfile: abs filename
    :param dstpath: dst path
    :return: None
    """
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.move(srcfile, os.path.join(dstpath, fname))  # 移动文件


def is_shot_by_camera(filename):
    try:
        f = open(filename, 'rb')

        tags = exifread.process_file(f)
        if tags is None:
            return False

        if 'Image Make' in tags or 'Image Model' in tags:
            return True
    except:
        print('open failed')
        return True
    return False
