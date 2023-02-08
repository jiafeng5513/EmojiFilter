import os
import exifread
import shutil
from glob import glob
from PIL import Image
import exifread
import json
import urllib.request

'''
已知图库里含有若干种型号已知的相机拍摄的图片和视频,手机屏幕截图,电脑截图,编辑过的图片副本,表情包,网络图片,社交平台上保存到手机的图片,
现在需要一个程序对图库进行过滤和分类,筛选并剔除表情包,屏幕截图和编辑过的图片副本,另存到其他目录中.

> 识别图片是否为表情包

1. 文件格式:png,jpeg,jpg,raw,mp4,avi,无法确定是否表情包,gif一般认为是表情包
2. 文件大小
3. 图片分辨率
4. 图片的相机信息 https://blog.csdn.net/foemat/article/details/104524436

> 识别图片是为为屏幕截图/编辑过的图片
1. 手机电量/信号标志
2. 不正常的分辨率
'''
SRC_DIR = "D:\WorkSpace\DOC"
DST_DIR = ""
EMOJI_FILE_SIZE_THRESHOLD = 1024
EMOJI_RESOLUTION_THRESHOLD = 640 * 480


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
        shutil.move(srcfile, dstpath + fname)  # 移动文件


def is_shot_by_camera(filename):
    f = open(filename, 'rb')

    tags = exifread.process_file(f)
    if tags is None:
        return False

    if 'Image Make' in tags or 'Image Model' in tags:
        return True

    return False


if __name__ == '__main__':
    for fpathe, dirs, fs in os.walk(SRC_DIR):
        for f in fs:
            filename = os.path.join(fpathe, f)
            print(filename)
            flag = False
            # mp4,avi,raw file is not an emoji
            if filename.endswith('mp4') or filename.endswith('avi') or filename.endswith('raw'):
                continue
            # if exif message has camera msg, is not an emoji TODO:是否有可能某些图片有相机信息,但是其实是ps之类的软件名呢
            if is_shot_by_camera(filename):
                continue
            # gif file is emoji
            if filename.endswith('gif'):
                flag = True
            # size < EMOJI_FILE_SIZE_THRESHOLD is emoji
            if os.path.getsize(filename) < EMOJI_FILE_SIZE_THRESHOLD:
                flag = True
            # resolution < EMOJI_RESOLUTION_THRESHOLD, is a emoji
            imageSize = Image.open(filename).size
            if imageSize[0] * imageSize[1] < EMOJI_RESOLUTION_THRESHOLD:
                flag = True
            print(Image.open('1.jpg').size)  # 宽高
            # this is an emoji
            if flag:
                mov(filename, DST_DIR)
