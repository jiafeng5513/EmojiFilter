import os
import json
from PIL import Image
import exifread

'''
手动创建4个文件夹,分别放置相机拍摄的图片,截屏,表情包,网络图片
如果读取到iphone专用格式,相机专用格式,视频格式,直接跳过该图片.
命名为:camera, screen_shot, emoji, web, 分类标签为0,1,2,3
依次遍历四个文件夹中的所有图片,提取文件绝对路径, 标签, 并构造特征向量, 长度7
[height, width, 是否有Image Make信息(0/1), 是否有Image Model信息(0/1), 是否有gps信息(0/1), 文件格式(全局字典累计), 文件大小kb取整]
'''
DATA_SET_ROOT = "/train_val_set"
CLASS_0_FOLDER = "camera"
CLASS_1_FOLDER = "screen_shot"
CLASS_2_FOLDER = "emoji"
CLASS_3_FOLDER = "web"
EXEMPT_SUFFIX = ['avi', 'AVI', 'mp4', 'MP4', 'mov', 'MOV', 'raw', 'RAW', 'ARW', 'arw', 'heic', 'json']
SUFFIX_BANK = []



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


def get_label_from_filename(filename):
    label = -1
    if filename.find(CLASS_0_FOLDER) != -1:
        label = 0
    elif filename.find(CLASS_1_FOLDER) != -1:
        label = 1
    elif filename.find(CLASS_2_FOLDER) != -1:
        label = 2
    elif filename.find(CLASS_3_FOLDER) != -1:
        label = 3
    return label


def feature_encode(filename, imagesize, filesize, subfix, exif_tags):
    # filename, imagesize, filesize
    # subfix和exif_tags做onehot编码
    pass

if __name__ == '__main__':
    for fpathe, dirs, fs in os.walk(DATA_SET_ROOT):
        for f in fs:
            filename = os.path.join(fpathe, f)
            print(filename)
            file_subfix = os.path.splitext(filename)[-1].replace('.','')
            if file_subfix in EXEMPT_SUFFIX:
                print("skip {}".format(filename))
                continue
            label = get_label_from_filename(filename)
            if label == -1:
                raise RuntimeError("read label for {} return -1!".format(filename))

            imageSize = Image.open(filename).size
            filesize = os.path.getsize(filename)
            with open(filename, 'rb') as file:
                exif_tags = exifread.process_file(file)


            if filename.endswith('gif'):
                flag = True
            # size < EMOJI_FILE_SIZE_THRESHOLD is emoji
            if os.path.getsize(filename) < EMOJI_FILE_SIZE_THRESHOLD:
                flag = True
            # resolution < EMOJI_RESOLUTION_THRESHOLD, is a emoji

            if imageSize[0] * imageSize[1] < EMOJI_RESOLUTION_THRESHOLD:
                flag = True
            print(Image.open('1.jpg').size)  # 宽高
            # this is an emoji
            if flag:
                mov(filename, DST_DIR)
