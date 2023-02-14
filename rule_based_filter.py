import os
from PIL import Image
from utils import *

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
SRC_DIR = "E:/原始数据"
DST_DIR = "E:/training_data/emoji"
EMOJI_FILE_SIZE_THRESHOLD = 1024 * 50  # 50kB
EMOJI_RESOLUTION_THRESHOLD = 640 * 480

if __name__ == '__main__':

    # print(size)
    for fpathe, dirs, fs in os.walk(SRC_DIR):
        for f in fs:
            filename = os.path.join(fpathe, f)
            print(filename)
            flag = False
            # mp4,avi,raw file is not an emoji
            if filename.endswith('mp4') or filename.endswith('avi') or filename.endswith('raw') or \
                    filename.endswith('MP4') or filename.endswith('AVI') or filename.endswith('hdlr') or \
                    filename.endswith('RAW') or filename.endswith('HEIC') or filename.endswith('MOV'):
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

            # this is an emoji
            if flag:
                mov(filename, DST_DIR)
                print("mov {} to {}".format(filename, DST_DIR))
