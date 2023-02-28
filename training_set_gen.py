import json
import os.path

from PIL import Image
from GlobalConsts import *
from utils import *

'''
手动创建3个文件夹,分别放置相机拍摄的图片,截屏,表情包
如果读取到iphone专用格式,相机专用格式,视频格式,直接跳过该图片.
命名为:camera, screen_shot, emoji,  分类标签为0,1,2
依次遍历四个文件夹中的所有图片,提取文件绝对路径, 标签, 并构造特征向量, 长度7
[height, width, 是否有Image Make信息(0/1), 是否有Image Model信息(0/1), 是否有gps信息(0/1), 文件格式(全局字典累计), 文件大小kb取整]
'''
DATA_SET_ROOT = "E:/val_data"
# DATA_SET_ROOT = "E:/training_data"

if __name__ == '__main__':
    imageList = []
    root = os.path.abspath(DATA_SET_ROOT).replace('\\', '/')
    for fpathe, dirs, fs in os.walk(DATA_SET_ROOT):
        for f in fs:
            filename = os.path.join(fpathe, f).replace('\\', '/')
            print("process {}".format(filename))
            file_subfix = os.path.splitext(filename)[-1].replace('.', '')
            if file_subfix in EXEMPT_SUFFIX:
                print("skip {}".format(filename))
                continue
            label = get_label_from_filename(filename)
            if label == -1:
                raise RuntimeError("read label for {} return -1!".format(filename))

            try:
                image_size = Image.open(filename).size
                file_size = os.path.getsize(filename)
                with open(filename, 'rb') as file:
                    exif_tags = exifread.process_file(file)
                abs_filename = os.path.abspath(filename).replace('\\', '/')
                path_from_root = abs_filename.replace(root, '')
                img_item = {'path': path_from_root,
                            'image_size': image_size,
                            'file_size': file_size,
                            'label': label}
            except:
                print("can not open {}".format(filename))
            imageList.append(img_item)
        # end of for f in fs:
        # end of os.walk(DATA_SET_ROOT)

        dataset = {'root': root,
                   'len': len(imageList),
                   'images': imageList}
    with open(os.path.join(root, "dataset.json"), "w") as dataset_json:
        json.dump(dataset, dataset_json)
        dataset_json.close()
