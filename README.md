相册过滤
============

我有这样一种烦恼，我喜欢拍照，每过一段时间就会把相机和手机里的图片导入到电脑里，然后会用Lightroom管理，每过一段时间会备份一次，每年会用光盘把过去的一年的照片封存起来，但是久而久之就会产生一个问题：相机还好，手机简直就是一个垃圾制造器，手机的相册中混有大量的截图，emoji，还有从社交网络和浏览器或者app中有意无意保存下来的图片，在向电脑中备份的时候会一起混在里面，日复一日的，相册存储库里装满了各种”非拍摄“的图片，而这些非拍摄的图片中，只有很少的部分有保留下来的意义，绝大多数都是没必要保存的。那么有没有一种方法能将这些非拍摄图片从相册中过滤出来呢？

## 大体思路
### 不同来历的图片可能具有一些特征，例如：
1. 大多数emoji，不会具有很大的文件体积和分辨率，但并不绝对。
2. 相机拍摄的照片可能具有特殊的扩展名和有规律的文件名，但有些旧型号相机拍摄的照片可能不满足这种特征。
3. 相机会在文件中记录相机型号，制造商，焦距等信息，但有些照片虽然是相机拍摄的，但是几经周转已经失去了原有的文件特征。
4. 截屏通常是竖画幅的，很多截屏会带有手机的时间，信号，电量等图标。
5. 社交网络保存的图片可能具有水印。
6. 一个gif图片是emoji的可能性很大。
7. iphone的新型号有自己独有的照片格式，妥善的备份方式会保留这种特征。
8. 照相机拍摄的照片可能具有超大的分辨率。
9. 社交网络和app上保存的图片可能会具有一个很长类似乱码的文件名。

* 但是显然这些特征并不十分清晰，有些特征可能是模糊的，无法编写一个仅仅使用这些特征的程序达到足够高的分辨准确度

### 方案设计
1. 分为四类camera, screen_shot, emoji, web。
2. 采用基于规则的过滤和基于神经网络的过滤相结合，规则用于判别一些显然是相机和iphone拍摄的图片，神经网络用来对无法基于规则判断的图片进行基于内容的分析。
3. 特殊格式（RAW, ARW, HEIC, MOV, MP4)直接判定为camera。
4. exif信息带有相机制造商，焦距和摄影诸元，相机型号的，直接判定为camera。
5. 其他的交由神经网络判定。
6. 无论是训练，验证还是测试，都先做规则过滤。

### log
1. 目前三分类，有些图片不容易区分到底是什么，但是大多数截图和大多数emoji是非常明显的。
2. loss有明显的的震荡痕迹，虽然train loss能收敛且val acc能稳步提高，但是val loss明显震荡。
3. 经过数据清洗之后，val acc的抬升趋势更加明显了，5个epoch大概收敛到93%。
4. ![img](./doc/2023-2-14-221927.png)