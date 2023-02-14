learning_rate = 1e-3  # init learning rate
max_epoch = 5  # loop training set by max_epoch times
batch_size = 16  # mini batch size
val_each_iter = 100  # val on each val_iter mini batch
resize_w = 512  # resize w result in preprocess
resize_h = 512  # resize h result in preprocess
model_name = 'emoji_filter_net.pth'  # name of model, for save and load
CLASSES_FOLDER = ["camera", "screen_shot", "emoji"]
CLASSES_COUNT = len(CLASSES_FOLDER)
EXEMPT_SUFFIX = ['avi', 'AVI', 'mp4', 'MP4', 'mov', 'MOV', 'raw', 'RAW', 'ARW', 'arw', 'heic', 'json', 'csv']
