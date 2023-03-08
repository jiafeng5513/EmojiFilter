import torchvision.transforms as transforms

learning_rate = 5e-4  # init learning rate
max_epoch = 30  # loop training set by max_epoch times
batch_size = 32  # mini batch size
val_each_iter = 100  # val on each val_iter mini batch
resize_w = 512  # resize w result in preprocess
resize_h = 512  # resize h result in preprocess
alpha = 0.2
beta = 0.8

MODEL_NAME = 'emoji_filter_net.pth'  # name of model, for save and load
CLASSES_FOLDER = ["camera", "emoji", "screen_capture", "screen_shot"]
CLASSES_COUNT = len(CLASSES_FOLDER)
EXEMPT_SUFFIX = ['avi', 'AVI', 'mp4', 'MP4', 'mov', 'MOV', 'raw', 'RAW', 'ARW', 'arw', 'heic', 'json', 'csv']
Multimodal_features = ['image_width', 'image_height', 'filesize']
TENSORBOARD_SUMMARY_DIR = './summary/'
image_mean = 0.5
image_std = 0.5


INFERENCE_TRANS = transforms.Compose([
    transforms.Resize([resize_h, resize_w]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[image_mean], std=[image_std])
])

TRAINING_TRANS = transforms.Compose([
    transforms.Resize([resize_h, resize_w]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[image_mean], std=[image_std])
])
