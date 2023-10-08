import json
import numpy as np
import os.path
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from global_consts import *
from utils import *
from model_base_on_resnet import *
from model_base_on_efficient_net import *
from dataloader import dataloader

# global def
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_function(logits, target):
    logits_bin = torch.zeros([logits.shape[0], 2]).to(device)
    logits_bin[:, 0] = logits[:, 0]
    logits_bin[:, 1] = torch.sum(logits[:, 1:], dim=1)

    # one_target = torch.ones_like(target)
    # zero_target = torch.zeros_like(target)
    # target_bin = torch.where(target == 0, zero_target, one_target)

    # loss_a = F.cross_entropy(logits_bin, target_bin)  # camera or not
    loss = F.cross_entropy(logits, target)  # all class
    # loss = alpha * loss_a + beta * loss_b
    return loss


def train_and_val(train_set_json_path, val_set_json_path):
    train_dataset = dataloader(json_path=train_set_json_path, train=True)
    val_dataset = dataloader(json_path=val_set_json_path, train=False)

    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    train_mini_batch_number = len(train_loader)
    val_mini_batch_number = len(val_loader)

    model, optimizer, step_schedule = get_model_ResNet()
    model.train()
    model.to(device)
    print(model)

    if os.path.exists(TENSORBOARD_SUMMARY_DIR):
        shutil.rmtree(TENSORBOARD_SUMMARY_DIR)
    os.makedirs(TENSORBOARD_SUMMARY_DIR)

    writer = SummaryWriter(log_dir=TENSORBOARD_SUMMARY_DIR)

    dummy_image = torch.rand(batch_size, 3, resize_h, resize_w).to(device)
    dummy_feature = torch.rand(batch_size, len(Multimodal_features)).to(device)
    writer.add_graph(model=model, input_to_model=[dummy_image, dummy_feature])

    iter_total = 1
    iter_val_total = 0
    times_val = 0
    for epoch in range(max_epoch):
        for iter, (image, label, feature) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            feature = feature.to(device)

            logits = model(image, feature)
            loss = loss_function(logits, label)
            print("epoch {}/{}, iter {}/{}, loss = {}".format(epoch + 1, max_epoch,
                                                              iter + 1, train_mini_batch_number, loss))
            writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=iter_total - 1)
            # writer.add_scalar(tag="train/loss_a", scalar_value=loss_a, global_step=iter_total - 1)
            # writer.add_scalar(tag="train/loss_b", scalar_value=loss_b, global_step=iter_total - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_schedule.step()
            writer.add_scalar(tag="train/lr", scalar_value=step_schedule.get_last_lr()[0], global_step=iter_total - 1)
            if iter_total % val_each_iter == 0:
                print("its time to val")
                correct_count = 0
                camera_correct_count = 0
                with torch.no_grad():
                    model.eval()
                    for val_iter, (val_image, val_label, feature) in enumerate(val_loader):
                        val_image = val_image.to(device)
                        val_label = val_label.to(device)
                        feature = feature.to(device)
                        val_logits = model(val_image, feature)
                        val_loss = loss_function(val_logits, val_label)
                        val_result = torch.argmax(val_logits, dim=1, keepdim=False)
                        correct_count += torch.sum(val_label == val_result)

                        one_target = torch.ones_like(val_label)
                        zero_target = torch.zeros_like(val_label)
                        val_label_bin = torch.where(val_label == 0, zero_target, one_target)

                        one_result = torch.ones_like(val_result)
                        zero_result = torch.zeros_like(val_result)
                        val_result_bin = torch.where(val_result == 0, zero_result, one_result)

                        camera_correct_count += torch.sum(val_result_bin == val_label_bin)
                        print("val, iter {}/{}, loss = {}".format(val_iter + 1, val_mini_batch_number, val_loss))
                        writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=iter_val_total)
                        # writer.add_scalar(tag="val/loss_a", scalar_value=val_loss_a, global_step=iter_val_total)
                        # writer.add_scalar(tag="val/loss_b", scalar_value=val_loss_b, global_step=iter_val_total)
                        iter_val_total += 1
                    acc = correct_count / (val_mini_batch_number * batch_size)
                    camera_acc = camera_correct_count / (val_mini_batch_number * batch_size)
                    print("val acc = {}, camera acc = {}".format(acc, camera_acc))
                    writer.add_scalar(tag="val/acc", scalar_value=acc, global_step=times_val)
                    writer.add_scalar(tag="val/camera acc", scalar_value=camera_acc, global_step=times_val)
                    times_val += 1
                    model.train()

            iter_total += 1
    pass
    writer.close()
    torch.save(model, MODEL_NAME)


def make_dist_dir(dist_path):
    for i in range(CLASSES_COUNT):
        dist_path_i = os.path.join(dist_path, CLASSES_FOLDER[i])
        if not os.path.exists(dist_path_i):
            os.mkdir(dist_path_i)


def inference(src_path, dist_path, is_data_cleaning=False):
    """
    inference & data cleaning
    in data_cleaning mode, if a picture is considered as something(result) different with its label,
        this picture will be moved to dist_path/${result}, this function will read label from file path,
        so the src_path must put all the pictures into folders named as classes name defined in CLASSES_FOLDER.
        files with suffix in EXEMPT_SUFFIX will be skipped and left in place.
    in inference mode, the src_path can be any format like ${src_path}/${time}/filename.jpg or others.
        this function will move picture to dist_path/${result}/${time}/filename.jpg,
        just replace ${src_path} with dist_path/${result},
        files with suffix in EXEMPT_SUFFIX will be considered as CLASSES_FOLDER[0], for now it is "camera".
    :param src_path: input root.
    :param dist_path: path to move and classify
    :param is_data_cleaning: True=data_cleaning mode, False=inference mode
    :return: None
    """
    make_dist_dir(dist_path)

    model = torch.load(MODEL_NAME)
    model.eval()

    inference_transforms = INFERENCE_TRANS

    if is_data_cleaning:
        import csv
        csv_file = open(os.path.join(dist_path, 'data_cleaning.csv'), 'w+', newline='')
        writer = csv.writer(csv_file)

    for fpathe, dirs, fs in os.walk(src_path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            print(filename)
            file_subfix = os.path.splitext(filename)[-1].replace('.', '')
            if file_subfix in EXEMPT_SUFFIX:
                if is_data_cleaning:
                    print("skip {}".format(filename))
                else:
                    dist = os.path.join(dist_path, CLASSES_FOLDER[0], os.path.relpath(filename, dist_path))
                    print("{} mov to {} because EXEMPT_SUFFIX".format(filename, dist))
                    mov(filename, dist)
                continue

            try:
                img = Image.open(filename)
                img_size = img.size
                file_size = os.path.getsize(filename)

                with torch.no_grad():
                    img_tensor = inference_transforms(img.convert("RGB")).unsqueeze(dim=0).to(device)
                    feature_tensor = torch.from_numpy(np.array([img_size[0], img_size[1], file_size],
                                                               dtype=np.float32)).to(device)

                    infer_logits = model(img_tensor, feature_tensor)
                    class_id = torch.argmax(infer_logits, dim=1, keepdim=False).cpu().numpy()[0]

                if class_id not in range(CLASSES_COUNT):
                    raise RuntimeError("class id overflow in file {}!".format(filename))

                if is_data_cleaning:
                    ground_truth = get_label_from_filename(filename)
                    if ground_truth == class_id:
                        print("{} Classification is correct! ".format(filename))
                    else:
                        writer.writerow([filename, ground_truth, class_id])
                        mov_dist = os.path.join(dist_path, CLASSES_FOLDER[class_id])
                        print("{} Classification is wrong! move to {}".format(filename, mov_dist))
                        mov(filename, mov_dist)
                else:
                    dist = os.path.join(dist_path, CLASSES_FOLDER[class_id], os.path.relpath(filename, dist_path))
                    print("{} is {}, move to {}".format(filename, CLASSES_FOLDER[class_id], dist))
                    mov(filename, dist)
            except:
                print("exception on {}".format(filename))

            # end of try
        # end of "for f in fs"
    # end of "os.walk"

    if is_data_cleaning:
        csv_file.close()


def tensor_to_cv_mat(input_tensor):
    if input_tensor.shape[0] != 1:
        raise RuntimeError("only tensor with shape [1, C, H, W] can be trans to cmMat!")
    mat = input_tensor.cpu().squeeze().numpy()
    mat = (mat * 0.5 + 0.5) * 255
    mat = np.uint8(mat)
    mat = mat.transpose(1, 2, 0)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    return mat


def visual_inference(val_set_json_path, max_loss):
    val_dataset = dataloader(json_path=val_set_json_path, train=False)

    val_loader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False
    )

    val_mini_batch_number = len(val_loader)

    model = torch.load(MODEL_NAME)
    model.eval()
    save_count = 0
    for val_iter, (val_image, val_label, feature) in enumerate(val_loader):
        val_image = val_image.to(device)
        val_label = val_label.to(device)
        feature = feature.to(device)
        with torch.no_grad():
            val_logits = model(val_image, feature)
        val_loss_a, val_loss_b, val_loss = loss_function(val_logits, val_label)
        val_result = torch.argmax(val_logits, dim=1, keepdim=False).cpu().numpy()[0]

        gt = val_label.cpu().numpy()[0]
        print("val, iter {}/{}, loss = {}, gt={}, result = {}".format(val_iter + 1, val_mini_batch_number, val_loss, gt,
                                                                      val_result))

        if val_loss >= max_loss or (gt != val_result):
            img_to_show = tensor_to_cv_mat(val_image)
            img_with_text = cv2.putText(img_to_show, "loss = {:.4f}".format(val_loss), (10, 22),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 0, 255), 2, cv2.LINE_AA, False)

            img_with_text = cv2.putText(img_with_text, "gt={}, result = {}".format(CLASSES_FOLDER[gt], CLASSES_FOLDER[val_result]), (10, 45),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 0, 255), 2, cv2.LINE_AA, False)
            cv2.imwrite('E:/corner_case/{}.png'.format(save_count), img_with_text)
            save_count += 1
            print("save")
        # cv2.imshow("img", img_with_text)
        # cv2.waitKey()

    pass


if __name__ == "__main__":
    train_and_val(train_set_json_path='E:/EmojiNet/training_data/dataset.json',
                  val_set_json_path='E:/EmojiNet/val_data/dataset.json')
    #
    # visual_inference(val_set_json_path='E:/val_data/dataset.json', max_loss=1.22)
    # data_cleaning(src_path='E:/training_data', dist_path='E:/clean_training_data')

    # y_np = np.array([0, 1, 2, 0, 1, 2])
    # y_tensor = torch.from_numpy(y_np)
    #
    # zero = torch.zeros_like(y_tensor)
    # one = torch.ones_like(y_tensor)
    # y_tensor_bin = torch.where(y_tensor > 0, one, zero)
    #
    # print("y_tensor = {}".format(y_tensor))
    # print("y_tensor_bin = {}".format(y_tensor_bin))
    #
    # gt_np = np.array([0, 1, 2, 0, 1, 2])
    # gt_tensor = torch.from_numpy(gt_np)
