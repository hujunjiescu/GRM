
# coding: utf-8

# In[ ]:


import os, pdb, cv2, random, datetime, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from loguru import logger
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM

from utils import get_settings, WarmupCosineSchedule

from dataloaders.single_imageset import Image_dataset
import dataloaders.image_transforms as tr
from networks.resnet import resnet50
from networks.inception import inception_v3
from networks.densenet import densenet121
import timm


class GuidedMask(object):
    def __call__(self, batch_input, gray_cams):
        img_w, img_h = batch_input.shape[-2:]
        for i in range(len(batch_input)):
            gray_cam = gray_cams[i]
            threshold = np.percentile(gray_cam, 90)
            cam_binary = gray_cam.copy()
            np.place(cam_binary, cam_binary<=threshold, 0)
            np.place(cam_binary, cam_binary>threshold, 1)
            cam_binary = cam_binary.astype(np.uint8)
            [contours, _] = cv2.findContours(cam_binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                max_x_radius = np.maximum(10, w//2)
                max_y_radius = np.maximum(10, h//2)
                
                x_radius, y_radius = np.random.randint(0, max_x_radius), np.random.randint(0, max_y_radius)
                
                x_center, y_center = np.random.randint(x, x+w), np.random.randint(y, y+h)
                
                x_start = (x_center-x_radius) if (x_center-x_radius) >=0 else 0
                x_end = (x_center+x_radius) if (x_center+x_radius)<img_w else img_w
                
                y_start = (y_center-y_radius) if (y_center-y_radius)>=0 else 0
                y_end = (y_center+y_radius) if (y_center+y_radius)<img_h else img_h
                
                batch_input[i, :, y_start:y_end, x_start:x_end] = 0
                
        return batch_input  
    

def train(model, data_loader, device, criterion, optimizer, epoch, writer, network, grm_obj):
    model.train()
    correct, total = 0, 0
    epoch_loss = []
    time_start = datetime.datetime.now()
    
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, (inputs, targets, img_names) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if config.network == "resnet50":
                cam_obj = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=True)
            elif config.network == "densenet121":
                cam_obj = GradCAM(model=model, target_layers=[model.features.denseblock4.denselayer16], use_cuda=True)
            elif config.network == "inception_v3":
                cam_obj = GradCAM(model=model, target_layers=[model.Mixed_7c.branch_pool], use_cuda=True)
            elif config.network == "vit":
                cam_obj = GradCAM(model=model, target_layers = [model.blocks[-1].norm1], use_cuda=True, reshape_transform=reshape_transform)
            else:
                raise("Unknown network: {}".format(model))

            gray_cams = cam_obj(input_tensor=inputs)
            inputs = grm_obj(inputs, gray_cams)

            cam_obj.__del__()
            
            optimizer.zero_grad()
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            _, predicted = torch.max(outputs.detach(), 1)

            total += targets.size(0)
            correct += torch.sum(predicted.detach() == targets.detach())
            pbar.update(1)
            pbar.set_description("Epoch: %d, Batch %d/%d, Train loss: %.4f, Train acc: %.4f"%(epoch, 
                                                                                               batch_idx+1, len(data_loader), 
                                                                                               np.mean(epoch_loss), float(correct)/total))
    time_end = datetime.datetime.now()
    acc = float(correct) / total
    loss = np.mean(epoch_loss)
    
    writer.add_scalar('train/epoch_acc', acc, epoch)
    writer.add_scalar('train/epoch_loss', loss, epoch)
    time_spend = (time_end-time_start).seconds
    return acc, loss, time_spend

def test(model, data_loader, device, num_classes, criterion, epoch, writer, mode = "Validation"):
    model.eval()
    correct, total = 0, 0
    epoch_loss, epoch_logits, y_logits, outputs_softmax = [], [], [], []
    time_start = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (inputs, targets, img_names) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            epoch_loss.append(loss.item())
            outputs_softmax = outputs_softmax + [softmax[1] for softmax in F.softmax(outputs, dim=1).detach().cpu().numpy()]
            
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += torch.sum(predicted.detach() == targets.detach())
            
            epoch_logits = epoch_logits + [idx for idx in predicted]
            y_logits = y_logits + [idx for idx in targets.detach().cpu().numpy()]
    
    
    time_end = datetime.datetime.now()
    acc = float(correct) / total
    loss = np.mean(epoch_loss)
    
    writer.add_scalar('{}/epoch_acc'.format(mode), acc, epoch)
    writer.add_scalar('{}/epoch_loss'.format(mode), loss, epoch)
    time_spend = (time_end-time_start).seconds
    return acc, loss, time_spend


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def _init__fn(worker_id):
    np.random.seed(config.seed)
    random.seed(config.seed)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--network", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="glaucoma")
    parser.add_argument("--suffix", type=str, default="20220901")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="Adadelta")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    config = parser.parse_args()
    
    log_path, writer, checkpoint_path = get_settings(config.dataset, config.network, config.suffix)
    logger.add(log_path, level="INFO")

    if config.seed is None:
        config.seed = random.randint(1, 10000)

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(str(config.seed))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda:{}".format(str(config.gpu)))

    torch.backends.cudnn.deterministic=True # this item has impact on the deterministic results when the optimizers are adaptive ones
    torch.cuda.set_device(device) # set the default device in order to allocate bernoulli variable on GPU

    if config.network == "vit":
        net = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        net.head = nn.Linear(768, config.num_classes)
        if config.in_channels != 3:
            net.patch_embed.proj = nn.Conv2d(config.in_channels, 768, kernel_size=(16, 16), stride=(16, 16))
    else:
        net = globals()[config.network](in_channels = config.in_channels, num_classes = config.num_classes)

    net.to(device)
    
    grm_obj = GuidedMask()
    
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        use_scheduler = False
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr = 1.0)
        use_scheduler = False
    elif config.optimizer == "AdamW":
        use_scheduler = True
        optimizer = optim.AdamW(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=15, t_total=config.epochs)
    else:
        raise("Unknown optimizer {}".format(config.optimizer))

    train_tr = transforms.Compose([
                tr.FixedResize([224, 224]), # h, w
                tr.RandomRotate(15),
                tr.Normalize(),
                tr.ToTensor()
            ])

    test_tr = transforms.Compose([
                tr.FixedResize([224, 224]),
                tr.Normalize(),
                tr.ToTensor()
            ])
    
#     trainset = xxx # create training dataset here
#     valset = xxx # create validation dataset here
#     testset = xxx # create test dataset here
    
    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, worker_init_fn=_init__fn)
    valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, worker_init_fn=_init__fn)
    testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, worker_init_fn=_init__fn)
    
    logger.info("Number of training images: {}, val images: {}, test images: {}".format(len(trainset), len(valset), len(testset)))
    
    best_val_acc, best_epoch, correspond_test_acc = -1, -1, -1
    for epoch in range(config.epochs):
        train_acc, train_loss, train_time = train(net, trainloader, device, criterion, optimizer, epoch, writer, config.network, grm_obj)
        logger.info("Epoch: {}, Train Time: {}(s), Train Loss: {}, Train Acc: {}".format(epoch, train_time, train_loss, train_acc))

        val_acc, val_loss, val_time = test(net, valloader, device, config.num_classes, criterion, epoch, writer, mode = "Validation")
        if val_acc > best_val_acc:
            logger.info("Epoch: {}, Val acc increased to {}".format(epoch, val_acc))
            best_val_acc = val_acc
            best_epoch = epoch
            test_acc, test_loss, test_time = test(net, testloader, device, config.num_classes, criterion, epoch, writer, mode = "Test")
            correspond_test_acc = test_acc
            logger.info("Epoch: {}, Test acc is {}".format(str(epoch), test_acc))
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_path = os.path.join(checkpoint_path, "epoch{}.pth".format(str(epoch)))
            torch.save(net.state_dict(), save_path)
            logger.info("Model saved in file: {}".format(save_path))
        else:
            logger.info("Epoch: {}, Val acc doesn't increase. Best Val acc is {} in epoch {}. Corresponding Test Acc is {}".format(epoch, best_val_acc, best_epoch, correspond_test_acc))

        if use_scheduler:
            lr_scheduler.step()
            logger.info("Epoch: {}, Learning rate: {}".format(epoch, lr_scheduler.get_last_lr()[0]))