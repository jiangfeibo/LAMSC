import json
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import cv2
from neural_nets import AttentionNet
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

class params():
    checkpoint_path = "checkpoints" # path to save model weights
    device = "cuda"
    dataset = "data/segments" # path to training data
    log_path = "logs" # path to save logs
    epoch = 100 # training epochs
    lr = 1e-4 # learning rate
    batchsize = 128
    segment_num = 5 # segment number of each image

# construction training dataset
class train_datasets(Dataset):
    def __init__(self, dataset_path):
        self.seg_imgs_path = [os.path.join(dataset_path,seg_imgs) for seg_imgs in os.listdir(dataset_path)]
        self.img_transform = self.transform()

    def __len__(self):
        return self.seg_imgs_path.__len__()

    def __getitem__(self, item):
        merge_imgs = []
        imgs_path = []
        for seg_img in os.listdir(self.seg_imgs_path[item]):
            if seg_img.endswith(".jpg") or seg_img.endswith('.png'):
                seg_img_path = os.path.join(self.seg_imgs_path[item],seg_img)
                img = Image.open(seg_img_path).convert('RGB')
                img = self.img_transform(img)
                merge_imgs.append(img)
                imgs_path.append(seg_img_path)
        temp_merge_imgs = merge_imgs[0]
        for img in merge_imgs[1:]:
            temp_merge_imgs = torch.cat((temp_merge_imgs, img), dim=0) # merge all segments into an image
        label = json.load(open(os.path.join(self.seg_imgs_path[item],r"label.txt")))
        label = torch.Tensor(np.array(label))
        return temp_merge_imgs, label, imgs_path

    def transform(self):
        compose = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        return transforms.Compose(compose)

# construction test dataset
class test_datasets(Dataset):
    def __init__(self, dataset_path):
        self.seg_imgs_path = [os.path.join(dataset_path,seg_imgs) for seg_imgs in os.listdir(dataset_path)]
        self.img_transform = self.transform()

    def __len__(self):
        return self.seg_imgs_path.__len__()

    def __getitem__(self, item):
        merge_imgs = []
        imgs_path = []
        for seg_img in os.listdir(self.seg_imgs_path[item]):
            if seg_img.endswith(".jpg") or seg_img.endswith('.png'):
                seg_img_path = os.path.join(self.seg_imgs_path[item],seg_img)
                img = Image.open(seg_img_path).convert('RGB')
                img = self.img_transform(img)
                merge_imgs.append(img)
                imgs_path.append(seg_img_path)
        temp_merge_imgs = merge_imgs[0]
        for img in merge_imgs[1:]:
            temp_merge_imgs = torch.cat((temp_merge_imgs, img), dim=0)  # merge all segments into an image
        return temp_merge_imgs, imgs_path

    def transform(self):
        compose = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        return transforms.Compose(compose)

# Training of the attention network
def Train_AttentionNet(data_path):
    # load dataset
    datasets = train_datasets(data_path)
    train_dataloader = DataLoader(dataset=datasets, batch_size=arg.batchsize, shuffle=True, num_workers=0,
                                  drop_last=False)
    # load model
    model = AttentionNet(in_dims=arg.segment_num*3)# the input dimension is the number of segments*3
    model.to(arg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr,weight_decay=1e-5)
    mse = nn.MSELoss()
    model.train()
    # model training
    for epoch in range(arg.epoch):
        losses = []
        acces = []
        for i, (x, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.to(arg.device)
            y = y.to(arg.device)
            y_ = model(x)
            loss = mse(y_,y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            y_ = torch.where(y_ > 0.7, 1.0, 0.0)
            acc = torch.sum(y_ == y)/y.numel()
            acces.append(acc.item())
        losses = np.mean(losses)
        acces = np.mean(acces)
        print(f"epoch-{epoch} Loss: {losses} | Acc: {acces}")
        torch.save(model.state_dict(), os.path.join(arg.checkpoint_path, "AttentionNet.pth"))

# generate semantic-aware images based on the attention network
@torch.no_grad()
def semantic_aware_images_generation(data_path):
    datasets = test_datasets(data_path)
    dataloader = DataLoader(dataset=datasets, batch_size=1, shuffle=False, num_workers=0,
                                  drop_last=False)
    save_dir = "data/semantic-aware_images"
    os.makedirs(save_dir,exist_ok=True)
    model = AttentionNet(arg.segment_num * 3)
    # load weights
    model.load_state_dict(torch.load(os.path.join(arg.checkpoint_path,"AttentionNet.pth"),map_location="cpu"))
    model.to(arg.device)
    # model test
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        x = x.to(arg.device)
        y_ = model(x)
        y_ = torch.squeeze(y_)
        select_img_indexes = torch.argwhere(y_ > 0.6).squeeze().cpu().tolist()
        if isinstance(select_img_indexes, int): # if the number of selected segments is 1
            select_imgs = y[select_img_indexes][0]
            semantic_aware_image = Image.open(select_imgs).convert('RGB')
            save_path = os.path.join(save_dir,select_imgs.split(os.path.sep)[-2] + "_aware.jpg")
            semantic_aware_image.save(save_path)
        else: # if the number of selected segments > 1
            select_imgs = [img_path[0] for i, img_path in enumerate(y) if i in select_img_indexes]
            semantic_aware_image = Image.open(select_imgs[0]).convert('RGB')
            semantic_aware_image = np.array(semantic_aware_image)
            for i in range(1,len(select_imgs)):
                temp_image = Image.open(select_imgs[i]).convert('RGB')
                semantic_aware_image = np.add(semantic_aware_image, np.array(temp_image))
            semantic_aware_image = Image.fromarray(semantic_aware_image)
            save_path = os.path.join(save_dir, select_imgs[0].split(os.path.sep)[-2] + "_aware.jpg")
            semantic_aware_image.save(save_path)
        print(save_path,"save success!")

arg = params()
if __name__ == '__main__':
    Train_AttentionNet(arg.dataset)
    semantic_aware_images_generation(arg.dataset)
