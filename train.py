import argparse
import os
import torch
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
import torch.optim as optim
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
import cvxpy as cp

def optim(loss, a, c):
   A = loss
   x = cp.Variable(loss.shape[0])
   objective = cp.Maximize(-a*cp.sum_squares(x)+cp.sum(cp.multiply(A,x)))
   constraints = [0 <= x, cp.sum(x) == c]
   prob = cp.Problem(objective, constraints)
   result = prob.solve()
   x.value = x.value
   return x.value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs

device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
device = torch.device(device)
print(f'We are using device name "{device}"')

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2)])

class CelebADataset(Dataset):
  def __init__(self, root_dir, csv_file, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)
    self.label = pd.read_csv(csv_file)

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    img = img.resize((64, 64))
    pos_1 = train_transform(img)
    pos_2 = train_transform(img)
    label = np.array(self.label.iloc[idx][1:])[20]
    sen_attr = np.array(self.label.iloc[idx][1:])[-1]
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)
      pos_1 = self.transform(pos_1)
      pos_2 = self.transform(pos_2)

    return pos_1, pos_2, img, label, sen_attr, idx

transform = transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
celeba = CelebADataset(csv_file='...',
                                    root_dir='...', transform=transform)
train = int(0.8 * len(celeba))
train_size = int(0.64 * len(celeba))
val_size = int(0.16 * len(celeba))
test_size = len(celeba) - train_size
train, test_dataset = torch.utils.data.random_split(celeba, [train_size, test_size])
train_dataset, val_dataset = torch.utils.data.random_split(train, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=False)


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


weight = np.ones(batch_size)
loss_all = np.zeros(len(celeba))
net = Model().to(device)
train_optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)

def train(epoch, device):
    torch.cuda.empty_cache()
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
    for pos_1, pos_2, image, label, sen, idx in train_bar:
        we_loss = 0
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        out_1 = out_1.to(device)
        out_2 = out_2.to(device)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * image.shape[0], device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        simi_matrix = sim_matrix.masked_select(mask).view(2 * image.shape[0], -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_simi = torch.cat([pos_sim, pos_sim], dim=0)
        loss = torch.log(simi_matrix.sum(dim=-1) / pos_simi)
        loss.backward(gradient=torch.ones_like(loss))
        contr_grad = [torch.autograd.grad(loss, net.parameters(), retain_graph=True)]
        torch.save(model.state_dict(), '...')


        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # encoder
                self.f = Model().f
                # classifier
                self.fc = nn.Linear(2048, 10, bias=True)
                self.load_state_dict(torch.load('...', map_location='gpu'), strict=False)

        classifier = Net().to(device)
        optimizer = optim.Adam(classifier.fc.parameters(), lr=1e-3, weight_decay=1e-6)
        for batch_idx, (images, targets) in enumerate(val_loader):

            optimizer.zero_grad()
            for param in classifier.fc.parameters():
                param.requires_grad = False
            images = images.to(device)
            output = classifier(images).to(device)
            targets = targets.to(device)
            loss = F.nll_loss(output, targets)  # Here is a typical loss function (negative log likelihood)
            loss.backward()
            optimizer.step()
        for param in model.features.parameters():
            param.requires_grad = False
        for batch_idx, (images, targets) in enumerate(val_loader):
            loss = F.nll_loss(output, targets)
            loss = loss.sorted
            loss = loss[0: k-1]
            loss.backward()
            val_grad = f.grad[i]

        for i in range(batch_size):
          weight[i] = torch.max(torch,tensor[torch.dot(val_grad[i], contr_grad[i]),0])

        for i in range(batch_size):
          weight[i] = weight[i] / (torch.sum(weight) + 1*(torch.sum(weight) == 0))
        for i in range(batch_size):
            loss[i] = loss * weight[i]
        loss.backward()
        optimizer.step()
        





    # training loop
for epoch in range(1, epochs + 1):
    train(epoch, device=device)
    if epoch % 100 == 0:
        torch.save(model.state_dict(), '...')