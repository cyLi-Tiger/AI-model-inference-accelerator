# -*- encoding: utf-8 -*-
'''
@File    :   test_model.py
@Time    :   2021/08/16 20:35:58
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   测试各种模型
'''
import time
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet


dataset = 'cifar10'
model_type="vgg19"
checkpoint = 'D:/code/network-slimming/vgg_slim/0.7/pruned.pth.tar'
base_checkpoint = "D:/code/network-slimming/vgg_sparity/model_best.pth.tar"
# base_checkpoint = "D:/code/network-slimming/vgg_baseline/model_best.pth.tar"

# model_type="resnet164"
# checkpoint = 'D:/code/network-slimming/resnet_slim/0.7/pruned.pth.tar'
# # checkpoint = "D:/code/network-slimming/resnet_sparity/model_best.pth.tar"
# base_checkpoint = "D:/code/network-slimming/resnet_baseline/model_best.pth.tar"
# # base_checkpoint = "D:/code/network-slimming/resnet_sparity/model_best.pth.tar"

    
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
    batch_size=256, shuffle=True)

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

if __name__=="__main__":
    checkpoint = torch.load(checkpoint)
    if(model_type=="vgg19"):
        model = vgg(dataset=dataset, cfg=checkpoint['cfg']).cuda()
    elif(model_type=="resnet164"):
        model = resnet(dataset=dataset, cfg=checkpoint['cfg']).cuda()
    else: raise ValueError(f"wrong model type: {model_type}")
            
    print("[Target] loading checkpoint '{}'".format(checkpoint))
    model.load_state_dict(checkpoint['state_dict'])
    
    base_checkpoint = torch.load(base_checkpoint)
    if(model_type=="vgg19"):
        base_model = vgg(dataset=dataset, depth=19).cuda()
    elif(model_type=="resnet164"):
        base_model = resnet(dataset=dataset, depth=164).cuda()
    else: raise ValueError(f"wrong model type: {model_type}")
    
    
    print("[Baseline] loading checkpoint '{}'".format(base_checkpoint))
    base_model.load_state_dict(base_checkpoint['state_dict'])
    
    # torch.save(base_model.state_dict(), "D:/code/network-slimming/test_model/vgg_baseline.pth.tar") 
    # model.save("D:/code/network-slimming/test_model/resnet_pruned.pt") 
     
     
    T1 = time.time()
    prec1 = test(base_model)
    T2 = time.time()
    print('Baseline 程序运行时间:%s毫秒' % ((T2 - T1)*1000))  
    
    T1 = time.time()
    prec2 = test(model)
    T2 = time.time()
    print('Pruned 程序运行时间:%s毫秒' % ((T2 - T1)*1000))  