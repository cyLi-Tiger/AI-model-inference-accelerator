# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2021/08/17 14:31:51
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   read model and resave only state_dict
'''

import torch





checkpoint = torch.load(checkpoint)
model = vgg(dataset=dataset, cfg=checkpoint['cfg']).cuda()
print("=> loading checkpoint '{}'".format(checkpoint))
model.load_state_dict(checkpoint['state_dict'])