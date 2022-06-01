import os
import torch
import argparse
import torch 
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
if __name__ == '__main__':
    net="ofa_mbv3_d234_e346_k357_w1.0"
    ofa_network = ofa_net(net, pretrained=False)
    # ofa_network.sample_active_subnet()
    ofa_network.set_active_subnet(ks=7, e=6, d=4) 
    subnet = ofa_network.get_active_subnet(preserve_weight=True)

    #print(type(ofa_network))
    #model_dict = subnet.state_dict()
    #checkpoint = torch.load(".tmp/eval_subnet/subnet.pth")
    #print(len(checkpoint.keys()))
    #print(len(subnet.state_dict().keys()))
    # pretrained_dict = {k: v for k, v in checkpoint.items() if k in subnet.state_dict()}
    # model_dict.update(pretrained_dict)

    #subnet.load_state_dict(checkpoint)

    run_config = ImagenetRunConfig(test_batch_size=2, n_worker=1)

    """ Test sampled subnet 
    """ 
    run_manager = RunManager('./eval_subnet', subnet, run_config, init=False)
    # assign image size: 128, 132, ..., 224
    run_config.data_provider.assign_active_img_size(224)
    # run_manager.reset_running_statistics(net=subnet)

    # print('Test random subnet:')
    # print(subnet.module_str)

    # loss, (top1, top5) = run_manager.validate(net=subnet)
    # print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))

    img = torch.rand((2,3,224,244)).cuda()
    model = subnet
    model.eval()
    model(img)