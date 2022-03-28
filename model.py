'''
This code is based on pytorch_ssd and M2Det

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
import torch.backends.cudnn as cudnn
import os,sys,time
from layers.nn_utils import *
from torch.nn import init as init
from utils.core import print_info
import pdb

from backbone.resnet import resnet101
from fpn_neck import FPN

class M2Det(nn.Module):
    def __init__(self, phase, size, config = None):
        '''
        M2Det: Multi-level Multi-scale single-shot object Detector
        '''
        super(M2Det,self).__init__()
        self.phase = phase
        self.size = size
        self.init_params(config)
        print_info('===> Constructing M2Det model', ['yellow','bold'])
        self.construct_modules()

    def init_params(self, config=None): # Directly read the config
        assert config is not None, 'Error: no config'
        for key,value in config.items():
            if check_argu(key,value):
                setattr(self, key, value)

    def construct_modules(self,):
        # construct tums - not needed
        
        # construct base features - else if not needed 
        # TODO: replaced with resnet101 from FCOS
        self.base = resnet101(pretrained=True,if_include_top=False)
        self.fpn = FPN(self.planes,use_p5=True)
        # construct others
        if self.phase == 'test':
            self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256)

        # construct localization and classification features
        loc_ = list()
        conf_ = list()
        self.num_scales =5 num_scales = 5 (P3, P4, P5, P6, P7)
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(self.planes,
                                       4 * 6, # 4 is coordinates, 6 is anchors for each pixels,
                                       3, 1, 1))
            conf_.append(nn.Conv2d(self.planes,
                                       self.num_classes * 6, #6 is anchors for each pixels,
                                       3, 1, 1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)        
    
    def forward(self,x):
        loc,conf = list(),list()
        C3,C4,C5= self.base(x)

        sources = self.fpn([C3,C4,C5])
        sources[0] = self.Norm(sources[0])

        for (x,l,c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return (
            (
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)),
            )
            if self.phase == "test"
            else (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        )

    def init_model(self, base_model_path):
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        
        print_info('Initializing weights for [tums, reduce, up_reduce, leach, loc, conf]...')
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        print_info('Loading weights into state dict...')
        self.load_state_dict(torch.load(base_file))
        print_info('Finished!')

def build_net(phase='train', size=320, config = None):
    return M2Det(phase, size, config)
