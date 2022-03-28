import math
import os.path
home = os.path.expanduser("~")
ddir = os.path.join(home,"11785Project/data/VOCdevkit/")
VOCroot = ddir
COCOroot = os.path.join(home,"11785Project/data/coco/")

def reglayer_scale(size, num_layer, size_the):
    reg_layer_size = []
    for i in range(num_layer + 1):
        size = math.ceil(size / 2.)
        if i >= 2:
            reg_layer_size += [size]
            if i == num_layer and size_the != 0:
                reg_layer_size += [size - size_the]
    return reg_layer_size

def get_scales(size, size_pattern):
    return [round(x * size, 2) for x in size_pattern]

def aspect_ratio(num):
    return [[2, 3] for _ in range(num)]

def mk_anchors(size, multiscale_size, size_pattern, step_pattern, num_reglayer = 5, param = 2):
    cfg = {
        'feature_maps': reglayer_scale(
            size, num_reglayer, param if size >= multiscale_size else 0
        )
    }

    cfg['min_dim'] = size
    cfg['steps'] = step_pattern
    cfg['min_sizes'] = get_scales(multiscale_size, size_pattern[:-1])
    cfg['max_sizes'] = get_scales(multiscale_size, size_pattern[1:])
    cfg['aspect_ratios'] = aspect_ratio(num_reglayer)
    cfg['variance'] = [0.1, 0.2]
    cfg['clip'] = True
    return cfg

