from torchvision import models
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
# 网络整体架构
# print(fcn)
# 整体架构分3层
encode = fcn.backbone # resnet101
decode = fcn.classifier
aux_classfier = fcn.aux_classifier
# print("{}++++++++++++++++++{}++++++++++++++++{}".format(encode, decode, aux_classfier))
# 取单独某一层的结构
# 如(backbone): IntermediateLayerGetter的第一个conv1
conv1_encode = list(fcn.backbone.children())[0]
print(conv1_encode)
# 如(classifier): 的第一个conv1
conv1_decode = list(fcn.classifier.children())[0]
print(conv1_decode)

