import torchvision
from torch import nn
from torch.nn import Conv2d
from torchsummary import torchsummary
from torchvision.models import resnet101, resnet50, shufflenet_v2_x1_0, resnet18, mobilenet_v2, googlenet, vgg11_bn, \
    densenet121


class MobileNetv2_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(MobileNetv2_input2channel, self).__init__()
        self.mb2 = mobilenet_v2(num_classes=cls)
        conv = self.mb2.features[0][0]
        self.mb2.features[0][0] = Conv2d(2,
                                         conv.out_channels,
                                         kernel_size=conv.kernel_size,
                                         stride=conv.stride,
                                         padding=conv.padding,
                                         bias=conv.bias)

    def forward(self, x):
        return self.mb2(x)


class MobileNetV3_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(MobileNetV3_input2channel, self).__init__()
        self.mb2 = mobilenet_v3_large(num_classes=cls)

        # orin_conv_op = self.mb2.features[0][0]
        # orin_conv_op.in_channels = 2
        conv = self.mb2.features[0][0]
        self.mb2.features[0][0] = Conv2d(2,
                                         conv.out_channels,
                                         kernel_size=conv.kernel_size,
                                         stride=conv.stride,
                                         padding=conv.padding,
                                         bias=conv.bias)

    def forward(self, x):
        return self.mb2(x)


class ShuffleNetV2_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(ShuffleNetV2_input2channel, self).__init__()
        self.mb2 = shufflenet_v2_x1_0(num_classes=cls)
        conv = self.mb2.conv1[0]
        self.mb2.conv1[0] = Conv2d(2,
                                   conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   bias=conv.bias)

    def forward(self, x):
        return self.mb2(x)


class Googlenet_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(Googlenet_input2channel, self).__init__()
        self.mb2 = googlenet(num_classes=cls, aux_logits=False)
        conv = self.mb2.conv1.conv
        self.mb2.conv1.conv = Conv2d(2,
                                     conv.out_channels,
                                     kernel_size=conv.kernel_size,
                                     stride=conv.stride,
                                     padding=conv.padding,
                                     bias=conv.bias)

    def forward(self, x):
        a = self.mb2(x)
        return a

class densenet121_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(densenet121_input2channel, self).__init__()
        self.mb2 = densenet121(num_classes=cls, )
        conv = self.mb2.features[0]
        self.mb2.features[0] = Conv2d(2,
                                     conv.out_channels,
                                     kernel_size=conv.kernel_size,
                                     stride=conv.stride,
                                     padding=conv.padding,
                                     bias=conv.bias)

    def forward(self, x):
        a = self.mb2(x)
        return a


class Vgg11_bn_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(Vgg11_bn_input2channel, self).__init__()
        self.mb2 = vgg11_bn(num_classes=cls)
        conv = self.mb2.features[0]
        self.mb2.features[0] = Conv2d(2,
                                      conv.out_channels,
                                      kernel_size=conv.kernel_size,
                                      stride=conv.stride,
                                      padding=conv.padding,
                                      )

    def forward(self, x):
        return self.mb2(x)


class resnet18_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(resnet18_input2channel, self).__init__()
        self.mb2 = resnet18(num_classes=cls)
        self.mb2.conv1.in_channels = 2
        conv = self.mb2.conv1
        self.mb2.conv1 = Conv2d(2,
                                conv.out_channels,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                bias=conv.bias)

    def forward(self, x):
        return self.mb2(x)


class resnet50_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(resnet50_input2channel, self).__init__()
        self.mb2 = resnet50(num_classes=cls)
        conv = self.mb2.conv1
        self.mb2.conv1 = Conv2d(2,
                                conv.out_channels,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                bias=conv.bias)

    def forward(self, x):
        return self.mb2(x)


class resnet101_input2channel(nn.Module):
    def __init__(self, cls=None):
        super(resnet101_input2channel, self).__init__()
        self.mb2 = resnet101(num_classes=cls)
        conv = self.mb2.conv1
        self.mb2.conv1 = Conv2d(2,
                                conv.out_channels,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                bias=conv.bias)

    def forward(self, x):
        return self.mb2(x)


if __name__ == '__main__':

    ms = [
        MobileNetv2_input2channel(cls=4),
        # # MobileNetV3_input2channel(cls=4),
        ShuffleNetV2_input2channel(cls=4),
        resnet18_input2channel(cls=4),
        # # resnet50_input2channel(cls=4),
        # # resnet101_input2channel(cls=4),
        densenet121_input2channel(cls=4),
        Googlenet_input2channel(cls=4),
        # Vgg11_bn_input2channel(cls=4)
    ]

    for m in ms:
        print("model name=========%s" % m.__class__.__name__)
        torchsummary.summary(m,
                             input_size=(2, 128, 128),
                             batch_size=1, device='cpu')
