import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b0
import fusion_strategy


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
        # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out



class MSCA_autoencoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(MSCA_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1
        pvt = pvt_v2_b0
        backbone = ASPP

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.pt = pvt()

        self.conv1 = ConvLayer(nb_filter[0], nb_filter[2], 1, stride)
        # self.DB1_0 = block(nb_filter[2], nb_filter[0], kernel_size, 1)
        self.DB1_0 = backbone(nb_filter[2], nb_filter[0])

        self.conv2 = ConvLayer(nb_filter[1], nb_filter[2], 1, stride)
        self.DB2_0 = backbone(nb_filter[2], nb_filter[1])

        self.conv3 = ConvLayer(nb_filter[2], nb_filter[2], 1, stride)
        self.DB3_0 = backbone(nb_filter[2], nb_filter[2])

        self.conv4 = ConvLayer(nb_filter[3], nb_filter[2], 1, stride)
        self.DB4_0 = backbone(nb_filter[2], nb_filter[3])

        # self.DB5_0 = block(nb_filter[3], nb_filter[4], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)

        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv11 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv22 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv33 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1 = self.DB1(x)
        xp2, xp3, xp4, _ = self.pt(x1)
        # print(xp1.shape)
        xc4 = self.conv4(xp4)
        x4_0 = self.DB4_0(xc4)

        xc3 = self.conv3(xp3)
        xu3 = xc3 + self.up_eval(xc3, xc4)
        x3_0 = self.DB3_0(xu3)

        xc2 = self.conv2(xp2)
        xu2 = xc2 + self.up_eval(xc2, xc3)
        x2_0 = self.DB2_0(xu2)

        xc1 = self.conv1(x1)
        # print(xc1.shape)
        # print(xc2.shape)
        xu1 = xc1 + self.up_eval(xc1, xc2)
        x1_0 = self.DB1_0(xu1)
        # x1_0 = self.DB1_0(x)
        # x2_0 = self.DB2_0(self.pool(x1_0))
        # x3_0 = self.DB3_0(self.pool(x2_0))
        # x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.addition_fusion

        f1_0 = fusion_function(en1[0], en2[0])
        f2_0 = fusion_function(en1[1], en2[1])
        f3_0 = fusion_function(en1[2], en2[2])
        f4_0 = fusion_function(en1[3], en2[3])
        return [f1_0, f2_0, f3_0, f4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv11(x1_1)
            output2 = self.conv22(x1_2)
            output3 = self.conv33(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):

        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]