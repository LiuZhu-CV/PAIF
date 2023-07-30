
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x

class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseBlock, self).__init__()
        # # out_channels_def = 16
        # out_channels_def = in_channels // 2
        # # out_channels_def = out_channels
        self.conv1 = ConvLeakyRelu2d(in_channels, in_channels, kernel_size=kernel_size)
        self.conv2 = ConvLeakyRelu2d(2*in_channels, in_channels, kernel_size=kernel_size)
        self.conv_down = nn.Conv2d(3*in_channels, out_channels, 1)
        # self.sobel = Sobelxy(in_channels)
        # self.conv_up = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_cat1 = torch.cat([x, x_1], dim=1)
        x_2 = self.conv2(x_cat1)
        x_cat2 = torch.cat([x_cat1, x_2], dim=1)
        x_down = self.conv_down(x_cat2)
        # x_grad = self.conv_up(self.sobel(x))
        # out = F.leaky_relu(x_down + x_grad, negative_slope=0.1)
        out = F.leaky_relu(x_down, negative_slope=0.1)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.leaky_relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class ConvLayerLast(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayerLast, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = F.tanh(out)/2+0.5
        return out

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

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""



class f_ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(f_ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout2d(p=0.2)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.batch_norm(out)
        out = F.relu(out, inplace=True)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.conv_pre = nn.Sequential(*[f_ConvLayer(dim, dim, 3, 1),
                                       f_ConvLayer(dim, dim, 3, 1)])
        self.ffn = nn.Sequential(*[f_ConvLayer(dim, dim, 3, 1),
                                       f_ConvLayer(dim, dim, 3, 1)])
        norm_layer=nn.LayerNorm
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq1 = nn.Linear(dim, dim, bias=qkv_bias)
        # self.wq2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk1 = nn.Linear(dim, dim, bias=qkv_bias)
        # self.wk2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv1 = nn.Linear(dim, dim, bias=qkv_bias)
        # self.wv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.end_proj1 = nn.Linear(dim, dim)
        # self.end_proj2 = nn.Linear(dim, dim)

    def forward(self, x1):
        skip = x1
        x1 = self.conv_pre(x1)
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        B, N, C = x1.shape
        q1 = self.wq1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # q2 = self.wq2(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1 = self.wk1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # k2 = self.wk2(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        v1 = self.wv1(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # v2 = self.wv2(x1).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        ctx1 = (q1.transpose(-2, -1) @ k1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        # ctx2 = (q2.transpose(-2, -1) @ k1) * self.scale
        # ctx2 = ctx2.softmax(dim=-2)

        x1 = (v1 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        # x2 = (v1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x1 = self.end_proj1(x1)
        # x2 = self.end_proj2(x2)
        # return x1, x2
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x1 = self.ffn(x1)
        # print(x1.shape)
        # exit()
        return skip+skip*x1

class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, num_heads, index):
        super(FusionBlock_res, self).__init__()


        # # self.axial_attn = AxialBlock(channels, channels//2, h, w)
        self.attn1 = SelfAttention(channels, num_heads)
        self.attn2 = SelfAttention(channels, num_heads)


    def forward(self, x_ir, x_vi):
        return (self.attn1(x_ir) + self.attn2(x_vi))/2

class BFFR(nn.Module):
    def __init__(self, device=None):
        super(BFFR, self).__init__()
        self.device = device
        # self.deepsupervision = False
        block = DenseBlock
        block_light = ConvLayer
        output_filter = 16
        kernel_size = 3
        stride = 1
        output_nc=1

        # nb_filter = [64, 112, 160, 208]
        # nb_filter = [32, 64, 96, 128]
        nb_filter = [16, 32, 64, 96]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv1_vi = ConvLayer(1, output_filter, 1, stride)
        self.DB1_vi = block(output_filter, nb_filter[0], kernel_size)
        self.DB2_vi = block(nb_filter[0], nb_filter[1], kernel_size)
        self.DB3_vi = block(nb_filter[1], nb_filter[2], kernel_size)
        self.DB4_vi = block(nb_filter[2], nb_filter[3], kernel_size)

        self.conv1_ir = ConvLayer(1, output_filter, 1, stride)
        self.DB1_ir = block(output_filter, nb_filter[0], kernel_size)
        self.DB2_ir = block(nb_filter[0], nb_filter[1], kernel_size)
        self.DB3_ir = block(nb_filter[1], nb_filter[2], kernel_size)
        self.DB4_ir = block(nb_filter[2], nb_filter[3], kernel_size)

        # img_w = [320, 160, 80, 40]
        # img_h = [240, 120, 60, 30]
        #img_size = [84,42,21,10]
        num_heads = [4, 8, 8, 16]
        self.fusion_block1 = FusionBlock_res(nb_filter[0], num_heads[0], 0)
        self.fusion_block2 = FusionBlock_res(nb_filter[1], num_heads[1], 1)
        self.fusion_block3 = FusionBlock_res(nb_filter[2], num_heads[2], 2)
        self.fusion_block4 = FusionBlock_res(nb_filter[3], num_heads[3], 3)

        # decoder
        self.DB1_1 = block_light(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size)
        self.DB2_1 = block_light(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size)
        self.DB3_1 = block_light(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size)

        # short connection
        self.DB1_2 = block_light(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size)
        self.DB2_2 = block_light(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size)
        self.DB1_3 = block_light(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size)

        self.conv_out = ConvLayerLast(nb_filter[0], output_nc, 1, stride)

    def forward(self, image_vis_y, image_ir):
        # return (image_vis_y+image_ir)/2
        x = self.conv1_vi(image_vis_y)
        x1_0 = self.DB1_vi(x)
        x2_0 = self.DB2_vi(self.pool(x1_0))
        x3_0 = self.DB3_vi(self.pool(x2_0))
        x4_0 = self.DB4_vi(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        en_vi = [x1_0, x2_0, x3_0, x4_0]

        x = self.conv1_ir(image_ir)
        x1_0 = self.DB1_ir(x)
        x2_0 = self.DB2_ir(self.pool(x1_0))
        x3_0 = self.DB3_ir(self.pool(x2_0))
        x4_0 = self.DB4_ir(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        en_ir = [x1_0, x2_0, x3_0, x4_0]
        
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])

        f_en = [f1_0, f2_0, f3_0, f4_0]
        # for i in f_en:
        #     print(i.shape)
        # exit()
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        output = self.conv_out(x1_3)

        return output
    


