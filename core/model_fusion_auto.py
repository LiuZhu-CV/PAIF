import torch.nn as nn
from guided_filter_pytorch.guided_filter import GuidedFilter

# from models_UMF.fusion_net import FusionNet
# from models_tardal.generator import Generator
from .segformer_head import SegFormerHead
from . import mix_transformer
import functools
class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = backbone
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

    def initialize(self,):
        state_dict = torch.load('pretrained/' + self.backbone + '.pth')
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        self.encoder.load_state_dict(state_dict, )
    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)

        return self.decoder(_x)
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class DRDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        print(in_ch_,in_ch)
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out
class Fusion_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB1 = DRDB(in_ch=64)
        self.DRDB2 = DRDB(in_ch=64)
        # self.DRDB3 = DRDB(in_ch=64)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        x1 = self.conv1(torch.cat([ir,vis],dim=1))
        x1 = self.relu(x1)
        f1 = self.DRDB1(x1)
        f2 = self.DRDB2(f1)
        # f2 = self.DRDB3(f2)
        f_final = self.relu(self.conv2(f2))
        f_final = self.relu(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final

class SKFF(nn.Module):
  def __init__(self, in_channels, height=3, reduction=8, bias=False):
    super(SKFF, self).__init__()

    self.height = height
    d = max(int(in_channels / reduction), 4)

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

    self.fcs = nn.ModuleList([])
    for i in range(self.height):
      self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
    self.softmax = nn.Softmax(dim=1)

  def forward(self, inp_feats):
    batch_size = inp_feats[0].shape[0]
    n_feats = inp_feats[0].shape[1]

    inp_feats = torch.cat(inp_feats, dim=1)
    inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

    feats_U = torch.sum(inp_feats, dim=1)
    feats_S = self.avg_pool(feats_U)
    feats_Z = self.conv_du(feats_S)

    attention_vectors = [fc(feats_Z) for fc in self.fcs]
    attention_vectors = torch.cat(attention_vectors, dim=1)
    attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
    # stx()
    attention_vectors = self.softmax(attention_vectors)

    feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

    return feats_V


class Fusion_Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB1 = DRDB()
        self.DRDB2 = DRDB()
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.skff = SKFF(64,2)
        self.skff2 = SKFF(64, 2)
        self.conv3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, 1, padding=0)

    def forward(self, ir, vis, out1, out2):
        # print(np.shape(out1),'----------------')
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        x1 = self.conv1(torch.cat([ir,vis],dim=1))
        x1 = self.relu(x1)
        f1 = self.DRDB1(x1)
        f1 = self.skff([f1,self.conv3(out1)])
        f2 = self.DRDB2(f1)
        f2 = self.skff2([f2,self.conv4(out2)])

        f_final = self.relu(self.conv2(f2))
        # ones = torch.ones_like(f_final)
        # zeros = torch.zeros_like(f_final)
        # f_final = torch.where(f_final > ones, ones, f_final)
        # f_final = torch.where(f_final < zeros, zeros, f_final)
        # # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final


### 1215 new architecture remove architecture retrain the network + discriminator
### Gradient_Norm or other multi-task learning schemes
###

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


##### BaselineCode
from operations_m import *

class MixedOp(nn.Module):

  def __init__(self, C, primitive):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    kernel = 3
    dilation = 1
    if primitive.find('attention') != -1:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
    else:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
        dilation = int(primitive.split('_')[2])
    print(name, kernel, dilation)
    self._op = OPS[name](C, kernel, dilation, False)

  def forward(self, x):
    return self._op(x)


class Cell_Chain(nn.Module):

  def __init__(self, C,type,concat):
    super(Cell_Chain, self).__init__()
    op_names, indices = zip(*type)
    concat = concat
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      print(name,index)
      stride = 1
      op = MixedOp(C,name)
      self._ops += [op]
    self._indices = indices

  def forward(self, inp_features):
    offset = 0
    s1 = inp_features
    for i in range(self._steps):
      s1 = self._ops[offset](s1)
      offset += 1
    return inp_features+s1
# class Cell_Decom(nn.Module):
#   def __init__(self,   C, types, concat):
#     super(Cell_Decom, self).__init__()
#     self._C = C
#     # self._steps = steps # inner nodes
#     self.radiux = [4]
#     self.eps_list = [0.001, 0.0001]
#     self._ops_1 = nn.ModuleList()
#     self._ops_2 = nn.ModuleList()
#     self.conv1x1_lf = nn.Conv2d(C*4, C, kernel_size=1, bias=True)
#     self.conv1x1_hf = nn.Conv2d(C*4, C, kernel_size=1, bias=True)
#     self._steps = len(concat)
#     # self.conv1x1_concat = nn.Conv2d(C*2, C, kernel_size=1, bias=True)
#     self.relu = nn.PReLU()
#     self.chain = Cell_Chain(C,types[0],concat)
#     self.chain2 = Cell_Chain(C,types[1],concat)
#
#   def forward(self, inp_ir, inp_vis):
#     lf_ir, hf_ir = self.decomposition(inp_ir,self._C)
#     lf_vis, hf_vis = self.decomposition(inp_vis,self._C)
#
#     lf = self.conv1x1_lf(torch.cat([lf_ir,lf_vis],dim=1))
#     hf = self.conv1x1_hf(torch.cat([hf_ir,hf_vis],dim=1))
#     ### reconstrcution
#     lf_re = self.chain(lf)
#     hf_re = self.chain2(hf)
#     return lf_re, hf_re
#   def get_residue(self, tensor):
#     max_channel = torch.max(tensor, dim=1, keepdim=True)
#     min_channel = torch.min(tensor, dim=1, keepdim=True)
#     res_channel = max_channel[0] - min_channel[0]
#     return res_channel
#   def decomposition(self, x,C):
#     LF_list = []
#     HF_list = []
#     res = self.get_residue(x)
#     # res = res.repeat(1, C, 1, 1)
#     for radius in self.radiux:
#       for eps in self.eps_list:
#         self.gf = GuidedFilter(radius, eps)
#         LF = self.gf(res, x)
#         LF_list.append(LF)
#         HF_list.append(x - LF)
#     LF = torch.cat(LF_list, dim=1)
#     HF = torch.cat(HF_list, dim=1)
#     return LF, HF
class Cell_Decom(nn.Module):
  def __init__(self,   C, types, concat):
    super(Cell_Decom, self).__init__()
    self._C = C
    # self._steps = steps # inner nodes
    self.radiux = [4]
    self.eps_list = [0.001, 0.0001]
    self._ops_1 = nn.ModuleList()
    self._ops_2 = nn.ModuleList()
    self.conv1x1_lf = nn.Conv2d(C*4, C, kernel_size=1, bias=True)
    self.conv1x1_hf = nn.Conv2d(C*4, C, kernel_size=1, bias=True)
    self._steps = len(concat)
    # self.conv1x1_concat = nn.Conv2d(C*2, C, kernel_size=1, bias=True)
    self.relu = nn.PReLU()
    self.chain = Cell_Chain(C,types[0],concat)
    self.chain2 = Cell_Chain(C,types[1],concat)

  def forward(self, inp_ir, inp_vis):
    lf_ir, hf_ir = self.decomposition(inp_ir,self._C)
    lf_vis, hf_vis = self.decomposition(inp_vis,self._C)
    lf = self.conv1x1_lf(torch.cat([lf_ir,hf_ir],dim=1))
    hf = self.conv1x1_hf(torch.cat([lf_vis,hf_vis],dim=1))
    lf_re = self.chain(lf)
    hf_re = self.chain2(hf)
    return lf_re+inp_ir, hf_re+inp_vis
  def get_residue(self, tensor):
    max_channel = torch.max(tensor, dim=1, keepdim=True)
    min_channel = torch.min(tensor, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel
  def decomposition(self, x,C):
    LF_list = []
    HF_list = []
    res = self.get_residue(x)
    # res = res.repeat(1, C, 1, 1)
    for radius in self.radiux:
      for eps in self.eps_list:
        self.gf = GuidedFilter(radius, eps)
        LF = self.gf(res, x)
        LF_list.append(LF)
        HF_list.append(x - LF)
    LF = torch.cat(LF_list, dim=1)
    HF = torch.cat(HF_list, dim=1)
    return LF, HF


class Cell_Decom_decom(nn.Module):
  def __init__(self,   C, types, concat):
    super(Cell_Decom_decom, self).__init__()
    self._C = C
    # self._steps = steps # inner nodes
    self.radiux = [4]
    self.eps_list = [0.001, 0.0001]
    self._ops_1 = nn.ModuleList()
    self._ops_2 = nn.ModuleList()
    self.conv1x1_lf = nn.Conv2d(C*4, C, kernel_size=1, bias=True)
    self.conv1x1_hf = nn.Conv2d(C*4, C, kernel_size=1, bias=True)
    self._steps = len(concat)
    # self.conv1x1_concat = nn.Conv2d(C*2, C, kernel_size=1, bias=True)
    self.relu = nn.PReLU()
    self.chain = Cell_Chain(C,types[0],concat)
    self.chain2 = Cell_Chain(C,types[1],concat)

  def forward(self, inp_ir, inp_vis):
    lf_ir, hf_ir,res_ir = self.decomposition(inp_ir,self._C)
    lf_vis, hf_vis, res_vis = self.decomposition(inp_vis,self._C)
    lf = self.conv1x1_lf(torch.cat([lf_ir,hf_ir],dim=1))
    hf = self.conv1x1_hf(torch.cat([lf_vis,hf_vis],dim=1))
    lf_re = self.chain(lf)
    hf_re = self.chain2(hf)
    return lf_re+inp_ir, hf_re+inp_vis, lf_ir, hf_ir,res_ir,  lf_vis, hf_vis, res_vis
  def get_residue(self, tensor):
    max_channel = torch.max(tensor, dim=1, keepdim=True)
    min_channel = torch.min(tensor, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel
  def decomposition(self, x,C):
    LF_list = []
    HF_list = []
    res = self.get_residue(x)
    # res = res.repeat(1, C, 1, 1)
    for radius in self.radiux:
      for eps in self.eps_list:
        self.gf = GuidedFilter(radius, eps)
        LF = self.gf(res, x)
        LF_list.append(LF)
        HF_list.append(x - LF)
    LF = torch.cat(LF_list, dim=1)
    HF = torch.cat(HF_list, dim=1)
    return LF, HF,res

class spatial_attn_layer_M(nn.Module):
  def __init__(self, kernel_size=5):
    super(spatial_attn_layer_M, self).__init__()
    self.compress = ChannelPool()
    self.spatial = BasicConv(4, 1, kernel_size, relu=False)

  def forward(self, ir, vis):
    x_compress = self.compress(ir, vis)
    x_out = self.spatial(x_compress)
    scale = torch.sigmoid(x_out)  # broadcasting
    return  scale





class Network_Fusion_Searched(nn.Module):
  def __init__(self, C, criterion,genotype_feature,steps=4, multiplier=3):
    super(Network_Fusion_Searched, self).__init__()
    self._C = C
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._genotype = genotype_feature
    self.stem_1 = nn.Sequential(
        nn.Conv2d(1, C, 3, padding=1, bias=False),
        nn.PReLU()
    )
    self.stem_2 = nn.Sequential(
      nn.Conv2d(1, C, 3, padding=1, bias=False),
      nn.PReLU()
    )
    self.stem_out = nn.Sequential(
      nn.Conv2d(C, C//2, 3, padding=1, bias=False),
      nn.Conv2d(C//2, 1, 3, padding=1, bias=False),
      nn.PReLU()
    )
    self.tanh = nn.Tanh()
    self.spa = spatial_attn_layer_M()
    self.decompation = Cell_Decom(C,[self._genotype.normal_1,self._genotype.normal_2],self._genotype.normal_1_concat)
    self.chain = Cell_Chain(C,self._genotype.normal_3,self._genotype.normal_1_concat)

  def forward(self, ir, vis):
    vis = vis[:, 0:1, :, :]
    ir = ir[:, 0:1, :, :]
    fir = self.stem_1(ir)
    fvis = self.stem_2(vis)
    ir_feature, vis_feature =self.decompation(fir,fvis)
    scale = self.spa(ir_feature, vis_feature)
    aggregated_feature = scale * ir_feature + (1 - scale) * vis_feature
    feature2 = self.chain(aggregated_feature)
    output = self.tanh(self.stem_out(feature2))
    return output


  def _loss(self, ir,vis, mask):
    logits = self(ir,vis)
    return self._criterion(ir,vis,logits,mask)


class Network_Fusion_Searched_showfeatures(nn.Module):
  def __init__(self, C, criterion,genotype_feature,steps=4, multiplier=3):
    super(Network_Fusion_Searched_showfeatures, self).__init__()
    self._C = C
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._genotype = genotype_feature
    self.stem_1 = nn.Sequential(
        nn.Conv2d(1, C, 3, padding=1, bias=False),
        nn.PReLU()
    )
    self.stem_2 = nn.Sequential(
      nn.Conv2d(1, C, 3, padding=1, bias=False),
      nn.PReLU()
    )
    self.stem_out = nn.Sequential(
      nn.Conv2d(C, C//2, 3, padding=1, bias=False),
      nn.Conv2d(C//2, 1, 3, padding=1, bias=False),
      nn.PReLU()
    )
    self.tanh = nn.Tanh()
    self.spa = spatial_attn_layer_M()
    self.decompation = Cell_Decom_decom(C,[self._genotype.normal_1,self._genotype.normal_2],self._genotype.normal_1_concat)
    self.chain = Cell_Chain(C,self._genotype.normal_3,self._genotype.normal_1_concat)

  def forward2(self, ir, vis):
    vis = vis[:, 0:1, :, :]
    ir = ir[:, 0:1, :, :]
    fir = self.stem_1(ir)
    fvis = self.stem_2(vis)
    ir_feature, vis_feature, lf_ir, hf_ir,res_ir,  lf_vis, hf_vis, res_vis  =self.decompation(fir,fvis)
    scale = self.spa(ir_feature, vis_feature)
    aggregated_feature = scale * ir_feature + (1 - scale) * vis_feature
    feature2 = self.chain(aggregated_feature)
    output = self.tanh(self.stem_out(feature2))
    return output, ir_feature, vis_feature, lf_ir, hf_ir,res_ir,  lf_vis, hf_vis, res_vis

  def forward(self, ir, vis):
    vis = vis[:, 0:1, :, :]
    ir = ir[:, 0:1, :, :]
    fir = self.stem_1(ir)
    fvis = self.stem_2(vis)
    ir_feature, vis_feature, lf_ir, hf_ir,res_ir,  lf_vis, hf_vis, res_vis  =self.decompation(fir,fvis)
    scale = self.spa(ir_feature, vis_feature)
    aggregated_feature = scale * ir_feature + (1 - scale) * vis_feature
    feature2 = self.chain(aggregated_feature)
    output = self.tanh(self.stem_out(feature2))
    return output

  def _loss(self, ir,vis, mask):
    logits = self(ir,vis)
    return self._criterion(ir,vis,logits,mask)


class Network_MM_CompModel(nn.Module):
    def __init__(self, model, f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_MM_CompModel, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.seg_loss = segloss
        self.enhance_net = model
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir[:,0:1, :, :], vis[:, 0:1, :, :])
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir[:,0:1, :, :], vis[:, 0:1, :, :])
        return fused

    def forward_object(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir[:,0:1, :, :], vis[:, 0:1, :, :])
        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)
        fused = (fused - torch.min(fused)) / (
                torch.max(fused) - torch.min(fused)
        )
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused,seg_map

    def _loss(self, ir, vis,mask,labels):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))

        return enhance_loss*0.1+denoise_loss*4

    def _loss_coupled(self, ir_, vis_, mask, labels):
        fused_img, seg_map = self(ir_[0], vis_[0])
        vis_ = RGB2YCrCb(vis_[1])
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
        enhance_loss = self._criterion(ir_[1], vis_, fused_img, mask)

        return enhance_loss*0.1+denoise_loss*4

    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()

class Network_MM_TARDAL(nn.Module):
    def __init__(self, f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_MM_TARDAL, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.seg_loss = segloss
        self.enhance_net = Generator()
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused

    def forward_object(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)
        fused = (fused - torch.min(fused)) / (
                torch.max(fused) - torch.min(fused)
        )
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused,seg_map

    def _loss(self, ir, vis,mask,labels):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))

        return enhance_loss*0.1+denoise_loss*4

    def _loss_coupled(self, ir_, vis_, mask, labels):
        fused_img, seg_map = self(ir_[0], vis_[0])
        vis_ = RGB2YCrCb(vis_[1])
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
        enhance_loss = self._criterion(ir_[1], vis_, fused_img, mask)

        return enhance_loss*0.1+denoise_loss*4

    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()


class Network_MM_UMF(nn.Module):
    def __init__(self, f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_MM_UMF, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.seg_loss = segloss
        self.enhance_net = FusionNet()
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused

    def forward_object(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)
        fused = (fused - torch.min(fused)) / (
                torch.max(fused) - torch.min(fused)
        )
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused,seg_map

    def _loss(self, ir, vis,mask,labels):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))

        return enhance_loss*0.1+denoise_loss*4

    def _loss_coupled(self, ir_, vis_, mask, labels):
        fused_img, seg_map = self(ir_[0], vis_[0])
        vis_ = RGB2YCrCb(vis_[1])
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
        enhance_loss = self._criterion(ir_[1], vis_, fused_img, mask)

        return enhance_loss*0.1+denoise_loss*4

    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()

class Network_MM_Searched(nn.Module):
    def __init__(self, C,genotype, f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_MM_Searched, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.seg_loss = segloss
        self.enhance_net = Network_Fusion_Searched(C,f_loss,genotype)
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused

    def forward_object(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)
        fused = (fused - torch.min(fused)) / (
                torch.max(fused) - torch.min(fused)
        )
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]
        seg_map = self.denoise_net(torch_norma)
        return fused,seg_map

    def _loss(self, ir, vis,mask,labels):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))

        return enhance_loss*0.1+denoise_loss*4

    def _loss_coupled(self, ir_, vis_, mask, labels):
        fused_img, seg_map = self(ir_[0], vis_[0])
        vis_ = RGB2YCrCb(vis_[1])
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))
        enhance_loss = self._criterion(ir_[1], vis_, fused_img, mask)

        return enhance_loss*0.1+denoise_loss*4

    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()



class Network_MM_SearchedFusion(nn.Module):
    def __init__(self, C,genotype, f_loss):
        super(Network_MM_SearchedFusion, self).__init__()
        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.enhance_net = Network_Fusion_Searched(C,f_loss,genotype)
        self.denoise_net = None

    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused
    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss

    def _loss_coupled(self, ir_, vis_, mask):
      # print(np.shape(ir_[0]),np.shape(vis_[0]),np.shape(ir_[1]),np.shape(vis_[1]))
      fused_img = self(ir_[0], vis_[0])
      enhance_loss = self._criterion(ir_[1], vis_[1], fused_img, mask)

      return enhance_loss

    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss


    def enhance_net_parameters(self):
        return self.enhance_net.parameters()


class Fusion_Network_auto(nn.Module):
    def __init__(self,genotype):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB_ir = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_vis = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_aggregation = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.conv_concat = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        ir_feature = self.conv1(ir)
        vis_feature = self.conv12(vis)

        ir_feature = self.relu(ir_feature)
        vis_feature = self.relu(vis_feature)

        ir_feature = self.DRDB_ir(ir_feature)
        vis_feature = self.DRDB_vis(vis_feature)
        # f2 = self.DRDB3(f2)
        aggregated_feature = self.conv_concat(torch.concat([ir_feature,vis_feature],dim=1))
        aggregated_feature = self.DRDB_aggregation(aggregated_feature)
        f_final = self.relu(self.conv2(aggregated_feature))
        f_final = self.tanh(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final


class Fusion_Network_Add(nn.Module):
    def __init__(self,genotype):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB_ir = Cell_Chain2(64,genotype.normal, genotype.normal_concat)
        self.DRDB_vis = Cell_Chain2(64,genotype.normal, genotype.normal_concat)
        self.DRDB_aggregation = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        # self.conv_concat = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        ir_feature = self.conv1(ir)
        vis_feature = self.conv12(vis)

        ir_feature = self.relu(ir_feature)
        vis_feature = self.relu(vis_feature)

        ir_feature = self.DRDB_ir(ir_feature)
        vis_feature = self.DRDB_vis(vis_feature)
        # f2 = self.DRDB3(f2)
        aggregated_feature = ir_feature + vis_feature
        aggregated_feature = self.DRDB_aggregation(aggregated_feature)
        f_final = self.relu(self.conv2(aggregated_feature))
        f_final = self.tanh(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final

class Fusion_Network_Average(nn.Module):
    def __init__(self,genotype):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB_ir = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_vis = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_aggregation = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        # self.conv_concat = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()

    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        ir_feature = self.conv1(ir)
        vis_feature = self.conv12(vis)

        ir_feature = self.relu(ir_feature)
        vis_feature = self.relu(vis_feature)

        ir_feature = self.DRDB_ir(ir_feature)
        vis_feature = self.DRDB_vis(vis_feature)
        # f2 = self.DRDB3(f2)
        aggregated_feature = (ir_feature + vis_feature)/2
        aggregated_feature = self.DRDB_aggregation(aggregated_feature)
        f_final = self.relu(self.conv2(aggregated_feature))
        f_final = self.relu(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final

class Fusion_Network_Max(nn.Module):
    def __init__(self,genotype):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB_ir = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_vis = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_aggregation = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        # self.conv_concat = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()

    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        ir_feature = self.conv1(ir)
        vis_feature = self.conv12(vis)

        ir_feature = self.relu(ir_feature)
        vis_feature = self.relu(vis_feature)

        ir_feature = self.DRDB_ir(ir_feature)
        vis_feature = self.DRDB_vis(vis_feature)

        aggregated_feature = torch.max(ir_feature,vis_feature)

        aggregated_feature = self.DRDB_aggregation(aggregated_feature)
        f_final = self.relu(self.conv2(aggregated_feature))
        f_final = self.relu(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final


class ChannelPool(nn.Module):
  def forward(self, ir, vis):
    return torch.cat((torch.max(ir, 1)[0].unsqueeze(1), torch.mean(ir, 1).unsqueeze(1),
                      torch.max(vis, 1)[0].unsqueeze(1), torch.mean(vis, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer_M(nn.Module):
  def __init__(self, kernel_size=5):
    super(spatial_attn_layer_M, self).__init__()
    self.compress = ChannelPool()
    self.spatial = BasicConv(4, 1, kernel_size, relu=False)

  def forward(self, ir, vis):
    x_compress = self.compress(ir, vis)
    x_out = self.spatial(x_compress)
    scale = torch.sigmoid(x_out)  # broadcasting
    return  scale

class Fusion_Network_SPA(nn.Module):
    def __init__(self,genotype):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(1, 64, 3, padding=1)
        self.DRDB_ir = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_vis = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_aggregation = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        # self.conv_concat = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.spa = spatial_attn_layer_M()

    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        ir_feature = self.conv1(ir)
        vis_feature = self.conv12(vis)

        ir_feature = self.relu(ir_feature)
        vis_feature = self.relu(vis_feature)

        ir_feature = self.DRDB_ir(ir_feature)
        vis_feature = self.DRDB_vis(vis_feature)

        scale = self.spa(ir_feature, vis_feature)


        aggregated_feature = scale * ir_feature + (1-scale) * vis_feature

        aggregated_feature = self.DRDB_aggregation(aggregated_feature)
        f_final = self.relu(self.conv2(aggregated_feature))
        f_final = self.relu(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final


class Fusion_Network_Direct(nn.Module):
    def __init__(self,genotype):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.DRDB_ir = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.DRDB_aggregation = Cell_Chain(64,genotype.normal, genotype.normal_concat)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv21 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.PReLU()
        self.tanh = nn.Tanh()
        self.spa = spatial_attn_layer_M()

    def forward(self, ir, vis):
        vis = vis[:,0:1,:,:]
        ir = ir[:, 0:1, :, :]
        ir_feature = self.conv1(torch.cat([ir,vis],dim=1))
        ir_feature = self.relu(ir_feature)
        ir_feature = self.DRDB_ir(ir_feature)
        aggregated_feature = self.DRDB_aggregation(ir_feature)
        f_final = self.relu(self.conv2(aggregated_feature))
        f_final = self.tanh(self.conv21(f_final))
        ones = torch.ones_like(f_final)
        zeros = torch.zeros_like(f_final)
        f_final = torch.where(f_final > ones, ones, f_final)
        f_final = torch.where(f_final < zeros, zeros, f_final)
        # new encode
        f_final = (f_final - torch.min(f_final)) / (
                torch.max(f_final) - torch.min(f_final)
        )
        return f_final

class Network_MM_Base(nn.Module):
    def __init__(self, genotype, f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_MM_Base, self).__init__()

        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.seg_loss = segloss
        self.enhance_net = Fusion_Network_auto(genotype=genotype)
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused

    def forward_object(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)
        # new encode
        fused = (fused - torch.min(fused)) / (
                torch.max(fused) - torch.min(fused)
        )
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused,seg_map

    def _loss(self, ir, vis,mask,labels):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))

        return enhance_loss*0.1+denoise_loss*4

    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)


        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()



class Network_MM_Auto(nn.Module):
    def __init__(self, model_fusion , f_loss, segloss,backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super(Network_MM_Auto, self).__init__()
        self.fusion_nums = 2
        self.seg_nums = 2
        self.fusion_channel = 48
        self.seg_channel = 64
        self._criterion = f_loss
        self.seg_loss = segloss
        self.enhance_net = model_fusion
        self.denoise_net = WeTr(backbone,num_classes,embedding_dim,pretrained)
        self.mean =[123.675, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
    def forward(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1*255
        for index in range(3):
            torch_norma[:,index,:,:] = (torch_norma[:,index,:,:] - self.mean[index])/self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused, seg_map

    def forward_fusion(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        return fused

    def forward_object(self, ir, vis):
        vis = RGB2YCrCb(vis)
        fused = self.enhance_net.forward(ir, vis)
        ones = torch.ones_like(fused)
        zeros = torch.zeros_like(fused)
        fused = torch.where(fused > ones, ones, fused)
        fused = torch.where(fused < zeros, zeros, fused)
        # new encode
        fused = (fused - torch.min(fused)) / (
                torch.max(fused) - torch.min(fused)
        )
        fused_seg = torch.cat((fused, vis[:, 1:2, :, :], vis[:, 2:, :, :]), dim=1)
        fused_seg = YCrCb2RGB(fused_seg)
        ones = torch.ones_like(fused_seg)
        zeros = torch.zeros_like(fused_seg)
        fused_seg = torch.where(fused_seg > ones, ones, fused_seg)
        fused_seg = torch.where(fused_seg < zeros, zeros, fused_seg)
        # new encode
        fused_seg = (fused_seg - torch.min(fused_seg)) / (
                torch.max(fused_seg) - torch.min(fused_seg)
        )
        fused_seg1 = fused_seg
        torch_norma = fused_seg1 * 255
        for index in range(3):
            torch_norma[:, index, :, :] = (torch_norma[:, index, :, :] - self.mean[index]) / self.std[index]

        seg_map = self.denoise_net(torch_norma)
        return fused,seg_map

    def _loss(self, ir, vis,mask,labels):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        denoise_loss = self.seg_loss(outputs,labels.type(torch.long))

        return enhance_loss*0.1+denoise_loss*4

    def _fusion_loss_lower(self, ir, vis,mask):
        fused_img, seg_map = self(ir,vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)


        return enhance_loss

    def _fusion_loss(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _fusion_loss_wogan(self, ir, vis, mask,):
        fused_img = self.forward_fusion(ir, vis)
        vis = RGB2YCrCb(vis)
        enhance_loss = self._criterion(ir, vis, fused_img,mask)
        return enhance_loss
    def _detection_loss(self, ir, vis, labels):
        fused_img, seg_map = self.forward_object(ir, vis)
        outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self.seg_loss(outputs, labels.type(torch.long))
        return denoise_loss

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()


### Cell_chain

## Cell_Decomposition


