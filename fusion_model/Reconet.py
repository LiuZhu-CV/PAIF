import torch
from torch import nn, Tensor

# from modules.layers.d_group import DGroup
import torch
from torch import nn, Tensor


class ConvGroup(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_bn: bool):
        super().__init__()

        # (Conv2d, BN, GELU)
        dim = conv.out_channels
        self.group = nn.Sequential(
            conv,
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.group(x)

class DGroup(nn.Module):
    """
    [channels: dim, s] -> DGroup -> [channels: 1, s]
    """

    def __init__(self, in_c: int, out_c: int, dim: int, k_size: int, use_bn: bool):
        super().__init__()

        # conv_d: [dim] -> [1]
        self.conv_d = nn.ModuleList([
            ConvGroup(nn.Conv2d(in_c, dim, kernel_size=k_size, padding='same', dilation=(i + 1)), use_bn=use_bn)
            for i in range(3)
        ])

        # conv_s: [3] -> [1]
        self.conv_s = nn.Sequential(
            nn.Conv2d(3 * dim, out_c, kernel_size=3, padding='same'),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        f_in = x
        # conv_d
        f_x = [conv(f_in) for conv in self.conv_d]
        # suffix
        f_t = torch.cat(f_x, dim=1)
        f_out = self.conv_s(f_t)
        return f_out



class ReCoNet(nn.Module):
    def __init__(self, depth: int, dim: int, use_bn: bool):
        super(ReCoNet,self).__init__()
        self.depth = depth

        # attention layer: [2] -> [1], [2] -> [1]
        self.att_a_conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)
        self.att_b_conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)

        # dilation fuse
        self.decoder = DGroup(in_c=3, out_c=1, dim=dim, k_size=3, use_bn=use_bn)

    def forward(self, i_1, i_2 , init_f: str = 'max', show_detail: bool = False):
        # recurrent subnetwork
        # generate f_0 with initial function

        i_f = [torch.max(i_1, i_2) if init_f == 'max' else (i_1 + i_2) / 2]
        att_a, att_b = [], []

        # loop in subnetwork
        for _ in range(self.depth):
            i_f_x, att_a_x, att_b_x = self._sub_forward(i_1, i_2, i_f[-1])
            i_f.append(i_f_x), att_a.append(att_a_x), att_b.append(att_b_x)

        # return as expected
        return (i_f, att_a, att_b) if show_detail else i_f[-1]

    def _sub_forward(self, i_1: Tensor, i_2: Tensor, i_f: Tensor):
        # attention
        att_a = self._attention(self.att_a_conv, i_1, i_f)
        att_b = self._attention(self.att_b_conv, i_2, i_f)

        # focus on attention
        i_1_w = i_1 * att_a
        i_2_w = i_2 * att_b

        # dilation fuse
        i_in = torch.cat([i_1_w, i_f, i_2_w], dim=1)
        i_out = self.decoder(i_in)

        # return fusion result of current recurrence
        return i_out, att_a, att_b

    @staticmethod
    def _attention(att_conv, i_a, i_b):
        i_in = torch.cat([i_a, i_b], dim=1)
        i_max, _ = torch.max(i_in, dim=1, keepdim=True)
        i_avg = torch.mean(i_in, dim=1, keepdim=True)
        i_in = torch.cat([i_max, i_avg], dim=1)
        i_out = att_conv(i_in)
        return torch.sigmoid(i_out)