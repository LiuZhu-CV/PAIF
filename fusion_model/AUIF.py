import torch
from torch import nn

channel = 64


class Cov1(nn.Module):
    def __init__(self):
        super(Cov1, self).__init__()
        self.cov1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, channel, 3, padding=0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.cov1(x)


class Cov2(nn.Module):
    def __init__(self):
        super(Cov2, self).__init__()
        self.cov2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.cov2(x)


class Cov3(nn.Module):
    def __init__(self):
        super(Cov3, self).__init__()
        self.cov3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            # nn.PReLU(),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.cov3(x)


class Cov4(nn.Module):
    def __init__(self):
        super(Cov4, self).__init__()
        self.cov4 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            # nn.PReLU(),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.cov4(x)


class Cov5(nn.Module):
    def __init__(self):
        super(Cov5, self).__init__()
        self.cov5 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.cov5(x)


class Cov6(nn.Module):
    def __init__(self):
        super(Cov6, self).__init__()
        self.cov6 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.cov6(x)


class Cov7(nn.Module):
    def __init__(self):
        super(Cov7, self).__init__()
        self.cov7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channel * 2, 1, 3, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.cov7(x)


class AE_Encoder(nn.Module):
    def __init__(self):
        super(AE_Encoder, self).__init__()
        self.cov1 = Cov1()
        self.cov2 = Cov2()
        self.cov3 = Cov3()
        self.cov4 = Cov4()

    def forward(self, data_train):
        feature_1 = self.cov1(data_train)
        feature_2 = self.cov2(feature_1)
        feature_B = self.cov3(feature_2)
        feature_D = self.cov4(feature_2)
        return feature_1, feature_2, feature_B, feature_D


class AE_Decoder(nn.Module):
    def __init__(self):
        super(AE_Decoder, self).__init__()
        self.cov5 = Cov5()
        self.cov6 = Cov6()
        self.cov7 = Cov7()

    def forward(self, feature_1, feature_2, feature_B, feature_D):
        Output1 = self.cov5(torch.cat([feature_B, feature_D], 1))
        Output2 = self.cov6(torch.cat([Output1, feature_2], 1))
        Output3 = self.cov7(torch.cat([Output2, feature_1], 1))
        return Output3

class DID(nn.Module):
    def __init__(self):
        super(DID, self).__init__()
        self.num_channels = 2
        self.num_features = 44
        self.growth = 44
        self.AE_Encoder1 =  AE_Encoder()
        self.AE_Encoder2 =  AE_Encoder()
        self.AE_Decoder1 =  AE_Decoder()

    def forward(self, x_over, x_under):
        F_i1, F_i2, F_ib, F_id = self.AE_Encoder1(x_over)
        F_v1, F_v2, F_vb, F_vd =  self.AE_Encoder2(x_under)
        F_b = (F_ib + F_vb)/2
        F_d = (F_id + F_vd)/2
        F_1 = (F_i1 + F_v1)/2
        F_2 = (F_i2 + F_v2)/2
        Out = self.AE_Decoder1(F_1, F_2, F_b, F_d)

        return Out

