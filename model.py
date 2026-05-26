import torch
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt


#Coded Aperture
class ApertureMask(nn.Module):
    def __init__(self,  view_point_num, aperture_num):
        super().__init__()
        self.view_point_num = view_point_num    
        self.aperture_num = aperture_num
        
        k = 1 / self.view_point_num
        param = torch.empty(self.view_point_num,int(self.aperture_num)-1).uniform_(-sqrt(k), sqrt(k))
        self.param = nn.Parameter(param)
        self.W = torch.empty(self.view_point_num,int(self.aperture_num)).uniform_(-sqrt(k), sqrt(k))
        self.scale = 1.02

    def forward(self, LF, epoch):
        W_rest = torch.sigmoid(self.param * (self.scale ** epoch))
        W_first = torch.zeros(self.view_point_num, 1, device=W_rest.device, dtype=W_rest.dtype)
        self.W = torch.cat((W_first, W_rest), dim=1)
        A = self.W[:,0].unsqueeze(1)
        for t in range(int(self.aperture_num)-1):
            A = torch.cat((A,self.W[:,t+1].unsqueeze(1)), dim=1)
        I = torch.permute(torch.matmul(torch.permute(LF,(0,2,3,1)),A),(0,3,1,2))
        
        return I

    def get_mask(self, num):
        W_num = num
        mask = self.W[:,num]
        return mask.detach()

    def clip_mask(self):
        self.W.data.clamp_(0.0, 1.0)


# Gradient-pass-through technique
class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.to(torch.int64).to(torch.float32)
    @staticmethod
    def backward(ctx, g):
        return g 


# Event simulator
class EventCam(torch.nn.Module):
    def __init__(self, aperture_num, threshold, eps, noise, th_noise, device):
        super().__init__()
        self.aperture_num = aperture_num
        self.thresh = threshold
        self.eps = eps
        self.noise = noise
        self.th_noise = th_noise
        self.device = device

    def forward(self, imgs):
        B,C,H,W = imgs.shape
        zero_img = torch.zeros_like(imgs[:,0,:,:])
        V_th = torch.log(zero_img+self.eps)   #Reference level initialization

        event_images = torch.zeros(B,self.aperture_num-1,H,W).to(self.device)
        for t in range(self.aperture_num-1):
            noise = self.noise * torch.randn((B,H,W)).to(self.device)
            th_noise = self.th_noise * torch.randn((B,H,W)).to(self.device)
            real_th = self.thresh + th_noise
            real_th[real_th<0.01] = 0.01

            V_diff = torch.log(imgs[:,t+1,:,:]+self.eps) - V_th + noise
            V_diff = V_diff/real_th
            event_images[:,t,:,:] = RoundNoGradient.apply(V_diff)
            V_th = V_th + self.thresh*event_images[:,t,:,:]

        return event_images
    

# RecNet
class Reconstruct(torch.nn.Module):
    def __init__(self, view_point_num, aperture_num, ch_mid):
        super(Reconstruct, self).__init__()
        self.view_point_num = view_point_num
        self.input_num = aperture_num-1
        self.conv1 = nn.Conv2d(self.input_num, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, self.view_point_num, 5, padding=2)

        self.ch_in = view_point_num
        self.ch_mid = ch_mid
        self.ch_out = view_point_num

        self.l1 = torch.nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l2 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l3 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l4 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l5 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l6 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l7 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l8 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l9 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l10 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l11 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l12 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l13 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l14 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l15 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l16 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l17 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l18 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l19 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_mid, kernel_size=3, stride=1, padding=1)
        self.l20 = torch.nn.Conv2d(in_channels=self.ch_mid, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)

        y1 = F.relu(self.l1(f3))
        y2 = F.relu(self.l2(y1))
        y3 = F.relu(self.l3(y2))
        y4 = F.relu(self.l4(y3))
        y5 = F.relu(self.l5(y4))
        y6 = F.relu(self.l6(y5))
        y7 = F.relu(self.l7(y6))
        y8 = F.relu(self.l8(y7))
        y9 = F.relu(self.l9(y8))
        y10 = F.relu(self.l10(y9))
        y11 = F.relu(self.l11(y10))
        y12 = F.relu(self.l12(y11))
        y13 = F.relu(self.l13(y12))
        y14 = F.relu(self.l14(y13))
        y15 = F.relu(self.l15(y14))
        y16 = F.relu(self.l16(y15))
        y17 = F.relu(self.l17(y16))
        y18 = F.relu(self.l18(y17))
        y19 = F.relu(self.l19(y18))
        y20 = self.l20(y19)
        return y20 + f3
        