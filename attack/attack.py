import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import lpips
import pytorch_ssim
import cv2
from core.loss import Fusionloss2

upper_limit = 1
lower_limit = 0


def YCrCb2RGB(input_im):
    '''
    YCrCb format to RGB format
  '''
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
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


def RGB2YCrCb(input_im):
    '''
    RGB format to RGB format
  '''
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
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


def clamp(X, lower_limit, upper_limit):
    '''
    Keep X in lower_limit and upper_limit
  '''
    return torch.max(torch.min(X, upper_limit), lower_limit)


def trans_format(image_fusion, images_vis):
    images_vis_ycrcb = RGB2YCrCb(images_vis)
    fusion_ycrcb = torch.cat(
        (image_fusion, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
        dim=1,
    )
    fusion_image = YCrCb2RGB(fusion_ycrcb)

    # keep pixel in (0,1)
    ones = torch.ones_like(fusion_image)
    zeros = torch.zeros_like(fusion_image)
    fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
    fused_image = torch.where(fusion_image < zeros, zeros, fusion_image)
    # fused_image = fusion_image.cpu().detach().numpy()

    # fused_image = np.uint8(255.0 * fused_image)

    # fused_image = fused_image.transpose((0, 2, 3, 1))  ### (N,C,H,W) -> (N,H,W,C) N表示图像的下标
    fused_image = (fused_image - torch.min(fused_image)) / (
            torch.max(fused_image) - torch.min(fused_image)
    )

    # fused_image = np.uint8(255.0 * fused_image)
    # fused_image = torch.tensor(fused_image)
    # fused_image = fused_image.cuda()
    return fused_image


class Seg_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = model
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, outputs, labels):
        # fused, seg_map = self.model(ir, vis)
        # vis = RGB2YCrCb(vis)
        # outputs = F.interpolate(seg_map, size=labels.shape[1:], mode='bilinear', align_corners=False)
        denoise_loss = self._loss(outputs, labels.type(torch.long))
        return denoise_loss


def pgd_attack_ir(model, X_vis, X_ir, X_fusion, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50, restarts=1,
                  attack_loss='l_2'):
    '''
    attack infrared image by PGD, default loss is l_2

    model: image fusion model
    X_vis: original vision image
    X_ir: original Infrared image
    X_fusion: fusion image fused by X_vis and X_ir (without perturbation)
  '''
    torch.cuda.empty_cache()
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    print(device)

    if attack_loss == 'l_2':
        criterion = nn.MSELoss()
    elif attack_loss == 'l_1':
        criterion = nn.L1Loss()
    elif attack_loss == 'l_ssim':
        criterion = pytorch_ssim.SSIM()
    elif attack_loss == 'l_seg':  # attack segmentatinon loss
        criterion = Seg_loss()  # Entropy loss
    else:
        pass

    X = X_ir  # X is attacked image
    max_loss = torch.zeros(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    output = X_fusion  # original fusion image
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)  # initialize delta
        delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
        delta = Variable(delta, requires_grad=True)
        # delta.requires_grad = True
        for _ in range(attack_iters):
            # image_fusion, seg1 = model.forward(images_ir_attacked,images_vis)
            with torch.enable_grad():
                logits, seg_map = model.forward(X + delta, X_vis)
                if attack_loss == 'l_seg':
                    outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear',
                                            align_corners=False)  # segmentation output
                    loss = criterion(outputs, label)
                else:
                    # 将robust_output(logits output)转换为RGB格式图像
                    robust_output = trans_format(logits, X_vis)  # 添加扰动之后的红外，原可见光图像
                    loss = criterion(robust_output, output)
            # loss.backward()
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g.data), min=-epsilon, max=epsilon)
            d = torch.clamp(d, min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d

    return delta


def pgd_attack_vision(model, X_vis, X_ir, X_fusion, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50,
                      restarts=1, attack_loss='l_seg'):
    '''
    attack vision image by PGD

    model: image fusion model
    X_vis: original vision image
    X_ir: original Infrared image
    X_fusion: fusion image fused by X_vis and X_ir (without perturbation)
  '''
    torch.cuda.empty_cache()

    if attack_loss == 'l_2':
        criterion = nn.MSELoss()
    elif attack_loss == 'l_1':
        criterion = nn.L1Loss()
    elif attack_loss == 'l_seg':  # attack segmentatinon loss
        criterion = Seg_loss()  # Entropy loss
    else:
        pass

    X = X_vis  # X is attacked image
    max_loss = torch.zeros(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    output = X_fusion  # original fusion image
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)  # initialize delta
        delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
        delta = Variable(delta, requires_grad=True)
        # delta.requires_grad = True
        for _ in range(attack_iters):
            # image_fusion, seg1 = model.forward(images_ir_attacked,images_vis)
            with torch.enable_grad():
                logits, seg_map = model.forward(X_ir, X + delta)
                if attack_loss == 'l_seg':
                    outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear',
                                            align_corners=False)  # segmentation output
                    loss = criterion(outputs, label)
                else:
                    # 将robust_output(logits output)转换为RGB格式图像
                    robust_output = trans_format(logits, X_vis)  # 添加扰动之后的红外，原可见光图像
                    loss = -criterion(robust_output, output)
                grad = torch.autograd.grad(loss, [delta])[0].detach()
                d = delta
                g = grad
                x = X
                d = torch.clamp(d + alpha * torch.sign(g.data), min=-epsilon, max=epsilon)
                d = torch.clamp(d, min=-epsilon, max=epsilon)
                d = clamp(d, lower_limit - x, upper_limit - x)
                delta.data = d

    return delta


def get_ir_mask(X_ir):
    '''
    注意将mask转为tensor的格式
  '''
    ir_img = np.uint8(255.0 * X_ir)
    ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
    ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2YCrCb)
    ir_img = ir_img[:, :, 0]
    ir_mask = map_generate3(ir_img)
    ir_mask = torch.from_numpy(ir_mask).cuda()
    ir_mask = ir_mask.to(torch.float32)
    # ir_mask = ir_mask / 255.
    return ir_mask


def fgsm_ir(model, X_vis, X_ir, X_fusion, epsilon=8 / 255., restarts=1, attack_loss='l_2', with_mask=False):
    '''
    attack infrared image by FGSM, default loss is l_2

    model: image fusion model
    X_vis: original vision image
    X_ir: original Infrared image
    X_fusion: fusion image fused by X_vis and X_ir (without perturbation)
  '''
    #  X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)
    torch.cuda.empty_cache()

    if attack_loss == 'l_2':
        criterion = nn.MSELoss()
    elif attack_loss == 'l_1':
        criterion = nn.L1Loss()
    elif attack_loss == 'l_ssim':
        criterion = pytorch_ssim.SSIM()
    elif attack_loss == 'l_entropy':
        criterion = nn.CrossEntropyLoss()
    elif attack_loss == 'lpips':  # useless
        criterion = lpips.LPIPS(net='alex').cuda()
    else:
        pass

    X = X_ir
    output = X_fusion
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
        delta = Variable(delta, requires_grad=True)
        if with_mask:
            black_X = torch.zeros_like(X).cuda()  # 0图
            img_map = torch.squeeze(X_ir)
            img_map = img_map.cpu().detach().numpy()
            ir_mask = get_ir_mask(img_map)
            # ir_mask.unsqueeze(0)
            # ir_mask.unsqueeze(1)
            delta = delta * ir_mask
            delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
        with torch.enable_grad():
            delta = Variable(delta, requires_grad=True)
            X_ = X + delta
            logits, seg = model.forward(X_, X_vis)
            robust_output = trans_format(logits, X_vis)
            if attack_loss != 'lpips':
                # loss2 = nn.MSELoss()
                # loss = criterion(robust_output, output) - 0.8 * loss2(robust_output, black_X)
                loss = -criterion(robust_output, black_X)
            else:
                loss = criterion(2 * robust_output - 1, 2 * robust_output - 1)
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            delta = torch.clamp(delta.data + epsilon * grad.data.sign(), lower_limit - X, upper_limit - X)

        if with_mask:
            delta = delta * ir_mask
    return delta


def seg_pgd(model, X_vis, X_ir, X_fusion, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50, restarts=1,
            attack_loss='l_seg', attack_mode='vis'):
    '''
    sge_pgd 中使用的epsilon为0.005 (1/200)
  '''
    torch.cuda.empty_cache()
    # loss function
    if attack_loss == 'l_seg':
        criterion = Seg_loss()
    else:
        pass

    if attack_mode == 'vis':
        X = X_vis
    else:
        X = X_ir  # X is attacked image

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
        delta = Variable(delta, requires_grad=True)

        for i in range(attack_iters):
            with torch.enable_grad():
                if attack_mode == 'vis':
                    _, seg_map = model(X_ir, X + delta)
                else:
                    _, seg_map = model(X + delta, X_vis)
                outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)

                lamb = (i - 1) / (attack_iters * 2)

                pred = torch.max(outputs, 1).values
                pred = torch.unsqueeze(pred, 1)
                # for true classification
                mask_t = pred == torch.unsqueeze(label, 1)
                mask_t = torch.squeeze(mask_t, 1).int()
                np_mask_t = torch.unsqueeze(mask_t, 1)
                # for false classification
                mask_f = pred != torch.unsqueeze(label, 1)
                mask_f = torch.squeeze(mask_f, 1).int()
                np_mask_f = torch.unsqueeze(mask_f, 1)

                loss_t = (1 - lamb) * criterion(np_mask_t * outputs, label)
                loss_f = lamb * criterion(np_mask_f * outputs, label)
                loss = loss_t + loss_f
                # loss = -1 * loss

            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g.data), min=-epsilon, max=epsilon)
            d = torch.clamp(d, min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d

    return delta


def cos_pgd(model, X_vis, X_ir, X_fusion, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50, restarts=1,
            attack_loss='l_seg', attack_mode='vis'):
    torch.cuda.empty_cache()
    if attack_loss == 'l_seg':
        criterion = Seg_loss()
    else:
        pass

    if attack_mode == 'vis':
        X = X_vis
    else:
        X = X_ir

    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
        delta = Variable(delta, requires_grad=True)

        for _ in range(attack_iters):
            with torch.enable_grad():
                if attack_mode == 'vis':
                    _, seg_map = model(X_ir, X + delta)
                else:
                    _, seg_map = model(X + delta, X_vis)

                outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
                pred = torch.max(outputs, 1).values
                pred = torch.squeeze(pred).flatten()

                _label = torch.squeeze(label).flatten()

                cossim = torch.nn.functional.cosine_similarity(pred, _label, dim=0)
                loss = cossim * criterion(outputs, label)
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g.data), min=-epsilon, max=epsilon)
            d = torch.clamp(d, min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d

    return delta





def attack_both(model, X_vis, X_ir, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50,
                restarts=1, attack_loss='l_seg', attack_mode='vis', attack_way='PGD'):
    '''
    攻击分割loss，同时在红外以及可见光图像上添加扰动
  '''

    torch.cuda.empty_cache()
    if attack_loss == 'l_seg':
        criterion = Seg_loss()
    elif attack_loss == 'l_2':
        criterion = nn.MSELoss()
    else:
        print('dont give the correct loss function')
        return -1

    for _ in range(restarts):
        delta_ir = torch.zeros_like(X_ir).cuda()
        delta_ir.uniform_(-epsilon, epsilon)
        delta_ir = torch.clamp(delta_ir, lower_limit - X_ir, upper_limit - X_ir)
        delta_ir = Variable(delta_ir, requires_grad=True)

        delta_vis = torch.zeros_like(X_vis).cuda()
        delta_vis.uniform_(-epsilon, epsilon)
        delta_vis = torch.clamp(delta_vis, lower_limit - X_vis, upper_limit - X_vis)
        delta_vis = Variable(delta_vis, requires_grad=True)

        for i in range(attack_iters):
            with torch.enable_grad():
                _, seg_map = model(X_ir + delta_ir, X_vis + delta_vis)
                outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
                if attack_way == 'PGD':
                    loss = criterion(outputs, label)
                elif attack_way == 'segPGD':
                    lamb = (i - 1) / (attack_iters * 2)

                    pred = torch.max(outputs, 1).values
                    pred = torch.unsqueeze(pred, 1)
                    # for true classification
                    mask_t = pred == torch.unsqueeze(label, 1)
                    mask_t = torch.squeeze(mask_t, 1).int()
                    np_mask_t = torch.unsqueeze(mask_t, 1)
                    # for false classification
                    mask_f = pred != torch.unsqueeze(label, 1)
                    mask_f = torch.squeeze(mask_f, 1).int()
                    np_mask_f = torch.unsqueeze(mask_f, 1)

                    loss_t = (1 - lamb) * criterion(np_mask_t * outputs, label)
                    loss_f = lamb * criterion(np_mask_f * outputs, label)
                    loss = loss_t + loss_f
                elif attack_way == 'cosPGD':
                    pred = torch.max(outputs, 1).values
                    pred = torch.squeeze(pred).flatten()

                    _label = torch.squeeze(label).flatten()

                    cossim = torch.nn.functional.cosine_similarity(pred, _label, dim=0)
                    loss = cossim * criterion(outputs, label)
                elif attack_way == 'newPGD':
                    pred = torch.max(outputs, 1).values
                    pred = torch.unsqueeze(pred, 1)
                    # for true classification
                    mask_t = pred == torch.unsqueeze(label, 1)
                    mask_t = torch.squeeze(mask_t, 1).int()
                    np_mask_t = torch.unsqueeze(mask_t, 1)
                    # for false classification
                    mask_f = pred != torch.unsqueeze(label, 1)
                    mask_f = torch.squeeze(mask_f, 1).int()
                    np_mask_f = torch.unsqueeze(mask_f, 1)

                    _label = torch.squeeze(label).flatten()

                    pred_t = np_mask_t * outputs
                    pred_t = torch.max(outputs, 1).values
                    pred_t = torch.squeeze(pred).flatten()

                    pred_f = np_mask_f * outputs
                    pred_f = torch.max(outputs, 1).values
                    pred_f = torch.squeeze(pred).flatten()

                    cossim_t = torch.nn.functional.cosine_similarity(pred_t, _label, dim=0)
                    cossim_f = torch.nn.functional.cosine_similarity(pred_f, _label, dim=0)

                    loss = (cossim_t / cossim_f) * criterion(outputs, label)

            loss.backward()
            # grad_ir = torch.autograd.grad(loss, [delta_ir])[0].detach()
            # grad_vis = torch.autograd.grad(loss, [delta_vis])[0].detach()
            d_ir = torch.clamp(delta_ir.data + alpha * torch.sign(delta_ir.grad.data), min=-epsilon, max=epsilon)
            d_ir = torch.clamp(d_ir, min=-epsilon, max=epsilon)
            d_ir = torch.clamp(d_ir, min=lower_limit - X_ir, max=upper_limit - X_ir)
            delta_ir.data = d_ir

            d_vis = torch.clamp(delta_vis.data + alpha * torch.sign(delta_vis.grad.data), min=-epsilon, max=epsilon)
            d_vis = torch.clamp(d_vis, min=-epsilon, max=epsilon)
            d_vis = torch.clamp(d_vis, min=lower_limit - X_vis, max=upper_limit - X_vis)
            delta_vis.data = d_vis

    return delta_ir, delta_vis


def attack_vis(model, X_vis, X_ir, X_fusion, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50,
               restarts=1, attack_loss='l_seg', attack_mode='vis', attack_way='PGD'):
    '''
    攻击分割loss，只攻击可见光图片
  '''

    torch.cuda.empty_cache()
    if attack_loss == 'l_seg':
        criterion = Seg_loss()
    elif attack_loss == 'l_2':
        criterion = nn.MSELoss()
    else:
        print('dont give the correct loss function')
        return -1

    for _ in range(restarts):
        delta_vis = torch.zeros_like(X_vis).cuda()
        delta_vis.uniform_(-epsilon, epsilon)
        delta_vis = torch.clamp(delta_vis, lower_limit - X_vis, upper_limit - X_vis)
        delta_vis = Variable(delta_vis, requires_grad=True)

        for i in range(attack_iters):
            with torch.enable_grad():
                _, seg_map = model(X_ir, X_vis + delta_vis)
                outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
                if attack_way == 'PGD':
                    loss = criterion(outputs, label)
                elif attack_way == 'segPGD':
                    lamb = (i - 1) / (attack_iters * 2)

                    pred = torch.max(outputs, 1).values
                    pred = torch.unsqueeze(pred, 1)
                    # for true classification
                    mask_t = pred == torch.unsqueeze(label, 1)
                    mask_t = torch.squeeze(mask_t, 1).int()
                    np_mask_t = torch.unsqueeze(mask_t, 1)
                    # for false classification
                    mask_f = pred != torch.unsqueeze(label, 1)
                    mask_f = torch.squeeze(mask_f, 1).int()
                    np_mask_f = torch.unsqueeze(mask_f, 1)

                    loss_t = (1 - lamb) * criterion(np_mask_t * outputs, label)
                    loss_f = lamb * criterion(np_mask_f * outputs, label)
                    loss = loss_t + loss_f
                elif attack_way == 'cosPGD':
                    pred = torch.max(outputs, 1).values
                    pred = torch.squeeze(pred).flatten()

                    _label = torch.squeeze(label).flatten()

                    cossim = torch.nn.functional.cosine_similarity(pred, _label, dim=0)
                    loss = cossim * criterion(outputs, label)
                elif attack_way == 'newPGD':
                    pred = torch.max(outputs, 1).values
                    pred = torch.unsqueeze(pred, 1)
                    # for true classification
                    mask_t = pred == torch.unsqueeze(label, 1)
                    mask_t = torch.squeeze(mask_t, 1).int()
                    np_mask_t = torch.unsqueeze(mask_t, 1)
                    # for false classification
                    mask_f = pred != torch.unsqueeze(label, 1)
                    mask_f = torch.squeeze(mask_f, 1).int()
                    np_mask_f = torch.unsqueeze(mask_f, 1)

                    _label = torch.squeeze(label).flatten()

                    pred_t = np_mask_t * outputs
                    pred_t = torch.max(outputs, 1).values
                    pred_t = torch.squeeze(pred).flatten()

                    pred_f = np_mask_f * outputs
                    pred_f = torch.max(outputs, 1).values
                    pred_f = torch.squeeze(pred).flatten()

                    cossim_t = torch.nn.functional.cosine_similarity(pred_t, _label, dim=0)
                    cossim_f = torch.nn.functional.cosine_similarity(pred_f, _label, dim=0)

                    loss = (cossim_t / cossim_f) * criterion(outputs, label)

            loss.backward()
            # grad_ir = torch.autograd.grad(loss, [delta_ir])[0].detach()
            # grad_vis = torch.autograd.grad(loss, [delta_vis])[0].detach()
            d_vis = torch.clamp(delta_vis.data + alpha * torch.sign(delta_vis.grad.data), min=-epsilon, max=epsilon)
            d_vis = torch.clamp(d_vis, min=-epsilon, max=epsilon)
            d_vis = torch.clamp(d_vis, min=lower_limit - X_vis, max=upper_limit - X_vis)
            delta_vis.data = d_vis

    return delta_vis


def attack_ir(model, X_vis, X_ir, X_fusion, label, epsilon=8 / 255., alpha=2 / 255., attack_iters=50,
              restarts=1, attack_loss='l_seg', attack_mode='vis', attack_way='PGD'):
    torch.cuda.empty_cache()
    if attack_loss == 'l_seg':
        criterion = Seg_loss()
    elif attack_loss == 'l_2':
        criterion = nn.MSELoss()
    else:
        print('dont give the correct loss function')
        return -1

    for _ in range(restarts):
        delta_ir = torch.zeros_like(X_ir).cuda()
        delta_ir.uniform_(-epsilon, epsilon)
        delta_ir = torch.clamp(delta_ir, lower_limit - X_ir, upper_limit - X_ir)
        delta_ir = Variable(delta_ir, requires_grad=True)

        for i in range(attack_iters):
            with torch.enable_grad():
                _, seg_map = model(X_ir + delta_ir, X_vis)
                outputs = F.interpolate(seg_map, size=label.shape[1:], mode='bilinear', align_corners=False)
                if attack_way == 'PGD':
                    loss = criterion(outputs, label)
                elif attack_way == 'segPGD':
                    lamb = (i - 1) / (attack_iters * 2)

                    pred = torch.max(outputs, 1).values
                    pred = torch.unsqueeze(pred, 1)
                    # for true classification
                    mask_t = pred == torch.unsqueeze(label, 1)
                    mask_t = torch.squeeze(mask_t, 1).int()
                    np_mask_t = torch.unsqueeze(mask_t, 1)
                    # for false classification
                    mask_f = pred != torch.unsqueeze(label, 1)
                    mask_f = torch.squeeze(mask_f, 1).int()
                    np_mask_f = torch.unsqueeze(mask_f, 1)

                    loss_t = (1 - lamb) * criterion(np_mask_t * outputs, label)
                    loss_f = lamb * criterion(np_mask_f * outputs, label)
                    loss = loss_t + loss_f
                elif attack_way == 'cosPGD':
                    pred = torch.max(outputs, 1).values
                    pred = torch.squeeze(pred).flatten()

                    _label = torch.squeeze(label).flatten()

                    cossim = torch.nn.functional.cosine_similarity(pred, _label, dim=0)
                    loss = cossim * criterion(outputs, label)
                elif attack_way == 'newPGD':
                    pred = torch.max(outputs, 1).values
                    pred = torch.unsqueeze(pred, 1)
                    # for true classification
                    mask_t = pred == torch.unsqueeze(label, 1)
                    mask_t = torch.squeeze(mask_t, 1).int()
                    np_mask_t = torch.unsqueeze(mask_t, 1)
                    # for false classification
                    mask_f = pred != torch.unsqueeze(label, 1)
                    mask_f = torch.squeeze(mask_f, 1).int()
                    np_mask_f = torch.unsqueeze(mask_f, 1)

                    _label = torch.squeeze(label).flatten()

                    pred_t = np_mask_t * outputs
                    pred_t = torch.max(outputs, 1).values
                    pred_t = torch.squeeze(pred).flatten()

                    pred_f = np_mask_f * outputs
                    pred_f = torch.max(outputs, 1).values
                    pred_f = torch.squeeze(pred).flatten()

                    cossim_t = torch.nn.functional.cosine_similarity(pred_t, _label, dim=0)
                    cossim_f = torch.nn.functional.cosine_similarity(pred_f, _label, dim=0)

                    loss = (cossim_t / cossim_f) * criterion(outputs, label)

            loss.backward()
            # grad_ir = torch.autograd.grad(loss, [delta_ir])[0].detach()
            # grad_vis = torch.autograd.grad(loss, [delta_vis])[0].detach()
            d_ir = torch.clamp(delta_ir.data + alpha * torch.sign(delta_ir.grad.data), min=-epsilon, max=epsilon)
            d_ir = torch.clamp(d_ir, min=-epsilon, max=epsilon)
            d_ir = torch.clamp(d_ir, min=lower_limit - X_ir, max=upper_limit - X_ir)
            delta_ir.data = d_ir

    return delta_ir
