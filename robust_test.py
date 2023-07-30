'''测试对抗训练模型的鲁棒性'''
import torch
import os
import numpy as np
from attack.attack import attack_both, attack_ir, attack_vis
from TaskFusion_dataset2 import Fusion_dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from core.model_fusion_auto import Network_MM_SearchedFusion, Network_MM_Searched
from core import Fusionloss_grad2
import torch.nn.functional as F
from collections import namedtuple
import argparse
from PIL import Image
import cv2
from tqdm import tqdm

from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix

from core.model_fusion_auto import Network_MM_Searched as Network
from utils import eval_seg
from attack.attack import attack_both
from utils.optimizer import PolyWarmupAdamW
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
parser.add_argument('--batch_size', '-B', type=int, default=1)
parser.add_argument('--gpu', '-G', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)

### attack setting
parser.add_argument('--epsilon', default=8/255., type=float)
parser.add_argument('--alpha', default=2/255., type=float)
parser.add_argument('--attack_iters', default=5, type=int)
parser.add_argument('--attack_loss', default='l_seg', type=str)
parser.add_argument('--attack_way',default='PGD', type=str)
parser.add_argument('--with_mask',default=False, type=bool)
parser.add_argument('--attack_mode',default='both',type=str, help='attack mode, ir|vis|both')
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
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
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
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

def val_segformer_robust(model, strategy):
  conf_total = np.zeros((9, 9))
  h = 480
  w = 640
  model.eval().cuda()
  vi_path = '/user33/objectdetection/test_all/Visible/'
  ir_path = '/user33/objectdetection/test_all/Infrared/'
  label_path = '/user33/objectdetection/test_all/Label/'
  test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path, label_path=label_path)
  test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
  )
  test_loader.n_iter = len(test_loader)

  with torch.no_grad():
    for attack_mode in ['both']: #'both', 'ir', 'vis'
      for _ , (images_vis, images_ir, label, name) in tqdm(enumerate(test_loader)):
        images_vis = Variable(images_vis)
        images_ir = Variable(images_ir)
        label = Variable(label)

        if args.gpu >= 0:
          images_vis = images_vis.cuda()
          images_ir = images_ir.cuda()
          label = label.cuda()

        save_root = './attack/{6}_{0}_{1}{2}_{3}_{4}_{5}/'.format(args.attack_loss,args.attack_way,args.attack_iters,int(args.epsilon*255),int(args.alpha*255),attack_mode,strategy)
        if attack_mode != 'both':
          attacked_path = save_root + '{0}_attacked/'.format(attack_mode)
          fused_path = save_root + 'fused_attacked/'
          seg_path = save_root + 'seg_attacked/'
          os.makedirs(attacked_path, mode=0o777, exist_ok=True)
          os.makedirs(fused_path, mode=0o777, exist_ok=True)
          os.makedirs(seg_path, mode=0o777, exist_ok=True)
        else:
          ir_attacked_path = save_root + 'ir_attacked/'
          vis_attacked_path = save_root + 'vis_attacked/'
          fused_path = save_root + 'fused_attacked/'
          seg_path = save_root + 'seg_attacked/'
          os.makedirs(ir_attacked_path, mode=0o777, exist_ok=True)
          os.makedirs(vis_attacked_path, mode=0o777, exist_ok=True)
          os.makedirs(fused_path, mode=0o777, exist_ok=True)
          os.makedirs(seg_path, mode=0o777, exist_ok=True)

        if attack_mode == 'both':
          d_ir, d_vi = attack_both(model, X_vis=images_vis, X_ir=images_ir, label=label,
                                      attack_loss=args.attack_loss, attack_iters=args.attack_iters, epsilon=args.epsilon, alpha=args.alpha)
          vi_attacked = images_vis + d_vi
          ir_attacked = images_ir + d_ir
          ## save attacked image 
          img_ir = ir_attacked.squeeze().cpu().detach().numpy()
          img_ir = np.uint8(255.0 * img_ir)  
          img_ir = Image.fromarray(img_ir)       
          save_path = os.path.join(ir_attacked_path, name[0])
          img_ir.save(save_path)

          img_ = vi_attacked.squeeze().cpu().detach().numpy()
          img_ = np.uint8(255.0 * img_)
          img_ = img_.transpose(1, 2, 0)
          img_ = Image.fromarray(img_)
          save_path = os.path.join(vis_attacked_path, name[0])
          img_.save(save_path)
          # print("### {0} attack successfully".format(name))

          vi_attacked = Variable(vi_attacked)
          ir_attacked = Variable(ir_attacked)
          image_fusion, seg1 = model.forward(ir_attacked, vi_attacked)
          # print("----attack both image mode " + name[0])
        elif attack_mode == 'vis':
          d_vi = attack_vis(model, X_vis=images_vis, X_ir=images_ir, X_fusion=image_fusion, label=label,
                                      attack_loss=args.attack_loss, attack_iters=args.attack_iters, epsilon=args.epsilon, alpha=args.alpha, attack_mode=attack_mode, attack_way='newPGD')
          vi_attacked = images_vis + d_vi
          _, seg1 = model.forward(images_ir, vi_attacked)
          # print("----attack visible image mode " + name[0])
        elif attack_mode == 'ir':
          d_vi = attack_ir(model, X_vis=images_vis, X_ir=images_ir, X_fusion=image_fusion, label=label,
                                      attack_loss=args.attack_loss, attack_iters=args.attack_iters, epsilon=args.epsilon, alpha=args.alpha, attack_mode=attack_mode, attack_way='newPGD')
          ir_attacked = images_ir + d_ir
          _, seg1 = model.forward(ir_attacked, images_vis)
          # print("----attack infrared image mode " + name[0])

        seg1 = F.interpolate(seg1, size=label.shape[1:], mode='bilinear', align_corners=False)
        images_vis_ycrcb = RGB2YCrCb(images_vis) 
        fusion_ycrcb = torch.cat(
            (image_fusion, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
            dim=1,
        )
        fusion_image = YCrCb2RGB(fusion_ycrcb)
        ones = torch.ones_like(fusion_image)
        zeros = torch.zeros_like(fusion_image)
        fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
        fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
        fused_image = fusion_image.cpu().detach().numpy()
        fused_image = np.uint8(255.0 * fused_image)

        fused_image = fused_image.transpose((0, 2, 3, 1))
        fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
        )

        fused_image = np.uint8(255.0 * fused_image)
        for k in range(len(name)):
            image = fused_image[k, :, :, :]
            image = Image.fromarray(image)                
            save_path = os.path.join(fused_path, name[k])
            image.save(save_path)
            
        label = label.cpu().numpy().squeeze().flatten()
        prediction = seg1.argmax(
            1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
        conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
        conf_total += conf
        visualize(pth=seg_path,image_name=name, predictions=seg1.argmax(1), weight_name='0')

      precision_per_class, _, iou_per_class = compute_results(conf_total)

      # save_root = './checkpoint/robust_result/'
      res_path = '{0}_PGD{1}_{2}_{3}.txt'.format(strategy, args.attack_iters, int(args.epsilon*255), int(args.alpha*255))
      with open(save_root + res_path, 'w') as f:
        print("\n strategy :" + strategy, file=f)
        print("\n Attack Loss{0}, Attack way{1}, Attack iters{2}, epsilon={3}, alpha={4}".format(args.attack_loss, 'PGD', args.attack_iters, int(args.epsilon*255), int(args.alpha*255)), file=f)
        print(
            "*precision_per_class: \n    %.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f \t %.6f" \
            % (precision_per_class[0], precision_per_class[1], precision_per_class[2], precision_per_class[3], precision_per_class[4],
                precision_per_class[5], precision_per_class[6], precision_per_class[7], precision_per_class[8], np.mean(np.nan_to_num(precision_per_class))), file=f)
        print(
            "* iou per class: \n    %.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" \
            % (
            iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5],
            iou_per_class[6], iou_per_class[7], iou_per_class[8], np.mean(np.nan_to_num(iou_per_class))), file=f)
        print("* average values (np.mean(np.nan_to_num(x))) remove unlabel: \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class[1:])), np.mean(np.nan_to_num(iou_per_class[1:]))), file=f)
      
      
      print("\n strategy :" + strategy)
      print("\n Attack Loss{0}, Attack way{1}, Attack iters{2}, epsilon={3}, alpha={4}".format(args.attack_loss, 'PGD', args.attack_iters, int(args.epsilon*255), int(255*args.alpha)))
      print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
      print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))






      

if __name__ == '__main__':
  device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
  print('| testing on GPU #%d with pytorch' % (args.gpu))

  strategy = 'meta_final'
  Genotype = namedtuple('Genotype', 'normal_1 normal_1_concat normal_2 normal_2_concat normal_3 normal_3_concat')

  fusion_at = Genotype(normal_1=[('Denseblocks_3_1', 0), ('DilConv_3_2', 1)], normal_1_concat=[1, 2],
                     normal_2=[('Denseblocks_3_1', 0), ('Denseblocks_3_1', 1)], normal_2_concat=[1, 2],
                     normal_3=[('ECAattention_3', 0), ('Residualblocks_7_1', 1)], normal_3_concat=[1, 2])

  fusion_model_path = './checkpoint/model_meta30000_fusion_8.pth'
  # model = Network(32,fusion_at,None,None,cfg.exp.backbone,cfg.dataset.num_classes,256,True).cuda()
  criterion_fusion = Fusionloss_grad2().to(device)
  model = Network_MM_Searched(32, fusion_at, None,None,cfg.exp.backbone,num_classes=9)
  model.load_state_dict(torch.load(fusion_model_path), strict=False)
  print("---------model load done!-----------------")
  print("-----start testing model-----")
  val_segformer_robust(model, strategy=strategy)
