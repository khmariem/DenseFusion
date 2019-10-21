from __future__ import division
import _init_paths
import argparse
import os
import copy
import random
import numpy as np
import cv2 as cv
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import matplotlib.pyplot as plt
from tools.select_roi import selectme as sm
from tools.color_quant import quant

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 340.61
cam_cy = 241.71
cam_fx = 173.36
cam_fy = 175.62
cam_scale = 1.0
camera_intrinsics[0,0] = cam_fx
camera_intrinsics[1,1] = cam_fy
camera_intrinsics [2,2] = 1.0
camera_intrinsics[0,2] = cam_cx
camera_intrinsics[1,2] = cam_cy
num_obj = 4
img_width = 480
img_length = 640
num_points = 500
num_points_mesh = 500
iteration = 2
bs = 1
d={'sandwich':2,'onigiri':3,'bento':4,'cup':1}

wrs_toolbox_dir = 
result_wo_refine_dir = 'experiments/eval_result/wrs/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/wrs/Densefusion_iterative_result'
model='trained_models/wrs/pose_model_TO_BE_DEFINED.pth'
#refine_model='trained_models/wrs/pose_refine_model_TO_BE_DEFINED.pth'

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.load_state_dict(torch.load(model))
estimator.eval()

#refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
#refiner.load_state_dict(torch.load(refine_model))
#refiner.eval()

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

while 1:
    #xtion
        #img
        img = Image.fromarray(img_array)
        img = Image.open()
        #depth
        depth = Image.fromarray(img_array)
        depth = np.array(Image.open())
    #yolo
        #labels
        label = quant()
        #lst obj
        lst=['sandwich']
        ##bbxs
        pose = sm()

    my_result_wo_refine = []
    my_result = []
    for idx in range(len(lst)):
        itemid = d[lst[idx]]
        try:
            #rmin, rmax, cmin, cmax = #yolo
            rmin = pose[1]
            rmax = rmin + pose[3]
            cmin = pose[0]
            cmax = cmin + pose[2]

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine.append(my_pred.tolist())

            # for ite in range(0, iteration):
            #     T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            #     my_mat = quaternion_matrix(my_r)
            #     R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            #     my_mat[0:3, 3] = my_t
                
            #     new_cloud = torch.bmm((cloud - T), R).contiguous()
            #     pred_r, pred_t = refiner(new_cloud, emb, index)
            #     pred_r = pred_r.view(1, 1, -1)
            #     pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            #     my_r_2 = pred_r.view(-1).cpu().data.numpy()
            #     my_t_2 = pred_t.view(-1).cpu().data.numpy()
            #     my_mat_2 = quaternion_matrix(my_r_2)

            #     my_mat_2[0:3, 3] = my_t_2

            #     my_mat_final = np.dot(my_mat, my_mat_2)
            #     my_r_final = copy.deepcopy(my_mat_final)
            #     my_r_final[0:3, 3] = 0
            #     my_r_final = quaternion_from_matrix(my_r_final, True)
            #     my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            #     my_pred = np.append(my_r_final, my_t_final)
            #     my_r = my_r_final
            #     my_t = my_t_final

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
            #print(my_t)
            rot = quaternion_matrix(my_r)
            trans = my_t
            rot = rot[:3,:3]
            #print(trans)
            #print(np.dot(rot,rot.transpose()))
            h=np.zeros((3,4))
            h[:,:3] = rot
            h[:,3] = trans
            hh[:,:-1]=cld[itemid]
            #im = plt.imread('tools/'+testlist[now-1]+'-color.png')
            implot = plt.imshow(im)
            proj=h
            proj = np.dot(camera_intrinsics, h)
            cloud_proj = np.dot(proj,hh.transpose())
            #print(cloud_proj.transpose())
            #plt.scatter(cloud_proj[0,:],cloud_proj[1,:])
            #plt.scatter(cld[itemid][:,0],cld[itemid][:,1], cld[itemid][:,2])
            cloud_proj[0,:] = cloud_proj[0,:]/cloud_proj[2,:]
            cloud_proj[1,:] = cloud_proj[1,:]/cloud_proj[2,:]
            #print(cloud_proj)
            #cloud_proj = cloud_proj[0:3,:].transpose()
            cloud_proj = cloud_proj.transpose()
            plt.scatter(cloud_proj[:,0],cloud_proj[:,1])
            print(cloud_proj)
            #plt.scatter(cloud_proj[:,0],cloud_proj[:,1])
            plt.show()
            my_result.append(my_pred.tolist())
        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])


