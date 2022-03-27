import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)

    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        N = boxes.size()[0]
        #boxes_temp = torch.zeros((N, 4))
        boxes_temp = torch.clone(boxes)              # MAYBE THIS torch.clone IS A PROBLEM TOO #

        boxes_temp[:,0] = boxes[:,0] / self.S - 0.5*boxes[:,2]
        boxes_temp[:,1] = boxes[:,1] / self.S - 0.5*boxes[:,3]
        boxes_temp[:,2] = boxes[:,0] / self.S + 0.5*boxes[:,2]
        boxes_temp[:,3] = boxes[:,1] / self.S + 0.5*boxes[:,3]

        return boxes_temp

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 4) ...]
        box_target : (tensor)  size (-1, 5)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        box_target = self.xywh2xyxy(box_target)

        box_pred_01 = pred_box_list[0]
        box_pred_02 = pred_box_list[1]

        box_pred_1 = self.xywh2xyxy(pred_box_list[0]) 
        box_pred_2 = self.xywh2xyxy(pred_box_list[1])

        iou_1 = compute_iou(box_target, box_pred_1[:,0:4])
        iou_1 = torch.diag(iou_1,0)

        iou_2 = compute_iou(box_target, box_pred_2[:,0:4])
        iou_2 = torch.diag(iou_2,0)

        N = box_target.size()[0]
        #best_ious = torch.zeros((N,1))
        #best_boxes = torch.zeros((N,5))    # YOU SHOULDNT CREATE A NEW TENSOR FOR TENSORS THAT CARRY GRADIENT, THAT MESSES UP THE GRADIENT #
        best_ious = []
        best_boxes = []




        for i in range(N):
            if iou_1[i] > iou_2[i]:
                best_ious.append(iou_1[i])
                best_boxes.append(box_pred_01[i])
            else:
                best_ious.append(iou_2[i])
                best_boxes.append(box_pred_02[i]) 

        best_ious = torch.stack(best_ious).detach()#.cuda()
        best_boxes = torch.stack(best_boxes)#.cuda()

        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here

        return F.mse_loss(classes_pred[has_object_map], classes_target[has_object_map], reduction='sum') 

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes] ### IS THIS THE SAME AS box_pred_response LIST? ###
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        no_object_map = ~has_object_map   

        #print('has_object_map shape = ', has_object_map.size())
        S = has_object_map.size()[1]

        no_object_pred_boxes_list = []
        temp1 = (pred_boxes_list[0][no_object_map]).view(-1,5)
        temp2 = (pred_boxes_list[1][no_object_map]).view(-1,5)
        no_object_pred_boxes_list.append(temp1)
        no_object_pred_boxes_list.append(temp2)

        #print('no_object_map shape = ', no_object_map.size())

        N = temp1.size()[0]
        M = temp1.size()[1]

        C = temp2.size()[0]
        D = temp2.size()[1]

        # print('temp1 size = ',temp1.size())
        # print('temp2 size = ',temp2.size())

        # zeros1 = torch.zeros((N,M)).cuda()
        # zeros2 = torch.zeros((C,D)).cuda()

        # print('zeros1 size = ',zeros1.size())
        # print('zeros2 size = ',zeros2.size())


        # loss1 = F.mse_loss(temp1, zeros1, reduction='sum').cuda()
        # loss2 = F.mse_loss(temp2, zeros2, reduction='sum').cuda()

        loss1 = torch.sum(torch.square(temp1[:,4]))
        loss2 = torch.sum(torch.square(temp2[:,4]))

        loss = loss1+loss2

        # loss = 0

        # for i in range(N):
        #     #loss += (no_object_pred_boxes_list[0][i, :, :, 4])**2 
        #     loss += (no_object_pred_boxes_list[0][i,4])**2    
        # for c in range(C):
        #     #loss += (no_object_pred_boxes_list[1][c, :, :, 4])**2
        #     loss += (no_object_pred_boxes_list[1][c,4])**2

        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        loss = F.mse_loss(box_pred_conf, box_target_conf.detach(), reduction='sum')
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        pred_xy = box_pred_response[:,:2]#.cuda()
        pred_wh = torch.sqrt(box_pred_response[:,2:4])#.cuda()
        target_xy = box_target_response[:,:2]
        target_wh = torch.sqrt(box_target_response[:,2:4])

        loss_xy = F.mse_loss(pred_xy, target_xy, reduction='sum')
        loss_wh = F.mse_loss(pred_wh, target_wh, reduction='sum')
        reg_loss = loss_xy + loss_wh

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        pred_boxes_list = []
        pred_boxes_list.append(pred_tensor[:,:,:,:5])
        pred_boxes_list.append(pred_tensor[:,:,:,5:10])

        # -- pred_cls (containing all classification prediction)
        pred_cls = pred_tensor[:,:,:,10:]

        # compute classification loss
        class_loss = self.get_class_prediction_loss(pred_cls,target_cls,has_object_map)

        # compute no-object loss
        noob_loss = self.get_no_object_loss(pred_boxes_list,has_object_map)

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        has_object_target_boxes = (target_boxes[has_object_map]).view(-1,4)
        
        has_object_pred_boxes_list = []
        temp1 = (pred_boxes_list[0][has_object_map]).view(-1,5)      
        temp2 = (pred_boxes_list[1][has_object_map]).view(-1,5)
        has_object_pred_boxes_list.append(temp1)
        has_object_pred_boxes_list.append(temp2)

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(has_object_pred_boxes_list, has_object_target_boxes)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(best_boxes, has_object_target_boxes)

        # compute contain_object_loss
        box_pred_conf = best_boxes[:,4].view(-1,1) 

        #n = has_object_target_boxes.size()[0]
        box_target_conf = best_ious.view(-1,1)   

        cont_obj_loss = self.get_contain_conf_loss(box_pred_conf,box_target_conf)#.cuda()
        

        # compute final loss
        final_loss = self.l_coord * reg_loss + cont_obj_loss + self.l_noobj * noob_loss + class_loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss=final_loss,
            reg_loss=reg_loss,
            containing_obj_loss=cont_obj_loss,
            no_obj_loss=noob_loss,
            cls_loss=class_loss,
        )
        return loss_dict

def test():
    yl = YoloLoss(2, 2, 5, 0.5)
    N = 2
    torch.random.manual_seed(0)
    pred_tensor = torch.rand((N, 2, 2, 30), requires_grad=True)
    target_box = 0.5 * torch.ones((N, 2, 2, 4))
    target_cls = torch.zeros((N, 2, 2, 20))
    target_cls[:, :, :, 10] = 1
    obj_map = torch.BoolTensor(size=(N, 2, 2))
    obj_map[:, :, :] = False
    obj_map[:, 0, 0] = True
    # print(pred_tensor)
    loss = yl(pred_tensor, target_box, target_cls, obj_map)
    loss["total_loss"].backward()
    print(pred_tensor.grad)

# test()