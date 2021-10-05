import numpy as np
import torch
from torchvision.ops import nms

def decodebox(regression, anchors, input_shape):
    dtype   = regression.dtype
    anchors = anchors.to(dtype)
    #--------------------------------------#
    #   计算先验框的中心
    #--------------------------------------#
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2

    #--------------------------------------#
    #   计算先验框的宽高
    #--------------------------------------#
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    #--------------------------------------#
    #   计算调整后先验框的宽高
    #   即计算预测框的宽高
    #--------------------------------------#
    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    #--------------------------------------#
    #   计算调整后先验框的中心
    #   即计算预测框的中心
    #--------------------------------------#
    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    #--------------------------------------#
    #   计算预测框的左上角右下角
    #--------------------------------------#
    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    #--------------------------------------#
    #   将预测框的结果进行堆叠
    #--------------------------------------#
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # grid_x = x_centers_a[0,-4*4*9:]
    # grid_y = y_centers_a[0,-4*4*9:]
    # plt.ylim(-600,1200)
    # plt.xlim(-600,1200)
    # plt.gca().invert_yaxis()
    # plt.scatter(grid_x.cpu(),grid_y.cpu())

    # anchor_left = anchors[0,-4*4*9:,1]
    # anchor_top = anchors[0,-4*4*9:,0]
    # anchor_w = wa[0,-4*4*9:]
    # anchor_h = ha[0,-4*4*9:]

    # for i in range(9,18):
    #     rect1 = plt.Rectangle([anchor_left[i],anchor_top[i]],anchor_w[i],anchor_h[i],color="r",fill=False)
    #     ax.add_patch(rect1)

    # ax = fig.add_subplot(122)
    
    # grid_x = x_centers_a[0,-4*4*9:]
    # grid_y = y_centers_a[0,-4*4*9:]
    # plt.scatter(grid_x.cpu(),grid_y.cpu())
    # plt.ylim(-600,1200)
    # plt.xlim(-600,1200)
    # plt.gca().invert_yaxis()
    
    # y_centers = y_centers[0,-4*4*9:]
    # x_centers = x_centers[0,-4*4*9:]

    # pre_left = xmin[0,-4*4*9:]
    # pre_top = ymin[0,-4*4*9:]
    
    # pre_w = xmax[0,-4*4*9:]-xmin[0,-4*4*9:]
    # pre_h = ymax[0,-4*4*9:]-ymin[0,-4*4*9:]

    # for i in range(9,18):
    #     plt.scatter(x_centers[i].cpu(),y_centers[i].cpu(),c='r')
    #     rect1 = plt.Rectangle([pre_left[i],pre_top[i]],pre_w[i],pre_h[i],color="r",fill=False)
    #     ax.add_patch(rect1)

    # plt.show()
    boxes[:, :, [0, 2]] = boxes[:, :, [0, 2]] / input_shape[1]
    boxes[:, :, [1, 3]] = boxes[:, :, [1, 3]] / input_shape[0]

    boxes = torch.clamp(boxes, min = 0, max = 1)
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
                 
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min = 1e-6)

    return iou

def retinanet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def non_max_suppression(prediction, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    
    #----------------------------------------------------------#
    #   预测只用一张图片，只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 4:], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()

        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 6]
        #   6的内容为：x1, y1, x2, y2, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)

        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4],
                nms_thres
            )
            max_detections = detections_class[keep]
            
            # #------------------------------------------#
            # #   按照存在物体的置信度排序
            # #------------------------------------------#
            # _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # #------------------------------------------#
            # #   进行非极大抑制
            # #------------------------------------------#
            # max_detections = []
            # while detections_class.size(0):
            #     #---------------------------------------------------#
            #     #   取出这一类置信度最高的，一步一步往下判断。
            #     #   判断重合程度是否大于nms_thres，如果是则去除掉
            #     #---------------------------------------------------#
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # #------------------------------------------#
            # #   堆叠
            # #------------------------------------------#
            # max_detections = torch.cat(max_detections).data
            
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = retinanet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output
