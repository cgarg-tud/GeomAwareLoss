"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import os
import math
import random
import numpy as np
import torch
import cv2
import itertools

import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib import cm

import matplotlib


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

discretized_jet = cmap_discretize(matplotlib.cm.jet, 10)

turbo_colormap_data = np.array(
                       [[0.18995,0.07176,0.23217],
                       [0.19483,0.08339,0.26149],
                       [0.19956,0.09498,0.29024],
                       [0.20415,0.10652,0.31844],
                       [0.20860,0.11802,0.34607],
                       [0.21291,0.12947,0.37314],
                       [0.21708,0.14087,0.39964],
                       [0.22111,0.15223,0.42558],
                       [0.22500,0.16354,0.45096],
                       [0.22875,0.17481,0.47578],
                       [0.23236,0.18603,0.50004],
                       [0.23582,0.19720,0.52373],
                       [0.23915,0.20833,0.54686],
                       [0.24234,0.21941,0.56942],
                       [0.24539,0.23044,0.59142],
                       [0.24830,0.24143,0.61286],
                       [0.25107,0.25237,0.63374],
                       [0.25369,0.26327,0.65406],
                       [0.25618,0.27412,0.67381],
                       [0.25853,0.28492,0.69300],
                       [0.26074,0.29568,0.71162],
                       [0.26280,0.30639,0.72968],
                       [0.26473,0.31706,0.74718],
                       [0.26652,0.32768,0.76412],
                       [0.26816,0.33825,0.78050],
                       [0.26967,0.34878,0.79631],
                       [0.27103,0.35926,0.81156],
                       [0.27226,0.36970,0.82624],
                       [0.27334,0.38008,0.84037],
                       [0.27429,0.39043,0.85393],
                       [0.27509,0.40072,0.86692],
                       [0.27576,0.41097,0.87936],
                       [0.27628,0.42118,0.89123],
                       [0.27667,0.43134,0.90254],
                       [0.27691,0.44145,0.91328],
                       [0.27701,0.45152,0.92347],
                       [0.27698,0.46153,0.93309],
                       [0.27680,0.47151,0.94214],
                       [0.27648,0.48144,0.95064],
                       [0.27603,0.49132,0.95857],
                       [0.27543,0.50115,0.96594],
                       [0.27469,0.51094,0.97275],
                       [0.27381,0.52069,0.97899],
                       [0.27273,0.53040,0.98461],
                       [0.27106,0.54015,0.98930],
                       [0.26878,0.54995,0.99303],
                       [0.26592,0.55979,0.99583],
                       [0.26252,0.56967,0.99773],
                       [0.25862,0.57958,0.99876],
                       [0.25425,0.58950,0.99896],
                       [0.24946,0.59943,0.99835],
                       [0.24427,0.60937,0.99697],
                       [0.23874,0.61931,0.99485],
                       [0.23288,0.62923,0.99202],
                       [0.22676,0.63913,0.98851],
                       [0.22039,0.64901,0.98436],
                       [0.21382,0.65886,0.97959],
                       [0.20708,0.66866,0.97423],
                       [0.20021,0.67842,0.96833],
                       [0.19326,0.68812,0.96190],
                       [0.18625,0.69775,0.95498],
                       [0.17923,0.70732,0.94761],
                       [0.17223,0.71680,0.93981],
                       [0.16529,0.72620,0.93161],
                       [0.15844,0.73551,0.92305],
                       [0.15173,0.74472,0.91416],
                       [0.14519,0.75381,0.90496],
                       [0.13886,0.76279,0.89550],
                       [0.13278,0.77165,0.88580],
                       [0.12698,0.78037,0.87590],
                       [0.12151,0.78896,0.86581],
                       [0.11639,0.79740,0.85559],
                       [0.11167,0.80569,0.84525],
                       [0.10738,0.81381,0.83484],
                       [0.10357,0.82177,0.82437],
                       [0.10026,0.82955,0.81389],
                       [0.09750,0.83714,0.80342],
                       [0.09532,0.84455,0.79299],
                       [0.09377,0.85175,0.78264],
                       [0.09287,0.85875,0.77240],
                       [0.09267,0.86554,0.76230],
                       [0.09320,0.87211,0.75237],
                       [0.09451,0.87844,0.74265],
                       [0.09662,0.88454,0.73316],
                       [0.09958,0.89040,0.72393],
                       [0.10342,0.89600,0.71500],
                       [0.10815,0.90142,0.70599],
                       [0.11374,0.90673,0.69651],
                       [0.12014,0.91193,0.68660],
                       [0.12733,0.91701,0.67627],
                       [0.13526,0.92197,0.66556],
                       [0.14391,0.92680,0.65448],
                       [0.15323,0.93151,0.64308],
                       [0.16319,0.93609,0.63137],
                       [0.17377,0.94053,0.61938],
                       [0.18491,0.94484,0.60713],
                       [0.19659,0.94901,0.59466],
                       [0.20877,0.95304,0.58199],
                       [0.22142,0.95692,0.56914],
                       [0.23449,0.96065,0.55614],
                       [0.24797,0.96423,0.54303],
                       [0.26180,0.96765,0.52981],
                       [0.27597,0.97092,0.51653],
                       [0.29042,0.97403,0.50321],
                       [0.30513,0.97697,0.48987],
                       [0.32006,0.97974,0.47654],
                       [0.33517,0.98234,0.46325],
                       [0.35043,0.98477,0.45002],
                       [0.36581,0.98702,0.43688],
                       [0.38127,0.98909,0.42386],
                       [0.39678,0.99098,0.41098],
                       [0.41229,0.99268,0.39826],
                       [0.42778,0.99419,0.38575],
                       [0.44321,0.99551,0.37345],
                       [0.45854,0.99663,0.36140],
                       [0.47375,0.99755,0.34963],
                       [0.48879,0.99828,0.33816],
                       [0.50362,0.99879,0.32701],
                       [0.51822,0.99910,0.31622],
                       [0.53255,0.99919,0.30581],
                       [0.54658,0.99907,0.29581],
                       [0.56026,0.99873,0.28623],
                       [0.57357,0.99817,0.27712],
                       [0.58646,0.99739,0.26849],
                       [0.59891,0.99638,0.26038],
                       [0.61088,0.99514,0.25280],
                       [0.62233,0.99366,0.24579],
                       [0.63323,0.99195,0.23937],
                       [0.64362,0.98999,0.23356],
                       [0.65394,0.98775,0.22835],
                       [0.66428,0.98524,0.22370],
                       [0.67462,0.98246,0.21960],
                       [0.68494,0.97941,0.21602],
                       [0.69525,0.97610,0.21294],
                       [0.70553,0.97255,0.21032],
                       [0.71577,0.96875,0.20815],
                       [0.72596,0.96470,0.20640],
                       [0.73610,0.96043,0.20504],
                       [0.74617,0.95593,0.20406],
                       [0.75617,0.95121,0.20343],
                       [0.76608,0.94627,0.20311],
                       [0.77591,0.94113,0.20310],
                       [0.78563,0.93579,0.20336],
                       [0.79524,0.93025,0.20386],
                       [0.80473,0.92452,0.20459],
                       [0.81410,0.91861,0.20552],
                       [0.82333,0.91253,0.20663],
                       [0.83241,0.90627,0.20788],
                       [0.84133,0.89986,0.20926],
                       [0.85010,0.89328,0.21074],
                       [0.85868,0.88655,0.21230],
                       [0.86709,0.87968,0.21391],
                       [0.87530,0.87267,0.21555],
                       [0.88331,0.86553,0.21719],
                       [0.89112,0.85826,0.21880],
                       [0.89870,0.85087,0.22038],
                       [0.90605,0.84337,0.22188],
                       [0.91317,0.83576,0.22328],
                       [0.92004,0.82806,0.22456],
                       [0.92666,0.82025,0.22570],
                       [0.93301,0.81236,0.22667],
                       [0.93909,0.80439,0.22744],
                       [0.94489,0.79634,0.22800],
                       [0.95039,0.78823,0.22831],
                       [0.95560,0.78005,0.22836],
                       [0.96049,0.77181,0.22811],
                       [0.96507,0.76352,0.22754],
                       [0.96931,0.75519,0.22663],
                       [0.97323,0.74682,0.22536],
                       [0.97679,0.73842,0.22369],
                       [0.98000,0.73000,0.22161],
                       [0.98289,0.72140,0.21918],
                       [0.98549,0.71250,0.21650],
                       [0.98781,0.70330,0.21358],
                       [0.98986,0.69382,0.21043],
                       [0.99163,0.68408,0.20706],
                       [0.99314,0.67408,0.20348],
                       [0.99438,0.66386,0.19971],
                       [0.99535,0.65341,0.19577],
                       [0.99607,0.64277,0.19165],
                       [0.99654,0.63193,0.18738],
                       [0.99675,0.62093,0.18297],
                       [0.99672,0.60977,0.17842],
                       [0.99644,0.59846,0.17376],
                       [0.99593,0.58703,0.16899],
                       [0.99517,0.57549,0.16412],
                       [0.99419,0.56386,0.15918],
                       [0.99297,0.55214,0.15417],
                       [0.99153,0.54036,0.14910],
                       [0.98987,0.52854,0.14398],
                       [0.98799,0.51667,0.13883],
                       [0.98590,0.50479,0.13367],
                       [0.98360,0.49291,0.12849],
                       [0.98108,0.48104,0.12332],
                       [0.97837,0.46920,0.11817],
                       [0.97545,0.45740,0.11305],
                       [0.97234,0.44565,0.10797],
                       [0.96904,0.43399,0.10294],
                       [0.96555,0.42241,0.09798],
                       [0.96187,0.41093,0.09310],
                       [0.95801,0.39958,0.08831],
                       [0.95398,0.38836,0.08362],
                       [0.94977,0.37729,0.07905],
                       [0.94538,0.36638,0.07461],
                       [0.94084,0.35566,0.07031],
                       [0.93612,0.34513,0.06616],
                       [0.93125,0.33482,0.06218],
                       [0.92623,0.32473,0.05837],
                       [0.92105,0.31489,0.05475],
                       [0.91572,0.30530,0.05134],
                       [0.91024,0.29599,0.04814],
                       [0.90463,0.28696,0.04516],
                       [0.89888,0.27824,0.04243],
                       [0.89298,0.26981,0.03993],
                       [0.88691,0.26152,0.03753],
                       [0.88066,0.25334,0.03521],
                       [0.87422,0.24526,0.03297],
                       [0.86760,0.23730,0.03082],
                       [0.86079,0.22945,0.02875],
                       [0.85380,0.22170,0.02677],
                       [0.84662,0.21407,0.02487],
                       [0.83926,0.20654,0.02305],
                       [0.83172,0.19912,0.02131],
                       [0.82399,0.19182,0.01966],
                       [0.81608,0.18462,0.01809],
                       [0.80799,0.17753,0.01660],
                       [0.79971,0.17055,0.01520],
                       [0.79125,0.16368,0.01387],
                       [0.78260,0.15693,0.01264],
                       [0.77377,0.15028,0.01148],
                       [0.76476,0.14374,0.01041],
                       [0.75556,0.13731,0.00942],
                       [0.74617,0.13098,0.00851],
                       [0.73661,0.12477,0.00769],
                       [0.72686,0.11867,0.00695],
                       [0.71692,0.11268,0.00629],
                       [0.70680,0.10680,0.00571],
                       [0.69650,0.10102,0.00522],
                       [0.68602,0.09536,0.00481],
                       [0.67535,0.08980,0.00449],
                       [0.66449,0.08436,0.00424],
                       [0.65345,0.07902,0.00408],
                       [0.64223,0.07380,0.00401],
                       [0.63082,0.06868,0.00401],
                       [0.61923,0.06367,0.00410],
                       [0.60746,0.05878,0.00427],
                       [0.59550,0.05399,0.00453],
                       [0.58336,0.04931,0.00486],
                       [0.57103,0.04474,0.00529],
                       [0.55852,0.04028,0.00579],
                       [0.54583,0.03593,0.00638],
                       [0.53295,0.03169,0.00705],
                       [0.51989,0.02756,0.00780],
                       [0.50664,0.02354,0.00863],
                       [0.49321,0.01963,0.00955],
                       [0.47960,0.01583,0.01055]])
cm.register_cmap('turbo', cmap=ListedColormap(turbo_colormap_data))
############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        ## Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            ## x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            ## No mask for this instance. Might happen due to
            ## resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    ## Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    ## Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    ## Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    ## Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


############################################################
#  Dataset
############################################################
def resize_image(image, min_dim=None, max_dim=None, padding=False, interp='bilinear'):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    ## Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    ## Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    ## Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    ## Resize image and mask
    if scale != 1:
        image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
    ## Need padding?
    if padding:
        ## Get new height and width
        h, w = image.shape[:2]
        top_pad = (min_dim - h) // 2
        bottom_pad = min_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = cv2.resize(m.astype(np.uint8) * 255, mini_shape)
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask

def minimize_depth(bbox, depth, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (len(bbox),), dtype=np.float32)
    for i in range(len(bbox)):
        y1, x1, y2, x2 = bbox[i][:4]
        m = cv2.resize(depth[y1:y2, x1:x2], mini_shape, interpolation=cv2.INTER_NEAREST)
        mini_mask[:, :, i] = m
    return mini_mask


## TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = cv2.resize(mask.astype(np.float32), (x2 - x1, y2 - y1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    ## Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    ## Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    ## Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    ## Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    ## Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    ## Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    ## Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    ## Anchors
    ## [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    anchors = np.concatenate(anchors, axis=0)
    return anchors


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


## Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)



## Visualization
class ColorPalette:
    def __init__(self, numColors):
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [128, 0, 255],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [255, 230, 180],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ], dtype=np.uint8)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3), dtype=np.uint8)], axis=0)
            pass

        return

    def getColorMap(self, returnTuples=False):
        if returnTuples:
            return [tuple(color) for color in self.colorMap.tolist()]
        else:
            return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
        pass
    

def writePointCloud(filename, point_cloud):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(point_cloud))
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
end_header
"""
        f.write(header)
        for point in point_cloud:
            for valueIndex, value in enumerate(point):
                if valueIndex < 3:
                    f.write(str(value) + ' ')
                else:
                    f.write(str(int(value)) + ' ')
                    pass
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


## The function to compute plane depths from plane parameters
def calcPlaneDepths(planes, width, height, camera, max_depth=10):
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
    
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    return planeDepths

## The function to compute plane XYZ from plane parameters
def calcPlaneXYZ(planes, width, height, camera, max_depth=10):
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
    
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    return np.expand_dims(planeDepths, -1) * np.expand_dims(ranges, 2)


## Compute surface normal from depth
def calcNormal(depth, camera):

    height = depth.shape[0]
    width = depth.shape[1]

    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    X = depth * urange
    Y = depth
    Z = -depth * vrange

    points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)


    if True:
        if width > 300:
            grids = np.array([-9, -6, -3, -1, 0, 1, 3, 6, 9], dtype=np.int32)
        else:
            grids = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.int32)
            pass

        normals = []
        for index in range(width * height):
            us = index % width + grids
            us = us[np.logical_and(us >= 0, us < width)]
            vs = index // width + grids
            vs = vs[np.logical_and(vs >= 0, vs < height)]
            indices = (np.expand_dims(vs, -1) * width + np.expand_dims(us, 0)).reshape(-1)
            planePoints = points[indices]
            planePoints = planePoints[np.linalg.norm(planePoints, axis=-1) > 1e-4]

            planePoints = planePoints[np.abs(planePoints[:, 1] - points[index][1]) < 0.05]

            try:
                plane = fitPlane(planePoints)
                normal = plane / np.maximum(np.linalg.norm(plane), 1e-4)
                if np.dot(normal, points[index]) > 0:
                    normal *= -1
                    pass
                normals.append(normal)
            except:
                if len(normals) > 0:
                    normals.append(normals[-1])
                else:
                    normals.append([0, -1, 0])
                    pass                    
            continue
        normal = np.array(normals).reshape((height, width, 3))
    else:
        from scipy.linalg import eigh
        from sklearn.neighbors import NearestNeighbors

        number_neighbors = 10

        normals = np.empty_like(points)

        neigh = NearestNeighbors(number_neighbors)
        neigh.fit(points)

        distances, neighbors = neigh.kneighbors(points)
        
        for i in range(len(points)):
            XYZ = points[neighbors[i][1:number_neighbors]]

            average = np.sum(XYZ, axis=0) / XYZ.shape[0]
            b = np.transpose(XYZ - average)
            cov = np.cov(b)
            e_val, e_vect = eigh(cov, overwrite_a=True, overwrite_b=True)
            norm =  e_vect[:,0]
            normals[i] = norm
            continue
        normal = normals.reshape((height, width, 3))
        pass

    return normal

def calcNormal_spx(depth, camera):

#    type(camera)
#    print(camera)
#    
#    type(depth)
#    print(depth)
    
    height = depth.shape[0]
    width = depth.shape[1]

    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    X = depth * urange
    Y = depth
    Z = -depth * vrange

    points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)


    if True:
        if width > 300:
            grids = np.array([-9, -6, -3, -1, 0, 1, 3, 6, 9], dtype=np.int32)
        else:
            grids = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.int32)
            pass

        normals = []
        for index in range(width * height):
            us = index % width + grids
            us = us[np.logical_and(us >= 0, us < width)]
            vs = index // width + grids
            vs = vs[np.logical_and(vs >= 0, vs < height)]
            indices = (np.expand_dims(vs, -1) * width + np.expand_dims(us, 0)).reshape(-1)
            planePoints = points[indices]
            planePoints = planePoints[np.linalg.norm(planePoints, axis=-1) > 1e-4]

            planePoints = planePoints[np.abs(planePoints[:, 1] - points[index][1]) < 0.05]

            try:
                plane = fitPlane(planePoints)
                normal = plane / np.maximum(np.linalg.norm(plane), 1e-4)
                if np.dot(normal, points[index]) > 0:
                    normal *= -1
                    pass
                normals.append(normal)
            except:
                if len(normals) > 0:
                    normals.append(normals[-1])
                else:
                    normals.append([0, -1, 0])
                    pass                    
            continue
        normal = np.array(normals).reshape((height, width, 3))
    else:
        from scipy.linalg import eigh
        from sklearn.neighbors import NearestNeighbors

        number_neighbors = 10

        normals = np.empty_like(points)

        neigh = NearestNeighbors(number_neighbors)
        neigh.fit(points)

        distances, neighbors = neigh.kneighbors(points)
        
        for i in range(len(points)):
            XYZ = points[neighbors[i][1:number_neighbors]]

            average = np.sum(XYZ, axis=0) / XYZ.shape[0]
            b = np.transpose(XYZ - average)
            cov = np.cov(b)
            e_val, e_vect = eigh(cov, overwrite_a=True, overwrite_b=True)
            norm =  e_vect[:,0]
            normals[i] = norm
            continue
        normal = normals.reshape((height, width, 3))
        pass

    return normal


## Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))

## Draw depth image
def drawDepthImage(depth, maxDepth=5):
    depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
    depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
    return depthImage

def drawerrorImage(error):
    
    plt.figure()
    plt.imshow(error , cmap = plt.get_cmap('jet') )
#    plt.imshow(error , cmap = discretized_jet).set_clim(0.0,1.8)
    plt.colorbar()
   
#    plt.hist(error.ravel(),256,[0,256])
    plt.show()
    
#    plt.savefig('error.png')

## Draw normal image
def drawNormalImage(normal):
    normalImage = np.clip((normal + 1) / 2 * 255, 0, 255).astype(np.uint8)
    normalImage = normalImage[:, :, ::-1]
    return normalImage

## Draw depth image
def drawMaskImage(mask):
    return (np.clip(mask * 255, 0, 255)).astype(np.uint8)

## Draw flow image
def drawFlowImage(flow):
    max_flow = 0.1

    V = np.full((flow.shape[0], flow.shape[1]), fill_value=0.8)
    S = np.minimum(np.linalg.norm(flow, axis=-1) / max_flow, 1)
    H = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    F = H - H // (np.pi / 3)
    P = V * (1 - S)
    Q = V * (1 - S * F)
    T = V * (1 - S * (1 - F))
    images = np.stack([np.stack([P, T, V], axis=-1), np.stack([P, V, Q], axis=-1), np.stack([T, V, P], axis=-1), np.stack([V, Q, P], axis=-1), np.stack([V, P, T], axis=-1), np.stack([Q, P, V], axis=-1)], axis=-1)
    labels = ((H + 2 * np.pi) // (np.pi / 3)).astype(np.int32) % 6
    image = np.sum(images * np.expand_dims(np.expand_dims(labels, axis=-1) == np.arange(6, dtype=np.int32), 2), axis=-1)
    image = (image * 255).astype(np.uint8)
    return image

## Fit a 3D plane from points
def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]
    return

## Run PlaneNet inference
def predictPlaneNet(image):
    sys.path.append('../../')
    from PlaneNet.planenet_inference import PlaneNetDetector
    detector = PlaneNetDetector()
    pred_dict = detector.detect(image)
    return pred_dict


## Clean segmentation
def cleanSegmentation(image, planes, plane_info, segmentation, depth, camera, planeAreaThreshold=200, planeWidthThreshold=10, depthDiffThreshold=0.1, validAreaThreshold=0.5, brightThreshold=20, confident_labels={}, return_plane_depths=False):

    planeDepths = calcPlaneDepths(planes, segmentation.shape[1], segmentation.shape[0], camera).transpose((2, 0, 1))
    
    newSegmentation = np.full(segmentation.shape, fill_value=-1)
    validMask = np.logical_and(np.linalg.norm(image, axis=-1) > brightThreshold, depth > 1e-4)
    depthDiffMask = np.logical_or(np.abs(planeDepths - depth) < depthDiffThreshold, depth < 1e-4)

    for segmentIndex in np.unique(segmentation):
        if segmentIndex < 0:
            continue
        segmentMask = segmentation == segmentIndex

        try:
            plane_info[segmentIndex][0][1]
        except:
            print('invalid plane info')
            print(plane_info)
            print(len(plane_info), len(planes), segmentation.min(), segmentation.max())
            print(segmentIndex)
            print(plane_info[segmentIndex])
            exit(1)
        if plane_info[segmentIndex][0][1] in confident_labels:
            if segmentMask.sum() > planeAreaThreshold:
                newSegmentation[segmentMask] = segmentIndex
                pass
            continue
        oriArea = segmentMask.sum()
        segmentMask = np.logical_and(segmentMask, depthDiffMask[segmentIndex])
        newArea = np.logical_and(segmentMask, validMask).sum()
        if newArea < oriArea * validAreaThreshold:
            continue
        segmentMask = segmentMask.astype(np.uint8)
        segmentMask = cv2.dilate(segmentMask, np.ones((3, 3)))
        numLabels, components = cv2.connectedComponents(segmentMask)
        for label in range(1, numLabels):
            mask = components == label
            ys, xs = mask.nonzero()
            area = float(len(xs))
            if area < planeAreaThreshold:
                continue
            size_y = ys.max() - ys.min() + 1
            size_x = xs.max() - xs.min() + 1
            length = np.linalg.norm([size_x, size_y])
            if area / length < planeWidthThreshold:
                continue
            newSegmentation[mask] = segmentIndex
            continue
        continue
    if return_plane_depths:
        return newSegmentation, planeDepths
    return newSegmentation



def getLayout(planes, depth, plane_depths, plane_info, segmentation, camera, layout_labels={}, return_segmentation=True, get_boundary=True):
    parallelThreshold = np.cos(np.deg2rad(30))
    
    layoutSegmentation = np.full(segmentation.shape, fill_value=-1)

    layoutPlanePoints = []    
    layoutPlaneMasks = []
    layoutPlaneIndices = []
    for planeIndex, info in enumerate(plane_info):
        if info[0][1] not in layout_labels:
            continue
        mask = segmentation == planeIndex
        ys, xs = mask.nonzero()
        if len(xs) < depth.shape[0] * depth.shape[1] * 0.02:
            continue
        layoutSegmentation[mask] = len(layoutPlaneIndices)        
        layoutPlanePoints.append(np.stack([xs, ys], axis=-1))
        layoutPlaneMasks.append(mask)
        layoutPlaneIndices.append(planeIndex)
        continue
        
    if len(layoutPlaneIndices) == 0:
        if get_boundary:
            if return_segmentation:
                return layoutSegmentation, {}
            else:
                return {}, {}
        else:
            if return_segmentation:
                return layoutSegmentation
            else:
                return {}
            pass
        pass
            
    layoutPlaneInfo = zip(layoutPlanePoints, layoutPlaneMasks, layoutPlaneIndices)
    layoutPlaneInfo = sorted(layoutPlaneInfo, key=lambda x:-len(x[0]))
    layoutPlanePoints, layoutPlaneMasks, layoutPlaneIndices = zip(*layoutPlaneInfo)


    layout_areas = [len(points) for points in layoutPlanePoints]
    layoutPlaneIndices = np.array(layoutPlaneIndices)    
    layout_plane_depths = plane_depths[layoutPlaneIndices]
    layout_planes = planes[layoutPlaneIndices]
    
    relations = np.zeros((len(layoutPlanePoints), len(layoutPlanePoints)), dtype=np.int32)
    for index_1, points_1 in enumerate(layoutPlanePoints):
        plane_1 = layout_planes[index_1]
        offset_1 = np.linalg.norm(plane_1)
        normal_1 = plane_1 / max(offset_1, 1e-4)
        uv_1 = np.round(points_1.mean(0)).astype(np.int32)
        depth_value_1 = layout_plane_depths[index_1, uv_1[1], uv_1[0]]
        point_1 = np.array([(uv_1[0] - camera[2]) / camera[0], 1, -(uv_1[1] - camera[3]) / camera[1]]) * depth_value_1        
        for index_2, points_2 in enumerate(layoutPlanePoints):
            if index_2 <= index_1:
                continue
            plane_2 = layout_planes[index_2]
            offset_2 = np.linalg.norm(plane_2)
            normal_2 = plane_2 / max(offset_2, 1e-4)            
            if np.abs(np.dot(normal_2, normal_1)) > parallelThreshold:
                continue
            uv_2 = np.round(points_2.mean(0)).astype(np.int32)
            depth_value_2 = layout_plane_depths[index_2, uv_2[1], uv_2[0]]
            point_2 = np.array([(uv_2[0] - camera[2]) / camera[0], 1, -(uv_2[1] - camera[3]) / camera[1]]) * depth_value_2

            if np.dot(normal_1, point_2 - point_1) <= 0 and np.dot(normal_2, point_1 - point_2) < 0:
                relations[index_1][index_2] = 1
                relations[index_2][index_1] = 1                
            else:
                relations[index_1][index_2] = 2
                relations[index_2][index_1] = 2                
                pass
            continue
        continue
    
    
    combinations = []
    indices = range(len(layoutPlaneIndices))
    for num_planes in range(2, len(layoutPlaneIndices) + 1):
        for plane_indices in itertools.combinations(indices, num_planes):
            combinations.append((plane_indices, sum([layout_areas[plane_index] for plane_index in plane_indices])))
            continue
        continue
    combinations = sorted(combinations, key=lambda x:-x[1])
    combinations = [combination for combination in combinations if combination[1] > layout_areas[0]]

    valid_mask = depth > 1e-4
    valid_area = valid_mask.sum()
    layout_found = False
    layout_plane_depths[layout_plane_depths < 1e-4] = 10
    plane_mask_dict = {}
    for combination, area in combinations:
        combination = np.array(combination)        
        depths = layout_plane_depths[combination]
        combination_depth = np.zeros(segmentation.shape)
        for plane_index_1 in combination:
            plane_mask = np.ones(segmentation.shape, dtype=np.bool)
            for plane_index_2 in combination:                
                if plane_index_2 == plane_index_1:
                    continue                
                if (plane_index_1, plane_index_2) not in plane_mask_dict:
                    if relations[plane_index_1][plane_index_2] == 0:
                        plane_mask_dict[(plane_index_1, plane_index_2)] = 1 - layoutPlaneMasks[plane_index_2]
                        plane_mask_dict[(plane_index_2, plane_index_1)] = 1 - layoutPlaneMasks[plane_index_1]
                    elif relations[plane_index_1][plane_index_2] == 1:
                        plane_mask_dict[(plane_index_1, plane_index_2)] = layout_plane_depths[plane_index_1] < layout_plane_depths[plane_index_2]
                        plane_mask_dict[(plane_index_2, plane_index_1)] = layout_plane_depths[plane_index_1] > layout_plane_depths[plane_index_2]
                    else:
                        plane_mask_dict[(plane_index_1, plane_index_2)] = layout_plane_depths[plane_index_1] > layout_plane_depths[plane_index_2]
                        plane_mask_dict[(plane_index_2, plane_index_1)] = layout_plane_depths[plane_index_1] < layout_plane_depths[plane_index_2]
                        pass
                    pass
                plane_mask = np.logical_and(plane_mask, plane_mask_dict[(plane_index_1, plane_index_2)])
                continue
            combination_depth[plane_mask] = layout_plane_depths[plane_index_1][plane_mask]
            continue
        if ((combination_depth < depth - 0.2) * valid_mask).sum() > valid_area * 0.1:
            continue
        combination_segmentation = combination[depths.argmin(0)]
        combination_segmentation[combination_depth >= 10] = -1
        consistent_area = (combination_segmentation == layoutSegmentation).sum()
        if consistent_area < area * 0.9:
            continue
        layout = layoutPlaneIndices[combination_segmentation]
        layout[combination_segmentation < 0] = -1
        if get_boundary:
            print(layoutPlaneIndices)
            boundaries = {}
            for plane_index_1 in combination:
                for plane_index_2 in combination:
                    if plane_index_2 <= plane_index_1:
                        continue
                    if relations[plane_index_1][plane_index_2] == 0:
                        continue
                    plane_mask = plane_mask_dict[(plane_index_1, plane_index_2)].astype(np.uint8)
                    boundaries[(layoutPlaneIndices[plane_index_1], layoutPlaneIndices[plane_index_2])] = (cv2.dilate(plane_mask, np.ones((3, 3))) - cv2.erode(plane_mask, np.ones((3, 3))), relations[plane_index_1][plane_index_2])
                    continue
                continue
            return layout, boundaries
        return layout

    layoutSegmentation[layout_plane_depths[0] > 1e-4] = layoutPlaneIndices[0]
    if get_boundary:
        return layoutSegmentation, {}
    else:
        return layoutSegmentation
    
    layoutVisibleDepth = np.zeros(segmentation.shape)
    for layout_index, points in enumerate(layoutPlanePoints):
        xs = points[:, 0]
        ys = points[:, 1]
        layoutVisibleDepth[ys, xs] = layoutPlaneDepths[ys, xs, layout_index]
        continue


    invalidMask = {}
    while True:
        hasChange = False
        layoutMasks = {}
        for index_1, plane_1 in enumerate(layoutPlanes):
            if index_1 in invalidMask:
                continue
            layoutMask = layoutPlaneDepths[:, :, index_1] > 1e-4
            for index_2, plane_2 in enumerate(layoutPlanes):
                if index_2 == index_1:
                    continue
                if index_2 in invalidMask:
                    continue
                if relations[index_1][index_2] == 0:
                    continue
                elif relations[index_1][index_2] == 1:
                    layoutMask = np.logical_and(layoutMask, np.logical_or(layoutPlaneDepths[:, :, index_1] <= layoutPlaneDepths[:, :, index_2], layoutPlaneDepths[:, :, index_2] < 1e-4))
                else:
                    layoutMask = np.logical_and(layoutMask, layoutPlaneDepths[:, :, index_1] >= layoutPlaneDepths[:, :, index_2])
                    pass
                continue
            
            if np.logical_and(layoutPlaneMasks[index_1], layoutMask).sum() < layoutPlaneMasks[index_1].sum() * 0.9:
                print('invalid', index_1, np.logical_and(layoutPlaneMasks[index_1], layoutMask).sum(), layoutPlaneMasks[index_1].sum() * 0.9)
                hasChange = True
                invalidMask[index_1] = True
                break                
            validLayoutMask = np.logical_and(layoutMask, layoutVisibleDepth > 1e-4)
            layoutDepth = layoutPlaneDepths[:, :, index_1][validLayoutMask]
            visibleDepth = layoutVisibleDepth[validLayoutMask]
            if (layoutDepth < visibleDepth).sum() > len(visibleDepth) * 0.1:
                print('invalid depth', index_1, (layoutDepth < visibleDepth).sum(), len(visibleDepth) * 0.1)
                hasChange = True
                invalidMask[index_1] = True
                break
            layoutMasks[index_1] = layoutMask            
            continue
        if hasChange:
            continue
        for layoutIndex, layoutMask in layoutMasks.items():
            layoutSegmentation[layoutMask] = layoutPlaneIndices[layoutIndex]
            continue
        break

    if return_segmentation:
        return layoutSegmentation
    else:
        return {}
    
    
## Get structures
def getStructures(image, planes, plane_info, segmentation, depth, camera):
    parallelThreshold = np.cos(np.deg2rad(30))

    planePoints = []
    invalidPlanes = {}
    for planeIndex in range(len(planes)):
        mask = segmentation == planeIndex
        ys, xs = mask.nonzero()
        if len(ys) == 0:
            planePoints.append([])
            invalidPlanes[planeIndex] = True
            continue
        planePoints.append(np.round(np.array([xs.mean(), ys.mean()])).astype(np.int32))
        continue

    structurePlanesMap = {}
    individualPlanes = []
    for planeIndex, info in enumerate(plane_info):
        if planeIndex in invalidPlanes:
            continue
        if len(info) == 1:
            individualPlanes.append(planeIndex)
            continue
        for structureIndex, _ in info[1:]:
            if structureIndex not in structurePlanesMap:
                structurePlanesMap[structureIndex] = []
                pass
            structurePlanesMap[structureIndex].append(planeIndex)
            continue
        continue

    relations = np.zeros((len(planes), len(planes)), dtype=np.int32)
    structures = []
    for structurePlaneIndices in structurePlanesMap.values():
        if len(structurePlaneIndices) == 1:
            if structurePlaneIndices[0] not in individualPlanes:
                individualPlanes.append(structurePlaneIndices[0])
                pass
            continue
        
        planePairs = itertools.combinations(structurePlaneIndices, 2)
        planePairs = np.array(list(planePairs))
        for planeIndex_1, planeIndex_2 in planePairs:
            if relations[planeIndex_1][planeIndex_2] != 0:
                continue            
            plane_1 = planes[planeIndex_1]
            offset_1 = np.linalg.norm(plane_1)
            normal_1 = plane_1 / max(offset_1, 1e-4)
            
            plane_2 = planes[planeIndex_2]
            offset_2 = np.linalg.norm(plane_2)
            normal_2 = plane_2 / max(offset_2, 1e-4)

            if np.abs(np.dot(normal_2, normal_1)) > parallelThreshold:
                continue

            uv_1 = planePoints[planeIndex_1]
            depth_1 = depth[uv_1[1], uv_1[0]]
            point_1 = np.array([(uv_1[0] - camera[2]) / camera[0] * depth_1, depth_1, -(uv_1[1] - camera[3]) / camera[1] * depth_1])
            
            uv_2 = planePoints[planeIndex_2]
            depth_2 = depth[uv_2[1], uv_2[0]]
            point_2 = np.array([(uv_2[0] - camera[2]) / camera[0] * depth_2, depth_2, -(uv_2[1] - camera[3]) / camera[1] * depth_2])                
            
                
            if np.dot(normal_1, point_2 - point_1) <= 0 and np.dot(normal_2, point_1 - point_2) < 0:
                relations[planeIndex_1][planeIndex_2] = 1
                relations[planeIndex_2][planeIndex_1] = 1                
            else:
                relations[planeIndex_1][planeIndex_2] = 2
                relations[planeIndex_2][planeIndex_1] = 2                
                pass
            continue

        planePairs = np.array(list(planePairs))
        planePairRelations = relations[planePairs[:, 0], planePairs[:, 1]]
        numConvex = (planePairRelations == 1).sum()
        numConcave = (planePairRelations == 2).sum()

        if numConvex == 0 and numConcave == 0:
            for planeIndex in structurePlaneIndices:
                if planeIndex not in individualPlanes:                
                    individualPlanes.append(planeIndex)
                    pass
                continue
        elif numConcave == 0:
            structures.append((structurePlaneIndices, 0))
        elif numConvex == 0:
            structures.append((structurePlaneIndices, 1))
        else:
            targetRelation = 1 if numConvex > numConcave else 2
            planePairs = planePairs[planePairRelations == targetRelation]
            adjacency_matrix = np.diag(np.ones(len(planes), dtype=np.bool))
            adjacency_matrix[planePairs[:, 0], planePairs[:, 1]] = True
            adjacency_matrix[planePairs[:, 1], planePairs[:, 0]] = True
            usedMask = {}
            groupPlaneIndices = (adjacency_matrix.sum(-1) > 1).nonzero()[0]
            for groupPlaneIndex in groupPlaneIndices:
                if groupPlaneIndex in usedMask:
                    continue
                groupStructure = adjacency_matrix[groupPlaneIndex].copy()
                for neighbor in groupStructure.nonzero()[0]:
                    if np.any(adjacency_matrix[neighbor] < groupStructure):
                        groupStructure[neighbor] = 0
                        pass
                    continue
                groupStructure = groupStructure.nonzero()[0]
                if len(groupStructure) == 1:
                    if groupStructure[0] not in individualPlanes:
                        individualPlanes.append(groupStructure[0])
                        pass
                else:
                    structures.append((structurePlaneIndices, targetRelation - 1))
                    pass
                for planeIndex in groupStructure:
                    usedMask[planeIndex] = True
                    continue
                continue
            for planeIndex in structurePlaneIndices:
                if planeIndex not in usedMask:
                    if planeIndex not in individualPlanes:
                        individualPlanes.append(planeIndex)
                        pass
                    pass
                continue
            pass
        continue
    structures += [([planeIndex], 0) for planeIndex in individualPlanes]        

    labelStructures = {}
    for structureIndex, (planeIndices, convex) in enumerate(structures):
        structurePlanes = [planes[planeIndex] for planeIndex in planeIndices]
        if len(structurePlanes) == 1:
            if 0 not in labelStructures:
                labelStructures[0] = []
                pass
            mask = segmentation == planeIndices[0]
            labelStructures[0].append((structurePlanes[0], mask))
            continue

        convex = convex == 0
        
        masks = []
        for planeIndex in planeIndices:
            masks.append(segmentation == planeIndex)
            continue
        mask = np.any(np.array(masks), axis=0)
        
        structurePlanes = np.array(structurePlanes)
        structurePlaneDepths = calcPlaneDepths(structurePlanes, segmentation.shape[1], segmentation.shape[0], camera, max_depth=-1)
        if convex:
            structurePlaneDepths[structurePlaneDepths < 1e-4] = 10
            structurePlaneDepth = structurePlaneDepths.min(-1)
        else:
            structurePlaneDepth = structurePlaneDepths.max(-1)
            pass

        structureDepth = structurePlaneDepth[mask]
        visibleDepth = depth[mask]
        validMask = visibleDepth > 1e-4
        structureDepth = structureDepth[validMask]
        visibleDepth = visibleDepth[validMask]

        if (np.abs(structureDepth - visibleDepth) > 0.1).sum() > len(visibleDepth) * 0.2:

            if 0 not in labelStructures:
                labelStructures[0] = []
                pass            
            for planeIndex, mask in zip(planeIndices, masks):
                labelStructures[0].append((planes[planeIndex], mask))
                continue
            continue
                
        structurePlanes = sorted(structurePlanes, key=lambda x: x[0])
        if len(planeIndices) == 3:
            dotProducts = [np.abs(plane[2] / max(np.linalg.norm(plane), 1e-4)) for plane in structurePlanes]
            horizontalPlaneIndex = np.array(dotProducts).argmax()
            structurePlanes = [structurePlanes[horizontalPlaneIndex]] + structurePlanes[:horizontalPlaneIndex] + structurePlanes[horizontalPlaneIndex + 1:]
            pass

        parameters = np.concatenate(structurePlanes, axis=0)
        if convex:
            label = (len(planeIndices) - 2) * 2 + 1
        else:
            label = (len(planeIndices) - 2) * 2 + 2            
            pass
        if label not in labelStructures:
            labelStructures[label] = []
            pass
        labelStructures[label].append((parameters, mask))
        continue

    return labelStructures


def crossProductMatrix(vector):
    matrix = np.array([[0, -vector[2], vector[1]],
                       [vector[2], 0, -vector[0]],
                       [-vector[1], vector[0], 0]])
    return matrix
    
def rotationMatrixToAxisAngle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= max(np.linalg.norm(axis), 1e-8)
    return axis, angle

def axisAngleToRotationMatrix(axis, angle):
    diag = np.diag(np.ones(3))
    K = crossProductMatrix(axis)
    R = diag + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)
    return R

## Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

## Calculates rotation matrix to euler angles
## The result is the same as MATLAB except the order
## of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

## Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

## Compute the transformation given point matches
def calcTransformation(points_1, points_2):
    center_1 = points_1.mean(0)
    center_2 = points_2.mean(0)
    H = np.matmul((points_1 - center_1).transpose(), (points_2 - center_2))
    U, S, V = np.linalg.svd(H)
    
    R = np.matmul(V.transpose(), U.transpose())
    if np.linalg.det(R) < 0 and False:
        R[:, 2] *= -1
        pass
    t = -np.matmul(R, center_1) + center_2
    return R, t

## Compute transformation given two lists of planes based on RANSAC
def calcTransformationRANSAC(planes_1, planes_2):
    if len(planes_1) < 2 or len(planes_2) < 2:
        return np.zeros((3, 3)), np.zeros(3)
    numIterations = 1000
    all_points_1 = planes_1 / np.maximum(pow(np.linalg.norm(planes_1, axis=-1, keepdims=True), 2), 1e-4)
    all_points_2 = planes_2 / np.maximum(pow(np.linalg.norm(planes_2, axis=-1, keepdims=True), 2), 1e-4)

    all_indices_1 = np.arange(len(planes_1), dtype=np.int32)
    all_indices_2 = np.arange(len(planes_2), dtype=np.int32)
    bestTransformation = (0, (np.zeros((3, 3)), np.zeros(3)))
    distanceThreshold = 0.2
    for iteration in range(numIterations):
        indices_1 = np.random.choice(all_indices_1, 2, replace=False)
        points_1 = all_points_1[indices_1]        
        indices_2 = np.random.choice(all_indices_2, 2, replace=False)
        points_2 = all_points_2[indices_2]

        R, t = calcTransformation(points_1, points_2)

        transformed_points = np.matmul(R, all_points_1.transpose()).transpose() + t
        distances = np.linalg.norm(np.expand_dims(transformed_points, 1) - all_points_2, axis=-1).min(-1)
        
        inlierMask = distances < distanceThreshold
        numInliers = inlierMask.sum()

        if numInliers > bestTransformation[0]:
            bestTransformation = [numInliers, (R, t)]
            pass
        continue
    R, t = bestTransformation[1]
    transformed_points = np.matmul(R, all_points_1.transpose()).transpose() + t
    distances = np.linalg.norm(np.expand_dims(transformed_points, 1) - all_points_2, axis=-1)
    indices = distances.argmin(-1)
    inlierMask = distances.min(-1) < distanceThreshold
    R, t = calcTransformation(all_points_1[inlierMask], all_points_2[indices][inlierMask])
    return R, t

## Write image paths to HTML
def writeHTML(folder, info_list, numImages, labels=[], convertToImage=False, image_width=-1, filename=''):
    from simple_html import HTML

    h = HTML('html')
    h.p('Results')
    h.br()
    t = h.table(border='1')    
    if len(labels) == len(info_list):
        r_inp = t.tr()
        for label in labels:
            r_inp.td(label)
            continue
        pass
    
    for index in range(numImages):
        r_inp = t.tr()
        r_inp.td(str(index))
        for info in info_list:
            if image_width > 0:
                r_inp.td().img(src=str(index) + '_' + info + '.png', width=str(image_width))
            else:
                r_inp.td().img(src=str(index) + '_' + info + '.png')
                pass
            continue
        continue
    h.br()

    html_file = open(folder + '/index.html', 'w')
    html_file.write(str(h))
    html_file.close()
    if convertToImage:
        import imgkit
        filename = filename if filename != '' else folder.split('/')[-1]
        print(folder + '/' + filename + '.jpg')
        imgkit.from_file(folder + '/index.html', folder + '/' + filename + '.jpg')
        pass
    return

def writeHTMLComparison(filename, folders, common_info_list, comparison_info_list, numImages, convertToImage=False):
    from simple_html import HTML

    h = HTML('html')
    h.p('Results')
    h.br()
    for index in range(numImages):
        t = h.table(border='1')
        r_inp = t.tr()
        for info in common_info_list:
            r_inp.td().img(src=folders[0] + '/' + str(index) + '_' + info + '.png')
            continue
        for folder in folders:
            for info in comparison_info_list:
                r_inp.td().img(src=folder + '/' + str(index) + '_' + info + '.png')
                continue
            continue        
        h.br()
        continue

    html_file = open(filename, 'w')
    html_file.write(str(h))
    html_file.close()
    if convertToImage:
        import imgkit
        imgkit.from_file(filename, filename.replace('html', 'jpg'))
        pass
    return

def one_hot(values, depth):
    maxInds = values.reshape(-1)
    results = np.zeros([maxInds.shape[0], depth])
    results[np.arange(maxInds.shape[0]), maxInds] = 1
    results = results.reshape(list(values.shape) + [depth])
    return results

def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-4)

if __name__ == '__main__':
    from config import Config
    from models.modules import optimizeDetectionModule
    config = Config()
    config.GLOBAL_MASK = False
    config.loadAnchorPlanes('joint')
    
    detection_pair, input_pair = torch.load('test/debug.pth')
    optimized_pair = optimizeDetectionModule(config, detection_pair, input_pair)
    print([torch.norm(input_dict['parameters'] - detection_dict['detection'][:, 6:9]) for input_dict, detection_dict in zip(input_pair, optimized_pair)])
    exit(1)
    
    image = cv2.imread('test/image.png')
    pred_dict = predictPlaneNet(image)
    cv2.imwrite('test/planenet_segmentation.png', drawSegmentationImage(pred_dict['segmentation'], blackIndex=10))
    cv2.imwrite('test/planenet_depth.png', drawDepthImage(pred_dict['depth']))
    exit(1)
    planes_1 = np.load('test/planes_1.npy')
    planes_2 = np.load('test/planes_2.npy')
    pose = np.load('test/pose.npy')
    all_points_1 = planes_1 / np.maximum(pow(np.linalg.norm(planes_1, axis=-1, keepdims=True), 2), 1e-4)
    all_points_2 = planes_2 / np.maximum(pow(np.linalg.norm(planes_2, axis=-1, keepdims=True), 2), 1e-4)    
    R_pred, t = calcTransformationRANSAC(planes_1, planes_2)
    R_gt = axisAngleToRotationMatrix(pose[3:6], pose[6])
    for name, R in [('gt', R_gt), ('pred', R_pred)]:
        transformed_points = np.matmul(R, all_points_1.transpose()).transpose() + pose[:3]
        distances = np.linalg.norm(np.expand_dims(transformed_points, 1) - all_points_2, axis=-1).min(-1)
        print(name, distances.mean(), distances.min(), distances.max())
        print(R)
        print(rotationMatrixToAxisAngle(R))
        continue
    pass
