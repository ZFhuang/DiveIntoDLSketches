import math
import numpy as np
import torch
from collections import namedtuple

Pred_BB_Info = namedtuple(
    "Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])


def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    Generate multiple anchor boxes for feature map. 

    Args:
        feature_map (tensor): target image want to detect
        sizes (list, optional): sizes of the boxes. the real boxes width is 
    imagewidth*size*sqrt(ratio), height is imagewidth*size/sqrt(ratio). Defaults
    to [0.75, 0.5, 0.25].
        ratios (list, optional): width and height ratios of boxes. Defaults to 
    [1, 2, 0.5].

    Returns:
        tensor: every information of boxes: [batch size, number of boxes, two 
    positions describing the boxes that are left-up corner and the right-bottom]
    """
    pairs = []
    # pair of (size, sqrt(ration)). only construct by sizes[0] and ratios[0] in
    # order to reduce performance consumption
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    # constructing the pairs
    pairs = np.array(pairs)
    # width ratio
    ss1 = pairs[:, 0] * pairs[:, 1]
    # height ratio
    ss2 = pairs[:, 0] / pairs[:, 1]

    # using stack to batch processing the position and area of boxes
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    h, w = feature_map.shape[-2:]
    # ratio positions of every pixels' width and height
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h
    # put them into a grid
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    # reshape to make a list
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # using stack to batch processing the position and area of boxes
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

    # make the results
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


def compute_intersection(set_1, set_2):
    """
    Compute the intersection between set_1 and set_2.

    Args:
        set_1 (tensor): n1 boxes with 4 positions illustrate the area
        set_2 (tensor): n2 boxes with 4 positions illustrate the area

    Returns:
        tensor: intersections of all sets comparision
    """
    # PyTorch auto-broadcasts singleton dimensions
    # find out the center little area positions
    # unsqueeze is used to extend one dimension
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(
        1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(
        1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    # get widths and heights
    intersection_dims = torch.clamp(
        upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    # multiply to get spaces
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def compute_jaccard(set_1, set_2):
    """
    Compute the jaccard.(IoU)

    Args:
        set_1 (tensor): n1 boxes with 4 positions illustrate the area
        set_2 (tensor): n2 boxes with 4 positions illustrate the area

    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of 
        the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * \
        (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * \
        (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(
        1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    Assign the best anchor to current target box.

    Args:
        bb: real bounding box, shape:（nb, 4）
        anchor: target anchor waiting be assigned, shape:（na, 4）
        jaccard_threshold: float, threshold may be used

    Returns:
        assigned_idx: shape: (na, ), return the assigned indexes, if return -1 
    means this anchor is background 
    """
    na = anchor.shape[0]
    nb = bb.shape[0]
    jaccard = compute_jaccard(
        anchor, bb).detach().cpu().numpy()  # shape: (na, nb)
    assigned_idx = np.ones(na) * -1  # init

    # assign every best bounding boxes at first
    jaccard_cp = jaccard.copy()
    for j in range(nb):
        i = np.argmax(jaccard_cp[:, j])
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf")  # delete this row

    # assign the other bounding boxes if the jaccard value is below threshold
    for i in range(na):
        if assigned_idx[i] == -1:
            j = np.argmax(jaccard[i, :])
            # threshold checking
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    return torch.tensor(assigned_idx, dtype=torch.long)


def xy_to_cxcy(xy):
    """
    Change the xy storing form.

    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    
    Returns: 
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def MultiBoxTarget(anchor, label):
    """
    Normalize the anchors.
    
    Args:
        anchor: torch tensor, input anchors
        label: real label, shape为(bn, max anchor per image, 5) 5 is 4 positions
    plus one type describtion
               
    Returns:
        a list-> [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: every bbox's offset，(bn，anchorNum*4)
        bbox_mask: mask of bbox, positive is one, else is zero
        cls_labels: labels of bbox 
    """
    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]

    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        # Just compute a single anchor
        an = anc.shape[0]
        # get a assign anchor
        assigned_idx = assign_anchor(lab[:, 1:], anc)  # (anchorNum, )
        # get masks
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)
                     ).repeat(1, 4)  # (anchorNum, 4)
        # assign labels
        cls_labels = torch.zeros(an, dtype=torch.long)  # 0 means background
        assigned_bb = torch.zeros(
            (an, 4), dtype=torch.float32)  # the position of bbox
        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0:  # which is not a background
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1
                assigned_bb[i, :] = lab[bb_idx, 1:]

        center_anc = xy_to_cxcy(anc)  # (center_x, center_y, w, h)
        center_assigned_bb = xy_to_cxcy(assigned_bb)

        # calculate the offsets
        offset_xy = 10.0 * \
            (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        offset_wh = 5.0 * \
            torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], dim=1) * \
            bbox_mask  # (anchorNum, 4)

        return offset.view(-1), bbox_mask.view(-1), cls_labels

    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    # process every input anchors
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(
            anchor[0, :, :], label[b, :, :])
        # get their informations
        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)
    # put them into stack in order to return
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)

    return [bbox_offset, bbox_mask, cls_labels]


def non_max_suppression(bb_info_list, nms_threshold=0.5):
    """
    Output non-maximum suppression bboxes, will delete those overlay bbox

    Args:
        bb_info_list: Pred_BB_Info's list, containing anchor's informations
        nms_threshold: threshold deciding which bbox is overlay
    Returns:
        output: Pred_BB_Info's result list, only contain those thought unique
    """
    output = []
    # sort bbox by confidence
    sorted_bb_info_list = sorted(
        bb_info_list, key=lambda x: x.confidence, reverse=True)
    # check from top confidence to the end
    while len(sorted_bb_info_list) != 0:
        # currently best prediction
        best = sorted_bb_info_list.pop(0)
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)

        # compute jaccard value in order to next step
        # shape: (len(sorted_bb_info_list), )
        iou = compute_jaccard(torch.tensor(
            [best.xyxy]), torch.tensor(bb_xyxy))[0]

        n = len(sorted_bb_info_list)
        # choose those below threshold bbox to remain
        sorted_bb_info_list = [sorted_bb_info_list[i]
                               for i in range(n) if iou[i] <= nms_threshold]
    return output


def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold=0.5):
    """
    Similar to MultiBoxTarget, is for normalize the bbox and acting as an interface

    Args:
        cls_prob: the predictions, shape:(bn, predictTypeNum+1, anchorNum)
        loc_pred: predictions' offsets, shape:(bn, anchorNum*4)
        anchor: MultiBoxPrior's default anchor, shape: (1, anchorNum, 4)
        nms_threshold: threshold deciding which bbox is overlay
    Returns:
        all bboxes' informations, shape: (bn, anchorNum, 6)
        these information is make by [class_id, confidence, xmin, ymin, xmax, ymax]
        class_id=-1 means this bbox has been delete
    """
    assert len(cls_prob.shape) == 3 and len(
        loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold=0.5):
        # help main function compute a single anchor
        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)
               ).detach().cpu().numpy()  # offsets
        # get confidences
        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()

        # construct a data struct storing bbox's informations
        pred_bb_info = [Pred_BB_Info(
            index=i,
            class_id=class_id[i] - 1,
            confidence=confidence[i],
            xyxy=[*anc[i]])
            for i in range(pred_bb_num)]

        # positive's index
        obj_bb_idx = [bb.index for bb in non_max_suppression(
            pred_bb_info, nms_threshold)]

        # for outputs
        output = []
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])
        # return outputs to give to outter function
        return torch.tensor(output)  # shape: (anchorNum, 6)

    # construct batch_outputs
    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(
            cls_prob[b], loc_pred[b], anchor[0], nms_threshold))
    # using stack to output
    return torch.stack(batch_output)
