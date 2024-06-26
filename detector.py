'''
Description: Detection methods for cell detection (Stardist) and tissue detection (Segment Anything).
'''
import numpy as np
import utils


def cell_detector_from_model(img, model, prob=0.5, probnms=0.3):
    result = model.predict_instances(utils.norm(img[:,:,:3]), prob_thresh=prob, nms_thresh=probnms)
    cell_mask = result[0]
    return cell_mask

def tissue_detector_from_model(img, mask_generator):
    img = utils.norm(img[:,:,:3]) * 255
    img = img.astype("uint8")
    masks = mask_generator.generate(img)
    multi_value_mask = np.zeros(masks[0]["segmentation"].shape)
    mask_i = 1
    print("Tissue Mask Generated.")
    for mask in masks:
        seg = mask["segmentation"]
        if np.sum(seg) > 400 * 400:
            continue
        if np.sum(multi_value_mask[seg]) == 0:
            multi_value_mask[seg] = mask_i
            mask_i += 1
            continue
        else:
            unique_values = np.unique(multi_value_mask[seg])
            choose_value = mask_i
            value_sum = np.sum(seg*1)
            for value in unique_values:
                if value == 0:
                    continue
                else:
                    s = np.sum((multi_value_mask==value)*1)
                    if s > value_sum:
                        value_sum = s
                        choose_value = value
            if choose_value == mask_i:
                for value in unique_values:
                    if value == 0:
                        continue
                    else:
                        multi_value_mask[multi_value_mask==value] = 0
                multi_value_mask[seg] = mask_i
                mask_i += 1
            else:
                continue
    max_value = np.max(multi_value_mask)
    tissue_mask = np.zeros(multi_value_mask.shape)
    value = 1
    for i in range(int(max_value)):
        v = i + 1
        if np.sum((multi_value_mask==v)*1) == 0:
            continue
        else:
            tissue_mask[multi_value_mask==v] = value
            value += 1
    return tissue_mask

