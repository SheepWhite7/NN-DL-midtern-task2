import json
import os
import cv2
import numpy as np
from tqdm import tqdm

def image_id_to_filename(image_id):
    image_id_str = str(image_id)
    assert len(image_id_str) == 10, 'image_id格式应该是10位数字'
    year = image_id_str[:4]
    serial = image_id_str[4:]
    return f"{year}_{serial}.png"

def hex_to_bgr(hex_color):
    """#e0e0c0 -> (192,224,224) OpenCV是BGR"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4,2,0))

def mask_by_color(img, hex_color, thresh=10):
    """img: BGR图片, hex_color: '#e0e0c0', thresh: 容忍容差"""
    target = np.array(hex_to_bgr(hex_color), dtype=np.uint8)
    lower = np.clip(target-thresh, 0, 255)
    upper = np.clip(target+thresh, 0, 255)
    mask = cv2.inRange(img, lower, upper)
    return mask

def find_color_polygons(image_path, hex_color="#e0e0c0", thresh=10, simplify_epsilon=1.0):
    img = cv2.imread(image_path)  # 彩色
    if img is None:
        print(f"Warning: Image not found: {image_path}")
        return []
    mask = mask_by_color(img, hex_color, thresh=thresh)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        contour = cv2.approxPolyDP(contour, simplify_epsilon, True)
        if contour.shape[0] < 3:
            continue
        poly = contour.reshape(-1).astype(float).tolist()
        polygons.append(poly)
    return polygons

def main():
    # 路径设置
    json_path = "/mnt/tidal-alsh01/usr/yangshiyue/mmdetection/data/train2017.json"
    segm_folder = "/mnt/tidal-alsh01/usr/yangshiyue/mmdetection/data/VOCdevkit/VOC2012_with_seg/SegmentationClass"
    out_path = "/mnt/tidal-alsh01/usr/yangshiyue/mmdetection/data/coco/annotations/train2017.json"
    # 目标颜色与阈值设置
    target_color = "#e0e0c0"
    color_thresh = 10      # 如需要可调整
    simplify_epsilon = 1.0 # 多边形简化

    with open(json_path, 'r') as f:
        coco = json.load(f)

    for ann in tqdm(coco['annotations']):
        image_id = ann['image_id']
        seg_filename = image_id_to_filename(image_id)
        seg_path = os.path.join(segm_folder, seg_filename)
        polygons = find_color_polygons(seg_path, hex_color=target_color, thresh=color_thresh, simplify_epsilon=simplify_epsilon)
        if polygons:
            ann['segmentation'] = polygons
        else:
            ann['segmentation'] = []

    with open(out_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print('DONE')

if __name__ == '__main__':
    main()
