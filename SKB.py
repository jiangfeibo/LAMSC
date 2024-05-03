import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import copy
import sys

sys.path.append("../..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# generate a single image segment with a mask from SAM
def show_interesting_object(mask, image, ax):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)*image
    mask_image = np.clip(mask_image, 0, 255)
    mask_image = mask_image.astype('uint8')

    # ax.imshow(mask_image)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# implement semantic segmentation with human prompts
def SKB_with_human(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    plt.imshow(image)
    count = 1
    while True:
        # select the interesting segments by input the coordinates of interesting objects.
        select_points = input("Enter the coordinate values in sequence, separated by spaces:")
        if select_points == "":
            break
        select_points = select_points.split(" ")
        select_points = np.array([int(val) for val in select_points]).reshape(-1,2)
        print(select_points)

        input_label = np.array([1 for i in range(select_points.shape[0])])

        # generate masks using SAM according input coordinates
        masks, _, _ = predictor.predict(
            point_coords=select_points,
            point_labels=input_label,
            multimask_output=False,
        )

        plt.figure(figsize=(10,10))
        show_interesting_object(masks,image,plt.gca())
        plt.axis('off')
        plt.savefig(f"res_{count}.png",bbox_inches='tight', pad_inches=0)
        plt.show()
        count += 1

# automatically implement semantic segmentation
def SKB_with_auto(image_path,device="cuda"):
    save_path = os.path.join("data/segments")
    img_name = image_path.split(os.path.sep)[-1]
    seg_dir = os.path.join(save_path, img_name.replace('.jpg', '').replace('.png', ''))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_segment_num = 5 # Number of segments retained
    # load SAM
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               points_per_side=32,
                                               # points_per_batch=64,
                                               pred_iou_thresh=0.86,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=400,
                                               )

    # generate masks using SAM automatically
    masks = mask_generator.generate(image)
    plt.figure()
    # obtain image segments
    os.makedirs(save_path, exist_ok=True)
    count = 0
    masks = sorted(masks,key=lambda x:np.sum(x['segmentation']),reverse=True)
    if len(masks) > max_segment_num:# remove too small segments
        masks = masks[:max_segment_num]
    for mask in masks:
        show_interesting_object(mask['segmentation'], image, plt.gca())
        plt.axis('off')
        os.makedirs(seg_dir, exist_ok=True)
        seg_save_path = os.path.join(seg_dir,f"{str(count).zfill(4)}.jpg")
        print(seg_save_path)
        plt.savefig(seg_save_path, bbox_inches='tight', pad_inches=0)
        count += 1
        # plt.show()

if __name__ == '__main__':
    device = "cpu"
    image_path = "data/raw_images"
    for img in os.listdir(image_path):
        img_path = os.path.join(image_path,img)
        SKB_with_auto(img_path,device)