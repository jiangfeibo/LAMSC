import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import copy
import sys

sys.path.append("../..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

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

def SKB_with_human(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    plt.imshow(image)
    count = 1
    while True:
        select_points = input("Enter the coordinate values in sequence, separated by spaces:")
        if select_points == "":
            break
        select_points = select_points.split(" ")
        select_points = np.array([int(val) for val in select_points]).reshape(-1,2)
        print(select_points)

        input_label = np.array([1 for i in range(select_points.shape[0])])
        print(input_label)

        masks, _, _ = predictor.predict(
            point_coords=select_points,
            point_labels=input_label,
            multimask_output=False,
        )

        plt.figure(figsize=(10,10))
        show_interesting_object(masks,image,plt.gca())
        # show_mask(masks, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.savefig(f"res_{count}.png",bbox_inches='tight', pad_inches=0)
        plt.show()
        count += 1

def SKB_with_auto(image_path):
    image = cv2.imread(image_path)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                               points_per_side=32,
                                               # points_per_batch=64,
                                               pred_iou_thresh=0.86,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100,
                                               )

    masks = mask_generator.generate(image)
    plt.figure(figsize=(10, 10))
    for i,mask in enumerate(masks):
        show_interesting_object(mask['segmentation'], image, plt.gca())
        plt.axis('off')
        plt.savefig(f"{image_path.replace('.jpg','').replace('.png','')}/{str(i).zfill(4)}.jpg", bbox_inches='tight', pad_inches=0)
        # plt.show()

if __name__ == '__main__':
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    image_path = "result.jpg"
    SKB_with_auto(image_path)
    SKB_with_human(image_path)