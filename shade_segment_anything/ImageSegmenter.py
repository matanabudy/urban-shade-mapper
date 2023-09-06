import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from shade_segment_anything.segformer import segformer_segmentation as segformer_func
from shade_segment_anything.config.ade20k_id2label import CONFIG as id2label

from PIL import Image, ImageDraw


def paint_outside_circle_black(image_array):
    width, height, _ = image_array.shape
    circle_center = (width // 2, height // 2)
    radius = min(width, height) // 2
    square_side = 2 * radius
    square_left = circle_center[0] - radius
    square_top = circle_center[1] - radius
    square_right = square_left + square_side
    square_bottom = square_top + square_side

    black_image_array = np.zeros_like(image_array)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((square_left, square_top, square_right, square_bottom), fill=255)
    mask_array = np.array(mask) > 0
    black_image_array[mask_array] = image_array[mask_array]
    black_pixels = width * height - np.sum(mask_array)

    return black_image_array, black_pixels

def count_color_pixels(image_array, color):
    mask = np.all(image_array == color, axis=-1)
    count = np.sum(mask)
    return count

class ImageSegmenter:


    def show_image(image, blue_masks):

        for mask in blue_masks:
            indices = np.where(mask['segmentation'])
            image[indices] = [128, 0, 128]  # purple
        plt.show()

    def __init__(self):
        print("Initializing SAM model...")
        self.sam_checkpoint = "shade_segment_anything/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda"

        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam,
        points_per_side=32,
        pred_iou_thresh=0.65,
        stability_score_thresh=0.75,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2)

        print("Initializing Segformer model...")
        self.semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640")
        self.semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640").to(self.device)

    def segment_image(self, image_path, output_path):
        print("Loading image...")
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("Generating mask...")
        masks = self.mask_generator.generate(image)

        # sort by size and remove small segments
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sorted_masks = [m for m in sorted_masks if m['area'] > 0.02 * image.shape[0] * image.shape[1]]
        debug = False

        print("Infering segmentation...")
        class_names = []
        class_ids = segformer_func(image, self.semantic_branch_processor, self.semantic_branch_model, 0)
        semantc_mask = class_ids.clone()

        if debug:
            for i, mask in enumerate(sorted_masks):
                if i > 20:
                    break
                indices = np.where(mask['segmentation'])
                image[indices] = [np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255)]

            plt.imshow(image)

        else:
            # inference
            for mask in sorted_masks:
                valid_mask = mask['segmentation']
                propose_classes_ids = class_ids[valid_mask]
                num_class_proposals = len(torch.unique(propose_classes_ids))
                if num_class_proposals == 1:
                    semantc_mask[valid_mask] = propose_classes_ids[0]
                    mask['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                    mask['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
                    class_names.append(mask['class_name'])
                else:
                    top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
                    top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in
                                                 top_1_propose_class_ids]
                    semantc_mask[valid_mask] = top_1_propose_class_ids
                    mask['class_name'] = top_1_propose_class_names[0]
                    mask['class_proposals'] = top_1_propose_class_names[0]
                    class_names.append(mask['class_name'])

                if mask['class_name'] == "sky":
                    indices = np.where(mask['segmentation'])
                    image[indices] = [128, 0, 128]  # purple

                elif mask['class_name'] == "tree" or mask['class_name'] == "palm":
                    indices = np.where(mask['segmentation'])
                    image[indices] = [0, 255, 0]  # green

                elif mask['class_name'] == "building":
                    indices = np.where(mask['segmentation'])
                    image[indices] = [255, 0, 0]  # red


        plt.show()
        image, frame_pixels = paint_outside_circle_black(image)
        image_size = image.shape[0] * image.shape[1] - frame_pixels
        pixel_percentages = {}
        pixel_percentages["sky"] = count_color_pixels(image, [128, 0, 128])/image_size
        pixel_percentages["tree"] = count_color_pixels(image, [0, 255, 0])/image_size
        pixel_percentages["building"] = count_color_pixels(image, [255, 0, 0])/image_size
        out_path = f"{str(output_path).split('.')[0]}_seg.png"
        if debug == False:
            plt.imsave(out_path, image)
        return out_path, pixel_percentages
