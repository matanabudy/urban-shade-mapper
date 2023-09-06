# import numpy as np
# import os
# import torch
# import matplotlib.pyplot as plt
# import cv2
# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
# from segformer import segformer_segmentation as segformer_func
# from config.ade20k_id2label import CONFIG as id2label
#
# IMAGE_PATH = "imgs/"
# OUTPUT_PATH = "output/"
#
# class ImageSegmenter:
# def is_blue(mask):
#
#     blue_min = np.array([100, 150, 200])  # Lighter blue
#     blue_max = np.array([215, 230, 255])  # Even lighter blue
#     blue_pixels = image[mask]
#
#     mean_color = np.mean(blue_pixels, axis=0)
#     return np.all(blue_min <= mean_color) and np.all(mean_color <= blue_max)
#
#
# def show_image(image, blue_masks):
#
#     for mask in blue_masks:
#         indices = np.where(mask['segmentation'])
#         image[indices] = [128, 0, 128]  # purple
#     plt.imshow(image)
#     plt.show()
#
#
# if __name__ == '__main__':
#     print("Initializing SAM model...")
#     sam_checkpoint = "sam_vit_h_4b8939.pth"
#     model_type = "vit_h"
#     device = "cuda"
#
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     mask_generator = SamAutomaticMaskGenerator(sam)
#
#     print("Initializing Segformer model...")
#     semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
#         "nvidia/segformer-b5-finetuned-ade-640-640")
#     semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
#         "nvidia/segformer-b5-finetuned-ade-640-640").to(device)
#
#     for img in os.listdir("imgs"):
#         print("Loading image...")
#         image = cv2.imread(f"{IMAGE_PATH}{img}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         print("Generating mask...")
#         masks = mask_generator.generate(image)
#         print("Infering segmentation...")
#         class_names = []
#         class_ids = segformer_func(image, semantic_branch_processor, semantic_branch_model, 0)
#         semantc_mask = class_ids.clone()
#         # sort by size and remove small segments
#         sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
#         sorted_masks = [m for m in sorted_masks if m['area'] > 0.02 * image.shape[0] * image.shape[1]]
#         #inference
#         for mask in sorted_masks:
#             valid_mask = mask['segmentation']
#             propose_classes_ids = class_ids[valid_mask]
#             num_class_proposals = len(torch.unique(propose_classes_ids))
#             if num_class_proposals == 1:
#                 semantc_mask[valid_mask] = propose_classes_ids[0]
#                 mask['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
#                 mask['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
#                 class_names.append(mask['class_name'])
#             else:
#                 top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
#                 top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]
#                 semantc_mask[valid_mask] = top_1_propose_class_ids
#                 mask['class_name'] = top_1_propose_class_names[0]
#                 mask['class_proposals'] = top_1_propose_class_names[0]
#                 class_names.append(mask['class_name'])
#
#             if mask['class_name'] == "sky":
#                 indices = np.where(mask['segmentation'])
#                 image[indices] = [128, 0, 128]  # purple
#
#             elif mask['class_name'] == "tree":
#                 indices = np.where(mask['segmentation'])
#                 image[indices] = [0, 255, 0]  # green
#
#         plt.imshow(image)
#         plt.show()
#         plt.imsave(f"{OUTPUT_PATH}{img.split('.')[0]}_seg.png", image)
#
#     print("Done!")
#
#
#
#     # for img in os.listdir("imgs"):
#     #     print(f"Generating mask for {img}")
#     #     image = cv2.imread(f"imgs/{img}")
#     #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #     masks = mask_generator.generate(image)
#     #     # sort masks by area
#     #     sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
#     #     # remove all masks with area smaller than 5% of image area
#     #     sorted_masks = [m for m in sorted_masks if m['area'] > 0.05 * image.shape[0] * image.shape[1]]
#     #     print("Checking if masks are blue...")
#     #     blue_masks = []
#     #     for msk in sorted_masks:
#     #         mask = msk['segmentation']
#     #         if is_blue(mask):
#     #             blue_masks.append(msk)
#     #
#     #     show_image(image, blue_masks)
#     # print("Done!")
#     # exit(0)
#
#
