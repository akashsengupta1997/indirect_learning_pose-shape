import cv2
import os
from matplotlib import pyplot as plt


# ups31_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/images"
# ups31_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/masks"
# ups31_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31/vis_masks"
ups31_path = "/data/cvfs/as2562/4th_year_proj_datasets/upi-s1h/mpii/images/train_images/train"


# save_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/images"
# save_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/masks"
# save_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded/vis_masks"
save_path = "/data/cvfs/as2562/4th_year_proj_datasets/upi-s1h/mpii_padded/images/train"


for image_path in sorted(os.listdir(ups31_path)):
    if image_path.endswith(".png"):
        # print(image_path)
        img = cv2.imread(os.path.join(ups31_path, image_path))
        height, width, _ = img.shape
        # print(height, width)

        if width < height:
            border_width = (height - width)//2
            padded = cv2.copyMakeBorder(img, 0, 0, border_width, border_width,
                                        cv2.BORDER_CONSTANT, value=0)
        else:
            border_width = (width - height)//2
            padded = cv2.copyMakeBorder(img, border_width, border_width, 0, 0,
                                        cv2.BORDER_CONSTANT, value=0)

        cv2.imwrite(os.path.join(save_path, image_path), padded)

        # plt.figure(1)
        # plt.subplot(211)
        # plt.imshow(img)
        # plt.subplot(212)
        # plt.imshow(padded)
        # plt.show()
