#%%
import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image
from tqdm import tqdm
import os

# Printing the current working directory
print("Th Current working directory is: {0}".format(cwd))

# This will be the path to your .py file
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
print(FILE_PATH)

IMG_FILE_PATH = os.path.join(FILE_PATH, "images/elon.jpeg")
print(IMG_FILE_PATH)

# sys.exit()

image = Image.open(IMG_FILE_PATH)

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ],
            p=1.0,
        ),
    ]
)

images_list = [image]
image = np.array(image)
for i in tqdm(range(15)):
    augmentations = transform(image=image)
    augmented_img = augmentations["image"]
    images_list.append(augmented_img)
#%%
plot_examples(images_list)

# %%
