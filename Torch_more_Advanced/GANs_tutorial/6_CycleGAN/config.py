import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os 

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = FILE_PATH + "/dataset/trainA"
VAL_DIR = FILE_PATH + "/dataset/testA"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = FILE_PATH + "/checkpoints/genh.pth.tar"  # "/checkpoints/trained_genh.pth.tar"
CHECKPOINT_GEN_Z = FILE_PATH + "/checkpoints/genz.pth.tar"  # "/checkpoints/trained_genz.pth.tar"
CHECKPOINT_CRITIC_H = FILE_PATH + "/checkpoints/critich.pth.tar"  # "/checkpoints/trained_critich.pth.tar"
CHECKPOINT_CRITIC_Z = FILE_PATH + "/checkpoints/criticz.pth.tar"  # "/checkpoints/trained_criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
