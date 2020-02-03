import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(r"E:\Work\Projects\Youssef\Mask_RCNN-master (1)\Mask_RCNN-master")  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class FruitConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "apple"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0

############################################################
#  Dataset
############################################################

class FruitDataset(utils.Dataset):

    def load_apple(self, dataset_dir,subset):

        #add_class from Dataset class in utils.py
        #Adds new class using source, ID and class name parameters
        self.add_class("apple",1,"apple")

        #Is the current image batch for training or evaluation?
        assert subset in ["train","val"]

        dataset_dir = os.path.join(dataset_dir,subset)

        #annotation_dir is a hard-coded path of the annotations folder, change as appropriate

        annotations_dir = 'E:\\Work\\Projects\Youssef\\acfr-fruit-dataset\\apples\\annotations'

        #Change directory to annotations folder to start reading excel sheets


        #Training or validation subfolder
        annotations_dir = os.path.join(annotations_dir, subset)
        os.chdir(annotations_dir)
        files = os.listdir(annotations_dir)

        #Read each excel file using pandas, and load each image with its respective annotations
        circle_x=[]
        circle_y=[]
        radius=[]
        df = pd.DataFrame()
        for f in files:
            #print(f)
            df = pd.read_csv(f)
            if (len(df.index)!=0):
                circle_x = df['c-x'].to_list()
                circle_y = df['c-y'].to_list()
                radius = df['radius'].to_list()
            circle = [[],[],[]]
            circle[0] = circle_x.copy()
            circle[1] = circle_y.copy()
            circle[2] = radius.copy()
            height = 202
            width = 308
            image_name = os.path.splitext(f)[0]
            image_name += '.png'
            image_path = os.path.join(dataset_dir,image_name)

            self.add_image(
                "apple",
                image_id=f,
                path=image_path,
                width = width, height = height,
                circle = circle
            )

    # Load masks of instances in an i mage
    # Creates new numpy array of the image, initialized with zeroes and instances are expressed as 1
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        #Each image is stored with its corresponding instances defined by center and radius
        circle = image_info["circle"]
        mask = np.zeros([image_info["height"], image_info['width'], len(circle[0])], dtype=np.uint8)
        for i in range(len(circle[0])):
                rr, cc = skimage.draw.circle(circle[1][i],circle[0][i],round(circle[2][i]), (202,308))
                mask[rr,cc,i]=1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference (self, image_id):
        info = self.image_info[image_id]
        if(info['source'] == 'apple'):
            return info['path']
        else:
            super(self.__class__,self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model):
    # Training dataset.
    dataset_train = FruitDataset()
    dataset_train.load_apple(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FruitDataset()
    dataset_val.load_apple(args.dataset, "val")
    dataset_val.prepare()

    # Train
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

############################################################
#  Visualize Dataset
############################################################

def visualizeDataset():
    fruitData = FruitDataset()
    fruitData.load_apple(args.dataset, "train")
    fruitData.prepare()
    image_ids = np.random.choice(fruitData.image_ids, 4)
    for image_id in image_ids:
        image = fruitData.load_image(image_id)
        print(image_id)
        mask, class_ids = fruitData.load_mask(image_id)
        bbox= utils.extract_bboxes(mask)
        visualize.display_top_masks(image, mask, class_ids, fruitData.class_names)
        visualize.display_instances(image, bbox, mask, class_ids, fruitData.class_names)

############################################################
#  Detection
############################################################

def color(image, mask):
	# Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def detect(model, image_path=None):

    image = skimage.io.imread(args.image)
    r = model.detect([image], verbose = 1)[0]
    result = r
    splash = color(image = image, mask = r['masks'])
    file_name = "detectedFile.png"
    ax = get_ax(1)
    visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                "apple", result['scores'], ax=ax,
                                title="Predictions")


    #print(result['rois'], result['masks'],result['scores'])
    skimage.io.imsave(file_name, splash)



############################################################
#  Main Function
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fruits.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Fruit dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "visualize":
        assert args.dataset

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FruitConfig()
    elif args.command == "visualize":
        config = FruitConfig()
    else:
        class InferenceConfig(FruitConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.9
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    if True:
        # Train or evaluate
        if args.command == "train":
            train(model)
        elif args.command == "detect":
            detect(model)
        elif args.command == "visualize":
            visualizeDataset()
        #    detect_and_color_splash(model, image_path=args.image,
        #                            video_path=args.video)
        else:
            print("'{}' is not recognized. "
                  "Use 'train' or 'splash'".format(args.command))
