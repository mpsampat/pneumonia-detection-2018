
# /home/ubuntu/kaggle/RSNA_pneumonia/vikas/logs/pneumonia20181012T0521
DATA_DIR = '/home/ubuntu/kaggle/RSNA_pneumonia/input'
# Directory to save logs and trained model
ROOT_DIR = '/home/ubuntu/kaggle/RSNA_pneumonia/vikas'

import os

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR,exist_ok=True)


os.chdir(ROOT_DIR)
import time
import warnings 
warnings.filterwarnings("ignore")
import os, shutil
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob 
timestr = time.strftime("%m%d%y")
# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Train and Test Directory
train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "pre_trained_weights/mask_44_0.166_rcnn_pneumonia.h5")
# /home/ubuntu/kaggle/RSNA_pneumonia/vikas/pre_trained_weights/mask_rcnn_balloon.h5
# mask_rcnn_pneumonia_kernel.h5 mask_rcnn_coco.h5
# /home/ubuntu/kaggle/RSNA_pneumonia/vikas/logs/pneumonia20181006T1147/mask_rcnn_pneumonia_0044.h5

def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 


class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    BACKBONE = 'resnet101' #resnet101
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1
    MEAN_PIXEL = np.array([124.7, 124.7, 124.7])
    STEPS_PER_EPOCH = 400
    

config = DetectorConfig()
config.display()

class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self) 
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
      #--      
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
      #-- 
    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image
      #-- 
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
anns.head()

image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
image = ds.pixel_array # get image array

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

# split dataset into training vs. validation dataset 
# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)

val_size = 1500
image_fps_val = image_fps_list[:val_size]
image_fps_train = image_fps_list[val_size:]

print(len(image_fps_train), len(image_fps_val))

# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# Show annotation(s) for a DICOM image 
test_fp = random.choice(image_fps_train)
image_annotations[test_fp]

# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


class_ids = [0]
while class_ids[0] == 0:  ## look for a mask
    image_id = random.choice(dataset_train.image_ids)
    image_fp = dataset_train.image_reference(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]


plt.imshow(masked, cmap='gray')
plt.axis('off')

print(image_fp)
print(class_ids)


# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# test on the same image as above
imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
plt.figure(figsize=(30, 12))
_ = plt.imshow(imggrid[:, :, 0], cmap='gray')

model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

# Exclude the last layers because they require a matching
# number of classes
# COCO_WEIGHTS_PATH='/home/ubuntu/kaggle/RSNA_pneumonia/vikas/logs/pneumonia20181012T0521/mask_rcnn_pneumonia_0035.h5'
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

# model.load_weights(model_path, by_name=True, exclude=[
#     "mrcnn_class_logits", "mrcnn_bbox_fc",
#     "mrcnn_bbox", "mrcnn_mask"])

LEARNING_RATE =1e-3
# 1e-4 1e-5 1e-6 0.005
# Train Mask-RCNN Model 
import warnings 
warnings.filterwarnings("ignore")

## train heads with higher lr to speedup the learning
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=10,
            layers='all',
            augmentation=None)  ## no need to augment yet

history = model.keras_model.history.history.copy() ## no need to augment yet

LEARNING_RATE= 1e-4
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=55,
            layers='all',
            augmentation=augmentation)
new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]



epochs = range(1,len(next(iter(history.values())))+1)
loss=pd.DataFrame(history, index=epochs)

loss.to_csv(timestr+'loss.csv')

plt.figure(figsize=(15,5))
plt.subplot(111)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.show()
plt.savefig(timestr+'_1.loss.png')

#-------------

# i start with 1e-3 for epoch 1-25; 1e-4 for epoch 26-50 and 1e-5 for epoch 50-75; 
# i saw in DSB some people in top-10 use Mask_RCNN and start with 1e-4;
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.95

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
# model_path='/home/ubuntu/kaggle/RSNA_pneumonia/vikas/pre_trained_weights/mask_44_0.166_rcnn_pneumonia.h5'
model_path='/home/ubuntu/kaggle/RSNA_pneumonia/vikas/logs/pneumonia20181012T0521/mask_rcnn_pneumonia_0034.h5'
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)
# Make predictions on test images, write out sample submission


# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.96):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        file.write("patientId,PredictionString\n")
        #
        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # view_position= getattr(c_dicom, 'ViewPosition', '')
            # if view_position == 'AP':
            #     min_conf=0.94.5
            # else:
            #     min_conf=0.96.5
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
            #
            patient_id = os.path.splitext(os.path.basename(image_id))[0]
            #
            results = model.detect([image])
            r = results[0]
            #
            out_str = ""
            out_str += patient_id
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])
                #
                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '
                        #
                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)
                        out_str += bboxes_str
            file.write(out_str+"\n")



submission_fp = os.path.join(ROOT_DIR, 'submission/'+timestr+'_submission_3_res101.csv')
predict(test_image_fps, filepath=submission_fp)
print(submission_fp)


output = pd.read_csv(submission_fp)
output.head(60)


#--- --------------visualize  val results ----------------------------------
dataset = dataset_val
save_dir= '/home/ubuntu/kaggle/RSNA_pneumonia/vikas/val_img_pred/'
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
#

os.makedirs(save_dir)
# zip -r val_img_pred.zip val_img_pred


# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors



# for i in range(6):
start_time= time.time()
for image_id in tqdm(dataset.image_ids):
    # https://github.com/matterport/Mask_RCNN/issues/134
    # image_id = random.choice(dataset.image_ids)
    img_name, _ = os.path.splitext(os.path.basename(dataset.image_reference(image_id)))
    c_dicom = pydicom.read_file(dataset.image_reference(image_id), stop_before_pixels=True)
    title= getattr(c_dicom, 'ViewPosition', '')
    #---
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    print(original_image.shape)
    fig, ax=plt.subplots(1,2)
    # fig, ax = plt.subplots()
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names,show_mask=False,
                                colors=get_colors_for_class_ids(gt_class_id), ax=ax[0],title=title)
    # plt.savefig('vikas3_pred.png')
    #https://github.com/matterport/Mask_RCNN/issues/134
    # plt.subplot(6, 2, 2*i + 2)
    results = model.detect([original_image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], show_mask=False,
                                colors=get_colors_for_class_ids(r['class_ids']), ax=ax[-1],title=title)
    #
    plt.savefig(os.path.join(save_dir,img_name+'.png'))


final_time= time.time()

print (final_time - start_time)

# print('time taken: ' {}.format(final_time- start_time))

# def display_instances(image, boxes, masks, class_ids, class_names,
#                       scores=None, title="",
#                       figsize=(16, 16), ax=None,
#                       show_mask=True, show_bbox=True,
# colors=None, captions=None)

#     c_dicom = pydicom.read_file(img, stop_before_pixels=True)
#     title= getattr(c_dicom, 'ViewPosition', '')
#     tag_dict = {c_tag: getattr(c_dicom, c_tag, '') 
#          for c_tag in DCM_TAG_LIST}
# DCM_TAG_LIST = ['PatientAge', 'BodyPartExamined', 'ViewPosition', 'PatientSex']

# getattr(c_dicom, 'ViewPosition', '') 