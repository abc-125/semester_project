from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator

import logging
import numpy as np
import copy
import torch
import os


def input_fusion_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict) 
        
    # read rgb and thermal images:
    image_rgb = utils.read_image(dataset_dict["file_name_rgb"], format="RGB")
    image_t = utils.read_image(dataset_dict["file_name_t"], format="L")

    # merge it to one 4 channel image:
    image = np.dstack((image_rgb[:,:,2], image_rgb[:,:,1], image_rgb[:,:,0], image_t))
        
    image = torch.from_numpy(image.transpose(2, 0, 1).astype("float32"))

    annos = [annotation for annotation in dataset_dict.pop("annotations")]

    return {
    # create the format that the model expects:
    "image": image,
    "instances": utils.annotations_to_instances(annos, image.shape[1:]),
    "height": dataset_dict["height"],
    "width": dataset_dict["width"],
    "image_id": dataset_dict["image_id"]
    }

class InputFusionTrainer(DefaultTrainer):
    '''Rewrites build_detection_test_loader and build_train_loader with custom mapper,
    loads weigths from 3 channel model into 4 channel model. Uses COCOEvaluator while training.'''

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=input_fusion_mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=input_fusion_mapper)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)

        # use pretrained 3-channel weights for 4-channel model:
        cfg2 = cfg.clone()
        cfg2.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]  # Default values are the mean pixel value from ImageNet
        cfg2.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]  # std has been absorbed into its conv1 weights, so the std needs to be set 1
        model2 = build_model(cfg2)
        DetectionCheckpointer(model2).load(cfg2.MODEL.WEIGHTS)  # load weights into model

        with torch.no_grad():
            conv1_weight = model2.backbone.bottom_up.stem.conv1.weight
            model.backbone.bottom_up.stem.conv1.weight[:, 3] = torch.mean(conv1_weight[:, :3], dim=1)
            model.backbone.bottom_up.stem.conv1.weight[:, :3] = conv1_weight[:, :3]

        print("backbone.bottom_up.stem.conv1.weight is loaded, shape is (64, 4, 7, 7)")

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))

        return model
