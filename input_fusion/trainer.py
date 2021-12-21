from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer

import numpy as np
import copy
import torch


def input_fusion_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict) 
        
    # read thermal and rgb images:
    image_rbg = utils.read_image(dataset_dict["file_name_rgb"], format="RGB")
    image_t = utils.read_image(dataset_dict["file_name_t"], format="L")

    # merge it to one 4 channel image:
    image = np.dstack((image_rbg[:,:,2], image_rbg[:,:,1], image_rbg[:,:,0], image_t))
        
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
    '''Rewrites build_detection_test_loader and build_train_loader with custom mapper'''
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=input_fusion_mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=input_fusion_mapper)
