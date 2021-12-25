import detectron2.data.transforms as T
from detectron2.engine import DefaultPredictor

import torch


class HalfwayFusionPredictor(DefaultPredictor):
        
    def __call__(self, original_image_rgb, original_image_t):

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image_rgb = original_image_rgb[:, :, ::-1]
                original_image_t = original_image_t[:, :, ::-1]
                
            height, width = original_image_rgb.shape[:2]

            image_rgb = torch.as_tensor(original_image_rgb.astype("float32").transpose(2, 0, 1))
            image_t = torch.as_tensor(original_image_t.astype("float32").transpose(2, 0, 1))

            inputs = {"image_rgb": image_rgb, "image_t": image_t, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            
            return predictions