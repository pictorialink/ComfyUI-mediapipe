import json
from ..services.holistic import holistic_static
import comfy
import torch
import numpy as np


class MediaPipeHolistic():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "tensor_image" : ("IMAGE", {}) },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "execute"
    CATEGORY = "mediapipe"
    def execute(self, tensor_image):
        show_pbar=True
        batch_size = tensor_image.shape[0]
        if show_pbar:
            pbar = comfy.utils.ProgressBar(batch_size)
        out_tensor = None
        openpose_dicts = []
        for i, image in enumerate(tensor_image):
            np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
            np_result,openpose_dict = holistic_static(np_image)
            out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
            if out_tensor is None:
                out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
            out_tensor[i] = out
            openpose_dicts.append(openpose_dict)
            if show_pbar:
                pbar.update(1)

        return {
            'ui': { "openpose_json": [json.dumps(openpose_dicts, indent=4)] },
            "result": (out_tensor, openpose_dicts)
        }



