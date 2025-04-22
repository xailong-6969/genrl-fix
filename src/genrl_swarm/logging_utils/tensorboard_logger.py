from torch.utils import tensorboard
from typing import Dict, Union

class TensorboardLoggerMixin:
    def __init__(self):
        self.logging_dir = None
        self.tracker = None

    def init_tracker(self, logging_dir: str):
        self.logging_dir = logging_dir
        self.tracker = tensorboard.SummaryWriter(self.logging_dir)
    
    def log(self, metrics: Dict[str, Union[int, float, str, Dict[str, Union[int, float]]]], global_step: int):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.tracker.add_scalar(k, v, global_step=global_step)
            elif isinstance(v, str):
                self.tracker.add_text(k, v, global_step=global_step)
            elif isinstance(v, dict):
                self.tracker.add_scalars(k, v, global_step=global_step)
        self.tracker.flush()


class ImageLoggerMixin(TensorboardLoggerMixin):
    def log_images(self, images, prompts, global_step):
        result = {}
        for image, prompt in zip(images, prompts):
            result[f"{prompt}"] = image.numpy()[None, ...]

        self.tracker.log_images(
            result,
            step=global_step,
        )