import numpy as np
import tensorflow as tf

from src.efficientnet.keras import EfficientNetB0, EfficiUNetB0
from typing import Optional


def load_model(path: str, out_channels: int, out_activation: Optional[str] = None) -> EfficiUNetB0:
    baseModel = EfficientNetB0(weights=None, include_top=False, input_shape=(None,None,3))
    model = EfficiUNetB0(baseModel, seg_maps=out_channels, seg_act=out_activation)
    model.load_weights(path)
    return model


def transform_image(image: np.ndarray) -> tf.Tensor:
    transformed_image = tf.convert_to_tensor(image / 255., dtype=float) 
    transformed_image -= tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=float)
    transformed_image /=  tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=float)
    return tf.expand_dims(transformed_image, axis=0)
