from typing import Any

from keras import backend as backend
from keras.src.engine import base_preprocessing_layer as base_preprocessing_layer

class Normalization(base_preprocessing_layer.PreprocessingLayer):
    axis: Any
    input_mean: Any
    input_variance: Any
    def __init__(
        self,
        axis: int = ...,
        mean: Any | None = ...,
        variance: Any | None = ...,
        **kwargs
    ) -> None: ...
    adapt_mean: Any
    adapt_variance: Any
    count: Any
    mean: Any
    variance: Any
    def build(self, input_shape) -> None: ...
    def adapt(
        self, data, batch_size: Any | None = ..., steps: Any | None = ...
    ) -> None: ...
    def update_state(self, data) -> None: ...
    def reset_state(self) -> None: ...
    def finalize_state(self) -> None: ...
    def call(self, inputs): ...
    def compute_output_shape(self, input_shape): ...
    def compute_output_signature(self, input_spec): ...
    def get_config(self): ...