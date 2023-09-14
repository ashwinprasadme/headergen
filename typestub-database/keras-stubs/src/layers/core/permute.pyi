from typing import Any

from keras.src.engine.base_layer import Layer as Layer
from keras.src.engine.input_spec import InputSpec as InputSpec

class Permute(Layer):
    dims: Any
    input_spec: Any
    def __init__(self, dims, **kwargs) -> None: ...
    def compute_output_shape(self, input_shape): ...
    def call(self, inputs): ...
    def get_config(self): ...