<<<<<<< Updated upstream
from .clip_model import CLIP_MODEL
from .clip_multi import CLIP_MULTI_MODEL
from .temperature_scaling import ModelWithTemperature

__all__ = [CLIP_MODEL, CLIP_MULTI_MODEL, ModelWithTemperature]
=======
from .clip import CLIP_MODEL
from .temperature_scaling import ModelWithTemperature

__all__ = [CLIP_MODEL, ModelWithTemperature]
>>>>>>> Stashed changes
