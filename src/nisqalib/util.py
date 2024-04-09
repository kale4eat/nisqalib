import os

from .type_def import PretrainedModel


def get_pretrained_model_weight_path(pretrained_model: PretrainedModel) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "nisqa",
        "weights",
        pretrained_model + ".tar",
    )
