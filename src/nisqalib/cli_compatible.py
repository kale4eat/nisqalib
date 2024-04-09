import os
from typing import Optional, Union

import pandas as pd

from .nisqa.nisqa import NISQA_model
from .type_def import PretrainedModel
from .util import get_pretrained_model_weight_path


def predict_file(
    pretrained_model: PretrainedModel,
    deg: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
) -> pd.DataFrame:
    """Prediction for a single audio file

    Parameters
    ----------
    pretrained_model : PretrainedModel
        Pretrained model
    deg : Union[str, os.PathLike]
        Input file path
    output_dir : Union[str, os.PathLike]
        Output directory

    Returns
    -------
    pd.DataFrame
        Results of the prediction
    """
    args = {}
    args["mode"] = "predict_file"
    args["pretrained_model"] = get_pretrained_model_weight_path(pretrained_model)

    if not os.path.isfile(deg):
        raise Exception(f"Input file {deg} does not exist")
    args["deg"] = deg
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise Exception(f"Output directory {output_dir} does not exist")

    args["output_dir"] = output_dir
    args["ms_channel"] = None
    nisqa = NISQA_model.nisqaModel(args)
    return nisqa.predict()


def predict_dir(
    pretrained_model: PretrainedModel,
    data_dir: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    num_workers: int = 0,
    bs: int = 10,
    progress_bar=False,
) -> pd.DataFrame:
    """Predict audio files in a directory

    Parameters
    ----------
    pretrained_model : PretrainedModel
        _Pretrained model
    data_dir : Union[str, os.PathLike]
        Directory containing audio files
    output_dir : Union[str, os.PathLike]
        Output directory
    num_workers : int, optional
        Number of workers for Pytorch Dataloader, by default 0
    bs : int, optional
        Batch size for Pytorch Dataloader, by default 10
    progress_bar : bool, optional
        Enable tqdm progress bar, by default False

    Returns
    -------
    pd.DataFrame
        Results of the prediction
    """
    args = {}
    args["mode"] = "predict_dir"
    args["pretrained_model"] = get_pretrained_model_weight_path(pretrained_model)

    if not os.path.isdir(data_dir):
        raise Exception(f"Directory containing audio files {data_dir} does not exist")
    args["data_dir"] = data_dir
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise Exception(f"Output directory {output_dir} does not exist")
    args["output_dir"] = output_dir
    args["tr_bs_val"] = bs
    args["tr_num_workers"] = num_workers
    args["ms_channel"] = None
    nisqa = NISQA_model.nisqaModel(args)
    return nisqa.predict(progress_bar)
