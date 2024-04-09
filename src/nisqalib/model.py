from typing import Literal

import librosa as lb
import numpy as np
import torch
import torchaudio

from .nisqa.nisqa import NISQA_lib as NL
from .type_def import PretrainedModel
from .util import get_pretrained_model_weight_path


class NisqaModel:
    def __init__(
        self,
        model: PretrainedModel,
        device: Literal["auto", "cpu", "cuda"] = "auto",
    ) -> None:
        """Prediction for waveform input

        Parameters
        ----------
        pretrained_model : PretrainedModel
            Pretrained model
        device : Literal["auto", "cpu", "cuda"]
            Device to use for computation

        """

        args = {}
        args["mode"] = "predict_file"
        args["ms_channel"] = None

        model_path = get_pretrained_model_weight_path(model)
        args["pretrained_model"] = model_path
        if device == "auto" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        checkpoint["args"].update(args)
        self.args = checkpoint["args"]

        if self.args["model"] == "NISQA_DIM":
            self.args["dim"] = True
            self.args["csv_mos_train"] = None  # column names hardcoded for dim models
            self.args["csv_mos_val"] = None
        else:
            self.args["dim"] = False

        if self.args["model"] == "NISQA_DE":
            self.args["double_ended"] = True
        else:
            self.args["double_ended"] = False
            self.args["csv_ref"] = None

        # Load Model
        self.model_args = {
            "ms_seg_length": self.args["ms_seg_length"],
            "ms_n_mels": self.args["ms_n_mels"],
            "cnn_model": self.args["cnn_model"],
            "cnn_c_out_1": self.args["cnn_c_out_1"],
            "cnn_c_out_2": self.args["cnn_c_out_2"],
            "cnn_c_out_3": self.args["cnn_c_out_3"],
            "cnn_kernel_size": self.args["cnn_kernel_size"],
            "cnn_dropout": self.args["cnn_dropout"],
            "cnn_pool_1": self.args["cnn_pool_1"],
            "cnn_pool_2": self.args["cnn_pool_2"],
            "cnn_pool_3": self.args["cnn_pool_3"],
            "cnn_fc_out_h": self.args["cnn_fc_out_h"],
            "td": self.args["td"],
            "td_sa_d_model": self.args["td_sa_d_model"],
            "td_sa_nhead": self.args["td_sa_nhead"],
            "td_sa_pos_enc": self.args["td_sa_pos_enc"],
            "td_sa_num_layers": self.args["td_sa_num_layers"],
            "td_sa_h": self.args["td_sa_h"],
            "td_sa_dropout": self.args["td_sa_dropout"],
            "td_lstm_h": self.args["td_lstm_h"],
            "td_lstm_num_layers": self.args["td_lstm_num_layers"],
            "td_lstm_dropout": self.args["td_lstm_dropout"],
            "td_lstm_bidirectional": self.args["td_lstm_bidirectional"],
            "td_2": self.args["td_2"],
            "td_2_sa_d_model": self.args["td_2_sa_d_model"],
            "td_2_sa_nhead": self.args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": self.args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": self.args["td_2_sa_num_layers"],
            "td_2_sa_h": self.args["td_2_sa_h"],
            "td_2_sa_dropout": self.args["td_2_sa_dropout"],
            "td_2_lstm_h": self.args["td_2_lstm_h"],
            "td_2_lstm_num_layers": self.args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": self.args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": self.args["td_2_lstm_bidirectional"],
            "pool": self.args["pool"],
            "pool_att_h": self.args["pool_att_h"],
            "pool_att_dropout": self.args["pool_att_dropout"],
        }

        if self.args["double_ended"]:
            self.model_args.update(
                {
                    "de_align": self.args["de_align"],
                    "de_align_apply": self.args["de_align_apply"],
                    "de_fuse_dim": self.args["de_fuse_dim"],
                    "de_fuse": self.args["de_fuse"],
                }
            )

        print("Model architecture: " + self.args["model"])
        if self.args["model"] == "NISQA":
            self.model = NL.NISQA(**self.model_args)
        elif self.args["model"] == "NISQA_DIM":
            self.model = NL.NISQA_DIM(**self.model_args)
        elif self.args["model"] == "NISQA_DE":
            self.model = NL.NISQA_DE(**self.model_args)
        else:
            raise NotImplementedError("Model not available")

        # Load weights if pretrained model is used ------------------------------------
        if self.args["pretrained_model"]:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
            print("Loaded pretrained model from " + self.args["pretrained_model"])
            if missing_keys:
                print("missing_keys:")
                print(missing_keys)
            if unexpected_keys:
                print("unexpected_keys:")
                print(unexpected_keys)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, float]:
        """Prediction for waveform input

        Parameters
        ----------
        waveform : torch.Tensor
            Input waveform
        sample_rate : int
            Sample rate of input waveform

        Returns
        -------
        dict[str, float]
            Results of the prediction
        """

        xb, yb, n_wins = self._extract_feature(waveform, sample_rate)
        with torch.no_grad():
            pred = (
                self.model(
                    xb.unsqueeze(0).to(self.device),
                    n_wins.unsqueeze(0).to(self.device),
                )
                .cpu()
                .numpy()
            )

        y_hat = pred

        ret = {}
        ret["mos_pred"] = y_hat[0, 0]
        ret["noi_pred"] = y_hat[0, 1]
        ret["dis_pred"] = y_hat[0, 2]
        ret["col_pred"] = y_hat[0, 3]
        ret["loud_pred"] = y_hat[0, 4]

        return ret

    def _extract_feature(self, waveform, sample_rate):
        if self.args["ms_sr"]:
            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.args["ms_sr"]
            )

            waveform = transform(waveform)

        spec = get_librosa_melspec(
            waveform,
            sr=sample_rate if self.args["ms_sr"] is None else self.args["ms_sr"],
            n_fft=self.args["ms_n_fft"],
            hop_length=self.args["ms_hop_length"],
            win_length=self.args["ms_win_length"],
            n_mels=self.args["ms_n_mels"],
            fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
        )

        if self.args["ms_seg_length"] is not None:
            x_spec_seg, n_wins = segment_specs(
                spec,
                self.args["ms_seg_length"],
                self.args["ms_seg_hop_length"],
                self.args["ms_max_segments"],
            )

        else:
            x_spec_seg = spec
            n_wins = spec.shape[1]
            if self.args["ms_max_segments"] is not None:
                x_padded = np.zeros((x_spec_seg.shape[0], self.args["ms_max_segments"]))
                x_padded[:, :n_wins] = x_spec_seg
                x_spec_seg = np.expand_dims(x_padded.transpose(1, 0), axis=(1, 3))
                if not torch.is_tensor(x_spec_seg):
                    x_spec_seg = torch.tensor(x_spec_seg, dtype=torch.float)

        # Get MOS (apply NaN in case of prediction only mode)
        if self.args["dim"]:
            # predict_only
            y = np.full((5, 1), np.nan).reshape(-1).astype("float32")
        else:
            y = np.full(1, np.nan).reshape(-1).astype("float32")

        return x_spec_seg, torch.tensor(y), torch.tensor(n_wins)


def get_librosa_melspec(
    waveform: torch.Tensor,
    sr=48000,
    n_fft=1024,
    hop_length=80,
    win_length=170,
    n_mels=32,
    fmax=16e3,
    ms_channel=None,
):
    """
    Calculate mel-spectrograms with Librosa.
    """
    # Calc spec
    if ms_channel is not None:
        if len(waveform.shape) > 1:
            y = waveform[ms_channel].numpy()
    else:
        if waveform.shape[0] > 1:
            y = waveform.mean(dim=0, keepdim=True)[0].numpy()
        else:
            y = waveform[0].numpy()

    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm="slaney",
    )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec


def segment_specs(x, seg_length, seg_hop=1, max_length=None):
    if seg_length % 2 == 0:
        raise ValueError("seg_length must be odd! (seg_lenth={})".format(seg_length))
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    n_wins = x.shape[1] - (seg_length - 1)

    # broadcast magic to segment melspec
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(1, 0)[idx3, :].unsqueeze(1).transpose(3, 2)

    if seg_hop > 1:
        x = x[::seg_hop, :]
        n_wins = int(np.ceil(n_wins / seg_hop))

    if max_length is not None:
        if max_length < n_wins:
            raise ValueError(
                "n_wins {} > max_length {}. Increase max window length ms_max_segments!".format(
                    n_wins,
                    max_length,
                )
            )
        x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
        x_padded[:n_wins, :] = x
        x = x_padded

    return x, np.array(n_wins)
