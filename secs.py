import numpy as np
import torch
from torch import nn
from resampy import resample

import torchaudio

import json

config_str = \
'''
{
    "model": "speaker_encoder",
    "run_name": "speaker_encoder",
    "run_description": "resnet speaker encoder trained with commonvoice all languages dev and train, Voxceleb 1 dev and Voxceleb 2 dev",
    "epochs": 100000,
    "batch_size": null,
    "eval_batch_size": null,
    "mixed_precision": false,
    "run_eval": true,
    "test_delay_epochs": 0,
    "print_eval": false,
    "print_step": 50,
    "tb_plot_step": 100,
    "tb_model_param_stats": false,
    "save_step": 1000,
    "checkpoint": true,
    "keep_all_best": false,
    "keep_after": 10000,
    "num_loader_workers": 8,
    "num_val_loader_workers": 0,
    "use_noise_augment": false,
    "output_path": "../checkpoints/speaker_encoder/language_balanced/normalized/angleproto-4-samples-by-speakers/",
    "distributed_backend": "nccl",
    "distributed_url": "tcp://localhost:54321",
    "audio": {
        "fft_size": 512,
        "win_length": 400,
        "hop_length": 160,
        "frame_shift_ms": null,
        "frame_length_ms": null,
        "stft_pad_mode": "reflect",
        "sample_rate": 16000,
        "resample": false,
        "preemphasis": 0.97,
        "ref_level_db": 20,
        "do_sound_norm": false,
        "do_trim_silence": false,
        "trim_db": 60,
        "power": 1.5,
        "griffin_lim_iters": 60,
        "num_mels": 64,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,
        "spec_gain": 20,
        "signal_norm": false,
        "min_level_db": -100,
        "symmetric_norm": false,
        "max_norm": 4.0,
        "clip_norm": false,
        "stats_path": null,
        "do_rms_norm": true,
        "db_level": -27.0
    },
    "datasets": [
        {
            "name": "voxceleb2",
            "path": "/workspace/scratch/ecasanova/datasets/VoxCeleb/vox2_dev_aac/",
            "meta_file_train": null,
            "ununsed_speakers": null,
            "meta_file_val": null,
            "meta_file_attn_mask": "",
            "language": "voxceleb"
        }
    ],
    "model_params": {
        "model_name": "resnet",
        "input_dim": 64,
        "use_torch_spec": true,
        "log_input": true,
        "proj_dim": 512
    },
    "audio_augmentation": {
        "p": 0.5,
        "rir": {
            "rir_path": "/workspace/store/ecasanova/ComParE/RIRS_NOISES/simulated_rirs/",
            "conv_mode": "full"
        },
        "additive": {
            "sounds_path": "/workspace/store/ecasanova/ComParE/musan/",
            "speech": {
                "min_snr_in_db": 13,
                "max_snr_in_db": 20,
                "min_num_noises": 1,
                "max_num_noises": 1
            },
            "noise": {
                "min_snr_in_db": 0,
                "max_snr_in_db": 15,
                "min_num_noises": 1,
                "max_num_noises": 1
            },
            "music": {
                "min_snr_in_db": 5,
                "max_snr_in_db": 15,
                "min_num_noises": 1,
                "max_num_noises": 1
            }
        },
        "gaussian": {
            "p": 0.0,
            "min_amplitude": 0.0,
            "max_amplitude": 1e-05
        }
    },
    "storage": {
        "sample_from_storage_p": 0.5,
        "storage_size": 40
    },
    "max_train_step": 1000000,
    "loss": "angleproto",
    "grad_clip": 3.0,
    "lr": 0.0001,
    "lr_decay": false,
    "warmup_steps": 4000,
    "wd": 1e-06,
    "steps_plot_stats": 100,
    "num_speakers_in_batch": 100,
    "num_utters_per_speaker": 4,
    "skip_speakers": true,
    "voice_len": 2.0
}
'''

class SVAudioPreprocess(nn.Module):
    def __init__(self, premp_coeff):
        super().__init__()
        self.premp_coeff = premp_coeff
        self.register_buffer("filter", torch.FloatTensor([-self.premp_coeff, 1.0]).unsqueeze(0).unsqueeze(0))

    @staticmethod
    def rms_norm(wav, db_level = -27):
        r = 10 ** (db_level / 20)
        a = torch.sqrt((len(wav) * (r**2)) / torch.sum(wav**2))
        return wav * a

    def splice(self, x):
        num_frames = 250 * 160
        max_len = x.shape[1]
        
        if max_len < num_frames:
            num_frames = max_len
        
        offsets = np.linspace(0, max_len - num_frames, num = 10)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        return frames_batch

    def forward(self, x):
        x = self.rms_norm(x).reshape(1,-1)
        x_batched = self.splice(x)
        return x_batched

class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)

class BaseEncoder(nn.Module):
    """Base `encoder` class. Every new `encoder` model must inherit this.

    It defines common `encoder` specific functions.
    """

    # pylint: disable=W0102
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    @torch.no_grad()
    def inference(self, x, l2_norm=True):
        return self.forward(x, l2_norm)

    @torch.no_grad()
    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        # map to the waveform size
        if self.use_torch_spec:
            num_frames = num_frames * self.audio_config["hop_length"]

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings

    def load_state_dict_from_dict(
        self,
        state_dict,
    ):
        self.load_state_dict(state_dict)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNetSpeakerEncoder(BaseEncoder):
    """Implementation of the model H/ASP without batch normalization in speaker embedding. This model was proposed in: https://arxiv.org/abs/2009.14153
    Adapted from: https://github.com/clovaai/voxceleb_trainer
    """

    # pylint: disable=W0102
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super(ResNetSpeakerEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = self.get_torch_mel_spectrogram_class(audio_config)
        else:
            self.torch_spec = None

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # pylint: disable=R0201
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        x.squeeze_(1)
        # if you torch spec compute it otherwise use the mel spec computed by the AP
        if self.use_torch_spec:
            x = self.torch_spec(x)

        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = json.loads(config_str)
        state = torch.load('saved_model_state_dict.pth')
        self.se_model = ResNetSpeakerEncoder(
                input_dim=64,
                proj_dim=512,
                log_input=True,
                use_torch_spec=True,
                audio_config=config["audio"])

        self.se_ap = SVAudioPreprocess(premp_coeff = 0.97)
        self.se_model.load_state_dict(state)
        self.se_model.eval()

    def forward(self, x):
        mel = self.se_ap(x)
        emb = self.se_model(mel, l2_norm=True)
        return torch.mean(emb, axis=0, keepdim = True)

if __name__ == '__main__':
    encoder = SpeakerEncoder()