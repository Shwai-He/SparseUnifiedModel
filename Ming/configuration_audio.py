# coding=utf-8
# Copyright 2022 shunxing1234 and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Audio configuration """

from transformers.configuration_utils import PretrainedConfig


class GLMAudioConfig(PretrainedConfig):
    model_type = "bailingmm"

    def __init__(
        self,
        audio_encoder_output_size=512,
        audio_decoder_type=None,  # None, "glmaudio", "glmv4audio"
        audio_id_shift=None,
        vocab_size_audio=0,  # audio vocab starts from audio_id_shift and ends at audio_id_shift + vocab_size_audio exclusively.
        use_audio_bpe_token=False,
        ds_conv_type="conv",  # "conv" or "dsconv"
        ds_kernel_size=1,
        ds_stride=1,
        norm_query_embeds=True,
        audio_wav_frontend_config_sanm={},  # SANMEncoder's WavFrontend related configs.
        audio_encoder_config_sanm={},  # SANMEncoder related configs.
        **kwargs
    ):
        # Audio related.
        self.audio_encoder_output_size = audio_encoder_output_size
        self.audio_decoder_type = audio_decoder_type
        self.audio_id_shift = audio_id_shift
        self.vocab_size_audio = vocab_size_audio
        self.use_audio_bpe_token = use_audio_bpe_token

        # Audio feature downsampler related.
        self.ds_conv_type = ds_conv_type
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride
        self.norm_query_embeds = norm_query_embeds

        # Third-party module configs.
        self.audio_wav_frontend_config_sanm = audio_wav_frontend_config_sanm
        self.audio_encoder_config_sanm = audio_encoder_config_sanm

