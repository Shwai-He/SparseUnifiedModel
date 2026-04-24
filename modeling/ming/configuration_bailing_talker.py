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
""" GLMAudio model configuration """

from typing import Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BailingTalkerConfig(PretrainedConfig):
    # model_type = "glmaudio"
    # attribute_map = {
    #     "num_hidden_layers": "num_layers"
    # }

    def __init__(
        self,
        pretrained_model_path=None,
        qa_model_hidden_size=2048,
        vocab_size=184445,
        text_vocab_size=151677,
        audio_vocab_size=32768,
        vp_feature_size=192,
        vp_kernel_size=1,
        vp_stride=1,
        s3bpe_tokenizer=None,
        **kwargs
    ):
        self.pretrained_model_path = pretrained_model_path
        self.qa_model_hidden_size = qa_model_hidden_size
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.vp_feature_size = vp_feature_size
        self.vp_kernel_size = vp_kernel_size
        self.vp_stride = vp_stride
        self.s3bpe_tokenizer = s3bpe_tokenizer
        super().__init__(
            **kwargs
        )
