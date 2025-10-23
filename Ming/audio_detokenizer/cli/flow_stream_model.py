# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import os

from audio_detokenizer.utils.file_utils import convert_onnx_to_trt
from audio_detokenizer.utils.common import TrtContextWrapper

class AudioDetokenizerModel:
    def __init__(self,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 50
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

    def load(self, flow_model, hift_model):
        if flow_model.endswith("flow.pt") or flow_model.endswith("cache.pt"):
            self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=False), strict=True)
        else:
            self.flow.load_state_dict(torch.load(flow_model, map_location=self.device, weights_only=False)["model"], strict=True)
        
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
    
    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent=1, fp16=True):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        
        def get_trt_kwargs():
            min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
            opt_shape = [(2, 80, 512), (2, 1, 512), (2, 80, 512), (2, 80, 512)]
            max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
            input_names = ["x", "mask", "mu", "cond"]
            return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}


        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, get_trt_kwargs(), flow_decoder_onnx_model, fp16)

        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)
        del self.flow.decoder.estimator