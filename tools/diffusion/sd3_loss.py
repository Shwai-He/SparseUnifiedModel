import torch
from diffusers import AutoencoderKL
import os
from collections import OrderedDict
import logging
from diffusers import FlowMatchEulerDiscreteScheduler
import torch.nn as nn
from .sd3_transformer import SD3Transformer2DModel
from .pipeline_stable_diffusion_3 import StableDiffusion3Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToClipMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm1 = nn.LayerNorm(2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

class SD3Model_withMLP(nn.Module):
    def __init__(self, transformer, vision_dim=1152):
        super().__init__()
        self.transformer = transformer
        self.dtype = self.transformer.dtype
        self.mlp = ToClipMLP(vision_dim, 4096)
        self.config = self.transformer.config
    
    def forward(self, hidden_states,
                    timestep,
                    encoder_hidden_states,
                    return_dict,
                    encoder_attention_mask=None,
                    extra_vit_input=None,
                     **kargs):

        encoder_hidden_states = self.mlp(encoder_hidden_states.to(self.mlp.fc1.weight.dtype)).to(self.dtype)
         
        pooled_projections = torch.nn.functional.adaptive_avg_pool1d(encoder_hidden_states.permute(0, 2, 1), output_size=1)
        pooled_projections = torch.nn.functional.adaptive_avg_pool1d(pooled_projections.squeeze(2), output_size=2048)

        if extra_vit_input is not None:
            encoder_hidden_states = torch.cat((encoder_hidden_states, extra_vit_input), dim=1)
            
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            hidden_states = self.transformer(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled_projections,
                        timestep=timestep,
                        return_dict=False,
                        **kargs
                    )
                
        return hidden_states

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

def load_denoising_pretrained_weights(
        net, 
        weights_path,
        names=None,
        prefix_to_remove=None,    
    ):
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "net" in state_dict:
        state_dict = state_dict["net"]

    if names is not None:
        selected_state_dict = OrderedDict()
        for ori_name in names:
            name = ori_name[len(prefix_to_remove):] if prefix_to_remove is not None else ori_name
            selected_state_dict[name] = state_dict[ori_name]
    
        state_dict = selected_state_dict

    net.load_state_dict(state_dict, strict=True)
    print(f'Loaded weights from {weights_path}. num param {len(state_dict)}')
    return 


class SD3Loss(torch.nn.Module):
    def __init__(self, 
            model_path, 
            scheduler_path,
            vision_dim=3584, 
            checkpoint_path=None, 
            mlp_state_dict=None,
            trainable_params='none_param', 
            device='cpu', 
            revision=None, 
            variant=None, 
            embed_dim=32,
            ref_add_noise=False,
            ref_add_noise_ratio=0.25,
            torch_dtype=torch.float32,
        ):
        super(SD3Loss, self).__init__()
        #self.torch_type = torch.bfloat16
        self.base_model_path = model_path
        
        self.ref_add_noise = ref_add_noise
        self.ref_add_noise_ratio = ref_add_noise_ratio

        self.device = torch.device(torch.cuda.current_device())    

        self.scheduler_path = scheduler_path
        # print(f"scheduler_path: {scheduler_path}")
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        
        # self.vae.to(self.torch_type).to(self.device)
        self.vae.requires_grad_(False)

        self.train_model = SD3Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", revision=revision, variant=variant,
            torch_dtype=torch_dtype,
        )

        if checkpoint_path is not None:
            assert os.path.exists(checkpoint_path)
            load_denoising_pretrained_weights(self.train_model, checkpoint_path)

        self.train_model = SD3Model_withMLP(self.train_model, vision_dim=vision_dim)
        assert mlp_state_dict is not None
        self.train_model.mlp.load_state_dict(mlp_state_dict, strict=True)

        self.train_model.enable_gradient_checkpointing()
        
        self.set_trainable_params(trainable_params)


        num_parameters_trainable = 0
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()
        logger.info(f"number of all Diffusion parameters: {num_parameters}, trainable: {num_parameters_trainable}")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.scheduler_path, subfolder="scheduler")
        self.pipelines = StableDiffusion3Pipeline(
            vae=self.vae,
            transformer=self.train_model,  
            text_encoder=None, 
            tokenizer=None,
            text_encoder_2=None, 
            tokenizer_2=None, 
            text_encoder_3=None, 
            tokenizer_3=None,  
            scheduler=self.noise_scheduler,
        ).to(self.device)

        #self._compile_pipeline()

    def _compile_pipeline(self):
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        #self.pipelines.set_progress_bar_config(disable=True)
        
        self.pipelines.vae.to(memory_format=torch.channels_last)
        self.pipelines.vae.decode = torch.compile(self.pipelines.vae.decode, mode="max-autotune", fullgraph=True)
        self.pipelines.transformer.to(memory_format=torch.channels_last)
        self.pipelines.transformer = torch.compile(self.pipelines.transformer, mode="max-autotune", fullgraph=True)
        
    def set_trainable_params(self, trainable_params):
        
        self.vae.requires_grad_(False)

        if trainable_params == 'all':
            self.train_model.requires_grad_(True)
        else:
            self.train_model.requires_grad_(False)
            for name, module in self.train_model.named_modules():
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        for params in module.parameters():
                            params.requires_grad = True

        num_parameters_trainable = 0
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()
        logger.info(f"number of all Diffusion parameters: {num_parameters}, trainable: {num_parameters_trainable}")

    def sample(
        self, encoder_hidden_states, steps=20, cfg=7.0, image_cfg=1.0, cfg_mode=1, seed=42, height=512, width=512, 
        negative_encoder_hidden_states=None, extra_vit_input=None, ref_x=None):
        
        pooled_projections = torch.nn.functional.adaptive_avg_pool1d(encoder_hidden_states.permute(0, 2, 1), output_size=1)
        pooled_projections = torch.nn.functional.adaptive_avg_pool1d(pooled_projections.squeeze(2), output_size=2048)
        image = self.pipelines(
            prompt_embeds=encoder_hidden_states,
            negative_prompt_embeds=negative_encoder_hidden_states if negative_encoder_hidden_states is not None else encoder_hidden_states * 0,
            pooled_prompt_embeds=pooled_projections,
            negative_pooled_prompt_embeds=pooled_projections*0,
            guidance_scale=cfg,
            image_guidance_scale=image_cfg,
            guidance_scale_mode=cfg_mode,
            generator=torch.manual_seed(seed),
            num_inference_steps=steps,
            height=height,
            width=width,
            max_sequence_length=512,
            device=self.device,
            extra_vit_input=extra_vit_input,
            ref_x=ref_x,
            ref_add_noise=self.ref_add_noise,
            ref_add_noise_ratio=self.ref_add_noise_ratio,
        ).images[0]

        return image  
