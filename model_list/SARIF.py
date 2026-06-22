from .SegmentAnything_ import sam_model_registry
from .SegmentAnything_ import SamPredictor, build_sam
from .SegmentAnything_.modeling import Sam
from model_list import pos_weight_calculator
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from .SegmentAnything_.utils.transforms import ResizeLongestSide
from icecream import ic
from typing import Any, Optional, Tuple, Type
from copy import deepcopy
def check_nan(tensor, tensor_name=""):
    if not torch.is_tensor(tensor):
        return  # Skip non-tensor
    if torch.isnan(tensor).any():
        # 현재 함수 호출 위치 (stack trace)
        frame = inspect.currentframe().f_back
        file_name = frame.f_code.co_filename
        line_no = frame.f_lineno
        func_name = frame.f_code.co_name

        print(f"\n🚨 NaN detected in tensor '{tensor_name}'")
        print(f"   → Location: File '{file_name}', function '{func_name}', line {line_no}")
        print(f"   → Shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"   → Values: {tensor}")

        # 프로그램 중단
        sys.exit(f"Aborted due to NaN in '{tensor_name}' at line {line_no}")

# Customized Segment Anything Model for Medical Image Segmentation
class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_k: nn.Module,
            linear_b_k: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self._lora_enabled = True

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        if self._lora_enabled:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_k = self.linear_b_k(self.linear_a_k(x))
            new_v = self.linear_b_v(self.linear_a_v(x))
            qkv[:, :, :, : self.dim] += new_q
            qkv[:, :, :, self.dim : 2 * self.dim] += new_k
            qkv[:, :, :, -self.dim:] += new_v
        return qkv

    def set_lora_enabled(self, enabled: bool) -> None:
        self._lora_enabled = enabled

class LoRA_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.lora_modules = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_k)
            self.w_Bs.append(w_b_linear_k)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
            )
            self.lora_modules.append(blk.attn.qkv)
        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def enable_lora(self, enabled: bool = True) -> None:
        for module in self.lora_modules:
            module.set_lora_enabled(enabled)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)

def load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes,verbose=True):
    new_state_dict = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k.startswith("image_encoder."):
            new_k = k[len("image_encoder."):]  # prefix 제거
            new_state_dict[new_k] = v
    token_size = int(image_size // vit_patch_size)  # 예: 512//16 = 32
    pe_key = "pos_embed"
    if pe_key in new_state_dict:
        pe = new_state_dict[pe_key]  # [1, Htok, Wtok, C] 가정
        if pe.dim() == 4 and pe.shape[1] != token_size:
            # if verbose:
            #     print(f"[pos_embed] resize: {tuple(pe.shape)} -> (1,{token_size},{token_size},{pe.shape[-1]})")
            pe = pe.permute(0, 3, 1, 2)                             # [1,C,Htok,Wtok]
            pe = F.interpolate(pe, size=(token_size, token_size),   # bilinear 보간
                               mode="bilinear", align_corners=False)
            pe = pe.permute(0, 2, 3, 1)                             # [1,Htok,Wtok,C]
            new_state_dict[pe_key] = pe

    # 3) rel_pos 크기 맞추기 (글로벌 어텐션 블록만)
    to_resize_keys = []
    for k in list(new_state_dict.keys()):
        if "rel_pos" in k:
            # blocks.{i}.attn.rel_pos_*
            parts = k.split(".")
            try:
                blk_idx = int(parts[1])
            except Exception:
                blk_idx = None
            if blk_idx is not None and blk_idx in encoder_global_attn_indexes:
                to_resize_keys.append(k)

    # if to_resize_keys and verbose:
    #     print(f"[rel_pos] resize targets ({len(to_resize_keys)})")

    for k in to_resize_keys:
        rp = new_state_dict[k]  # [L, D] 가정
        if rp.dim() == 2:
            L, D = rp.shape
            target_L = token_size * 2 - 1
            if L != target_L:
                # if verbose:
                #     print(f"[rel_pos] {k}: resize {L} -> {target_L}")
                rp_ = rp.unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
                rp_ = F.interpolate(rp_, size=(target_L, D), mode="bilinear", align_corners=False)
                new_state_dict[k] = rp_.squeeze(0).squeeze(0)

    return new_state_dict

class Sqzblock(nn.Module):
    def __init__(self,in_ch=1280, out_ch=32):
        super().__init__()
        self.block = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self,x):
        x = self.block(x)
        return x

class Diffblock(nn.Module):
    def __init__(self,in_ch=512,mid = 64, out_ch=512,kernel_size =3, padding=1,dilation_rate =1):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool= nn.AdaptiveMaxPool2d((1, 1))

            self.max_diff_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_ch, kernel_size=1)
            )

            self.agv_diff_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_ch, kernel_size=1)
            )

            self.block = nn.Sequential(
                nn.Conv2d(in_ch+1, mid, kernel_size=1),
                nn.BatchNorm2d(mid),
                nn.ReLU(),
                nn.Conv2d(mid, mid, kernel_size=kernel_size,dilation=dilation_rate, padding=padding),
                nn.BatchNorm2d(mid),
                nn.ReLU(),
                nn.Conv2d(mid, out_ch//2, kernel_size=1)
            )

            self.sigmoid = nn.Sigmoid()

    def forward(self,ori_block,fine_block):
            c = ori_block.size(1)
            cos = F.cosine_similarity(ori_block,fine_block,dim=1, eps=1e-8)
            cos = cos.unsqueeze(1)

            avg_ori_block = self.avgpool(ori_block).view(-1,c,1,1)
            max_ori_block = self.maxpool(ori_block).view(-1,c,1,1)

            avg_fine_block = self.avgpool(fine_block).view(-1,c,1,1)
            max_fine_block = self.maxpool(fine_block).view(-1,c,1,1)

            avg_diff = torch.cat([avg_ori_block, avg_fine_block],dim=1)
            max_diff = torch.cat([max_ori_block, max_fine_block],dim=1)

            avg_diff = self.sigmoid(self.agv_diff_conv(avg_diff))
            max_diff = self.sigmoid(self.max_diff_conv(max_diff))

            global_diff = (avg_diff+max_diff)

            feats = torch.cat([ori_block, fine_block],dim=1)
            feats = feats*global_diff

            feats = torch.cat([cos,feats],dim=1)

            diff_feats = self.block(feats)
            diff_feats = diff_feats+fine_block


            return diff_feats

class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()

        self.depth = 24
        self.sam_type = args.sam_type
        self.sam_finetunning = sam_model_registry[self.sam_type](
            checkpoint=args.sam_checkpoint,
            image_size=args.image_size,
            args=args,
        )
        self.lora_sam = LoRA_Sam(self.sam_finetunning,r = args.lora_layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool= nn.AdaptiveMaxPool2d((1, 1))

        self.ori_sqz_blocks = nn.ModuleList()
        self.fine_sqz_blocks = nn.ModuleList()
        self.encoder_diff_blocks = nn.ModuleList()
        self.prompt_layer = nn.ModuleList()
        self.sigmoid=  nn.Sigmoid()
        self.enc_indices =  [5, 11, 17, 23]
        self.kernel_size = [3,3,3,3,3]
        self.dilation_rate = [4,3,2,1,1]
        self.padding = [4,3,2,1,1]

        for i in range(5):
            prompt_layer = Sqzblock(in_ch=512,out_ch=256)
            self.prompt_layer.append(prompt_layer)

        for i in range(4):
            ori_block = Sqzblock(in_ch=1024, out_ch=256)
            fine_block = Sqzblock(in_ch= 1024,out_ch=256)
            diff_block = Diffblock(in_ch=512, mid=128, out_ch=512,kernel_size=self.kernel_size[i],dilation_rate=self.dilation_rate[i],padding=self.padding[i])
            self.ori_sqz_blocks.append(ori_block)
            self.fine_sqz_blocks.append(fine_block)
            self.encoder_diff_blocks.append(diff_block)
        

        self.embed_diff_block = Diffblock(in_ch=512, mid=128, out_ch=512, kernel_size=self.kernel_size[-1],dilation_rate=self.dilation_rate[-1],padding=self.padding[-1])
        self.ori_sqz = nn.Conv2d(256,256,1)
        self.fine_sqz = nn.Conv2d(256,256,1)

        self.sigmoid = nn.Sigmoid()

    def _L2normalization(self,x):
        # 방법 2: F.normalize 사용
        x_normalized = F.normalize(x, p=2, dim=0)  # 채널 방향으로 normalize
        return x_normalized

    def forward(self, data):
        x = data['image']
        _, _, H, W = x.shape
        # image encoder에서 embedding 생성
        self.lora_sam.enable_lora(False)
        with torch.no_grad():
            original_image_embeddings,  memory_bank = self.lora_sam.sam.image_encoder(x)
        self.lora_sam.enable_lora(True)

        finetunning_image_embeddings, finetunning_image_embedding_lists  = self.lora_sam.sam.image_encoder(x)
        # print((finetunning_image_embedding_lists.shape)) # B, 32, 1280,32,32
        B,N,C,_,_ = finetunning_image_embedding_lists.shape

        enc_diff_feats_list = []

        for (idx,ori_sqz_block, fine_sqz_block,diff_block) in zip(self.enc_indices, self.ori_sqz_blocks, self.fine_sqz_blocks, self.encoder_diff_blocks):
            ori_feat = memory_bank[:,idx,:,:,:] # B,1280,32,32
            fine_feat = finetunning_image_embedding_lists[:,idx,:,:,:] # B,1280,32,32

            ori_feat = ori_sqz_block(ori_feat)
            fine_feat = fine_sqz_block(fine_feat)

            diff_feat = diff_block(ori_feat,fine_feat)
            enc_diff_feats_list.append(diff_feat)

        ori_embed = self.ori_sqz(original_image_embeddings)
        fine_embed = self.fine_sqz(finetunning_image_embeddings)

        embed_diff = self.embed_diff_block(ori_embed,fine_embed)
        enc_diff_feats_list.append(embed_diff)

        enc_diff_feats_list = torch.stack(enc_diff_feats_list,dim=1) # 4,8,256,32,32

        pred_masks = []
        final_pred_masks = []
        
        for (finetunning_image_embedding,enc_diff_feat) in zip(finetunning_image_embeddings,enc_diff_feats_list):
            prompt_masks = []
            pred_masks = []
            sparse_embeddings, dense_embeddings = self.lora_sam.sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
            )
            prediction_mask,_ = self.lora_sam.sam.mask_decoder(
                        image_embeddings=finetunning_image_embedding.unsqueeze(0),
                        image_pe=self.lora_sam.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings= sparse_embeddings,
                        dense_prompt_embeddings =dense_embeddings,
                        multimask_output=False,
            )

            prompt_mask = self.sigmoid(prediction_mask)
            prompt_mask = (prompt_mask > 0.5).float()
            
            prediction_mask = F.interpolate(prediction_mask,size=(H, W),mode="bilinear",align_corners=False)
            pred_masks.append(prediction_mask)

            for idx, (diff_feat,prompt_layer) in enumerate(zip(enc_diff_feat,self.prompt_layer)):
                masks = None if prompt_mask is None else prompt_mask 

                sparse_embeddings, dense_embeddings = self.lora_sam.sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=masks,
                )
                prompt = torch.cat([diff_feat.unsqueeze(0),dense_embeddings],dim=1)
               
                prompt = prompt_layer(prompt)
                prediction_mask,_ = self.lora_sam.sam.mask_decoder(
                        image_embeddings=finetunning_image_embedding.unsqueeze(0),
                        image_pe=self.lora_sam.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings= sparse_embeddings,
                        dense_prompt_embeddings =prompt,
                        multimask_output=False,
                )

                prompt_mask = self.sigmoid(prediction_mask)
                prompt_mask = (prompt_mask > 0.5).float()

                prediction_mask = F.interpolate(prediction_mask,size=(H, W),mode="bilinear",align_corners=False)
                pred_masks.append(prediction_mask)

            pred_masks = torch.cat(pred_masks,dim=1)
            final_pred_masks.append(pred_masks)

        final_pred_masks = torch.cat(final_pred_masks, dim=0) # 4,8,512,512
        # print(final_pred_masks.shape)

        output_dict = {'prediction' : final_pred_masks[:,-1,:,:], 'predictions' : final_pred_masks,'original_image_embeddings' :original_image_embeddings ,'finetunning_image_embeddings' : finetunning_image_embeddings, 'original_enc_list' : memory_bank, 'finetunning_enc_list' : finetunning_image_embedding_lists,'enc_difference_list' : enc_diff_feats_list}

        output_dict = self._calculate_criterion(output_dict,data)

        return output_dict

    def get_predictor(self):
        return SamPredictor(self.model)
    def l2n(self,x, dim=1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    def _calculate_criterion(self,output_dict,data):
        pos_weight = pos_weight_calculator(data['target'])
        logits = output_dict['predictions']          
        target = data['target'] 
        target = target.expand(-1, logits.size(1), -1, -1)

        pred_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)  

        output_dict['loss'] = pred_loss 
        output_dict['pos_weight'] = pos_weight.detach()
        return output_dict

def _training_config(args):
    # Match the M2SFormer experiment configuration; keep only SARIF-specific
    # model arguments below.
    args.region_loss = 'DICE_FOCAL'
    args.skip_channels = 64
    args.cnn_backbone = None
    args.transformer_backbone = 'pvt_v2_b2'
    args.pretrained = True

    args.target_resolution = 32
    args.scale_branches = 2
    args.min_channel = 64
    args.min_resolution = 8
    args.frequency_branches = 16
    args.frequency_selection = 'top'
    args.reduction = 16
    args.num_heads = 1
    args.text_embedding_length = 300

    args.num_channels = 3
    args.num_classes = 1
    args.image_size = 256
    args.metric_list = ['DSC', 'IoU', 'E-Measure', 'AUC']
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]

    args.multi_scale_train = False
    args.train_batch_size = 32
    args.test_batch_size = 50
    args.final_epoch = 100

    args.optimizer_name = 'AdamW'
    args.lr = 1e-4
    args.momentum = 0.9
    args.nesterov = False
    args.weight_decay = 1e-4
    args.adjust_learning_rate = adjust_learning_rate

    args.freeze_image_encoder = True
    args.freeze_prompt_encoder = False
    args.freeze_mask_decoder = False

    args.lora_layer = 32
    args.sam_type = 'vit_l'
    args.sam_checkpoint = getattr(args, 'sam_checkpoint', 'sam_vit_L.pth')

    return args

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def adjust_learning_rate(optimizer, epochs, train_loader_len, learning_rate):
    total_steps = max(1, epochs * train_loader_len)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            min(step, total_steps),
            total_steps,
            1,  # lr_lambda computes multiplicative factor
            1e-6 / learning_rate))

    return scheduler

# if __name__ == "__main__":
#     inp = torch.randn((4, 3, 1024, 1024))
#     model = Model()
#     oup = model(inp)
#     print(oup.shape)
