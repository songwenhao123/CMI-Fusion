import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ClipCount.model.models_crossvit import CrossAttentionBlock, ConvCrossAttentionBlock, AttBlock

from ClipCount.util.pos_embed import get_2d_sincos_pos_embed, positional_encoding_1d
import clip
from torchvision import transforms
import einops
import functools
import operator
from collections import OrderedDict

import sys
sys.path.append('/home/wenhao/projects/LDFusion')

from fuse_CLIP import ImageTextFusionModule

class CLIPCount(nn.Module):
    def __init__(self, fim_depth:int=4, 
                 fim_num_heads:int=8,
                 mlp_ratio:float=4., 
                 norm_layer=nn.LayerNorm,
                 use_vpt:bool = True, 
                 vpt_width:int = 2, 
                 vpt_depth:int = 2,
                 use_coop:bool=True, 
                 coop_width:int = 2, 
                 backbone:str="b16",
                 use_fim:bool = True, 
                 use_mixed_fim:bool=False, 
                 unfreeze_vit:bool=False):
        """
        The CLIP-Count model   
        Param:
            fim_depth: the number of blocks for the patch-text interaction module, only useful for naive ViT.
            fim_num_heads: the number of heads for the patch-text interaction module.
            mlp_ratio: the ratio (mlp width)/(cross attn hidden dim) for the patch-text interaction module.
            norm_layer: the normalization layer for the patch-text interaction module.
            use_vpt: whether to use visual prompt tuning
            vpt_width: how much visual token used per layer,
            vpt_depth: how many layers used for visual prompt tuning (try allocate from the input layer first)
            use_coop: whether use coop for context learning.
            backbone: visual backbone of clip.
            use_fim: whether to use a naive transformer for patch-text interaction
            use_mixed_fim: whether to use a hierarchical transformer for patch-text interaction
            unfreeze_vit: whether to fintune all clip vit parameters.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if backbone == "b16":
            self.clip, clip_preprocess = clip.load("ViT-B/16")
            self.n_patches = 14*14
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512
        elif backbone == "b32":
            self.clip, clip_preprocess = clip.load("ViT-B/32")
            self.n_patches = 7*7
            self.clip_hidden_dim = 768
            self.clip_out_dim = 512

        elif backbone == "l14":
            self.clip, clip_preprocess = clip.load("ViT-L/14")
            self.n_patches = 16*16
            self.clip_hidden_dim = 1024
            self.clip_out_dim = 768

        self.clip = self.clip.to('cuda')
        if unfreeze_vit:
            # deal with some strange behavior of CLIP and pytorch-lightning.
            self.clip = self.clip.float() 
        self.clip.requires_grad_(False)
        self.preprocess = transforms.Compose([transforms.Resize((224,224)),
                            transforms.Normalize(
                                mean = (0.48145466, 0.4578275, 0.40821073),
                                std= (0.26862954, 0.26130258, 0.27577711)
                                ) 
                            ])

        self.use_vpt = use_vpt # what's this? it's visual prompt
        self.use_coop = use_coop # what's this? for context leraning
        self.vpt_width = vpt_width if use_vpt else 0
        self.vpt_depth = vpt_depth if use_vpt else 0
        self.coop_width = coop_width if use_coop else 0
        self.img_encoder = CLIPViT(self.clip, self.clip_hidden_dim, use_vpt=self.use_vpt, vpt_width=self.vpt_width,vpt_depth = self.vpt_depth,unfreeze=unfreeze_vit)
        self.text_encoder = CLIPTextTransformer(self.clip, use_coop=self.use_coop, n_ctx = self.coop_width)

        # --------------------------------------------------------------------------
        # Contrastive Learning related
        self.patch_feat_proj = nn.Linear(64, self.clip_out_dim, bias=True)
        self.patch_feat_proj_contrast = nn.Linear(self.clip_hidden_dim, self.clip_out_dim, bias=True)
        nn.init.xavier_normal_(self.patch_feat_proj.weight)

        n_token = self.n_patches
        # the PE for the patch embeddings \mathcal{E}_p
        self.patch_emb_pos_embed = nn.Parameter(torch.zeros(1, n_token, self.clip_out_dim), requires_grad=False)  # fixed sin-cos embedding
        decoder_pos_embed = positional_encoding_1d(self.clip_out_dim, n_token)
        self.patch_emb_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # --------------------------------------------------------------------------
        # The Hierarchical patch-text interaction module

        self.decoder_ln_pre = norm_layer(self.clip_out_dim)
        
        self.use_fim = use_fim
        self.use_mixed_fim = use_mixed_fim
        # cannot use mixed_fim and fim at the same time
        assert (not use_fim) or (not use_mixed_fim), "You can not use hierachical transformer and plain transformer at the same time!"
        self.fim_blocks = None
        if use_mixed_fim: # True
            self.fim_blocks = nn.ModuleList([
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution= 1.),
                ConvCrossAttentionBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1, resolution= 2.),
                ])

        elif use_fim:
            self.fim_blocks = nn.ModuleList([
                AttBlock(self.clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
                for _ in range(fim_depth)])


        self.decoder_norm = norm_layer(self.clip_out_dim)


        # --------------------------------------------------------------------------
        # CNN-based density decoder
        self.density_decoder = DensityDecoder(self.clip_out_dim, 384, use_hiearachy = use_mixed_fim)
        self.att = DualAttention(512, 512, 3)
        
        # --------------------------------------------------------------------------
    

    def forward_visual_encoder(self, x):
        """
        input: x: images, [B, 3, 384, 384]
        """
        # embed patches
        if x.shape[1]==1:
            x = x.repeat(1,3,1,1) 
        x = self.preprocess(x)  #([32, 3, 224, 224])
        _, cls_token, x = self.img_encoder(x)  #cls_token=([32, 1, 512])
        return cls_token, x   # x=([32, 197, 768])

    def forward_decoder(self, img_feat_patches, text_embedding, imgs_vi): # [4, 197, 768] [4, 1, 512]
        """

        """

        extra_out = {}
        
        # x_cls = cls_token    #([32, 1, 512]) 
        # extra_out['x_cls'] = x_cls
        extra_out['text_embedding'] = text_embedding
        # add pos embed

        # patch_feat = img_feat_patches[:,1:,:]   #([32, 196, 768])
        patch_feat = img_feat_patches   #([32, 196, 768]) [3, 65536, 64]
        patch_embedding = self.patch_feat_proj(patch_feat)  # [32, 196, 512]
        extra_out['patch_embedding'] = patch_embedding
        # patch_embedding_contrast = self.patch_feat_proj_contrast(patch_feat)
        # extra_out['patch_embedding_contrast'] = patch_embedding_contrast
        x = patch_embedding    #([32, 196, 512]),([3, 65536, 512])
        # x = x + self.patch_emb_pos_embed # ([32, 196, 512])

        y_ = text_embedding #  ([32, 1, 512])


        """
        informaton fusion block
        """
        # apply Transformer blocks (cross-attention)
        if self.use_mixed_fim: 
            xs = []
            for blk in self.fim_blocks:
                x = blk(x, y_)    #([4, 784, 512])
                xs.append(self.seq_2_2d(x))
        elif self.use_fim:
            for blk in self.fim_blocks:
                x = blk(x, y_)
        else: #add
            x = x + y_
        x = self.decoder_norm(x)    #([4, 784, 512])
        
        # Density map regression
        x = self.seq_2_2d(x,imgs_vi)  #[32, 512, 14, 14]
        feature = x

        extra_out['pixel_text_matching_map'] = x
        if self.use_mixed_fim:
            pred_density = self.density_decoder.forward_hierarchical(xs)  #([32, 384, 384])
        else:
            pred_density = self.density_decoder(x)

        return pred_density, extra_out, feature

    def forward(self, imgs, text, return_extra:bool = False, coop_require_grad:bool = False):

        text_token = clip.tokenize(text).to(imgs.device)  # text_token = [32, 77]) ([32, 77])

        if coop_require_grad:
            text_embedding = self.text_encoder(text_token).float()
        else:
            with torch.no_grad():
                text_embedding = self.text_encoder(text_token).float() # ([32, 1, 512])

        cls_token, img_feat_patches = self.forward_visual_encoder(imgs) # imgs: [32, 3, 384, 384]
        pred_density, extra_out = self.forward_decoder(img_feat_patches, text_embedding, cls_token)  # [N, 384, 384] ([32, 384, 384])
        
        if return_extra:
            return pred_density, extra_out
        return pred_density
    
    def seq_2_2d(self,x,imgs_vi):
        n, hw, c = x.shape
        h,w = imgs_vi.shape[2], imgs_vi.shape[3]
        # h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w) 
        return x

class CLIPViT(nn.Module):
    """
    ViT encoder for CLIP
    """
    def __init__(self, 
                 clip_model, 
                 clip_embed_dim:int, 
                 use_vpt:bool, 
                 vpt_width:int, 
                 vpt_depth:int = 8, 
                 unfreeze:bool=False) -> None:
        """
        Param:
            clip_model: pretrained OpenAI CLIP model
            use_vpt: whether to use visual prompt tuning
            vpt_width: number of vpt token per layer
            vpt_depth: number of vpt layers. 1: vpt at the first layer (shallow), >1: deep vpt
            unfreeze: If true, unfreeze the CLIP model
        """
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.vit = clip_model.visual
        if unfreeze:
            for param in self.vit.parameters():
                param.requires_grad = True
        self.use_vpt = use_vpt # what's the default of this? 
        self.visual_prompt = None
        self.vpt_dropout = None
        self.vpt_norm = None
        self.vpt_proj = None
        self.vpt_depth = vpt_depth
        self.vpt_width = vpt_width
        self.visual_prompt = None
        if use_vpt:
            self.vpt_dropout = nn.Dropout(0.1)
            self.vpt_norm = nn.LayerNorm(clip_embed_dim, eps=1e-6)
            self.vpt_proj = nn.Linear(clip_embed_dim, clip_embed_dim)
            nn.init.kaiming_normal_(self.vpt_proj.weight, a=0, mode='fan_out') 

            patch_size = self.vit.conv1.kernel_size
            val = math.sqrt(6. / float(3 * functools.reduce(operator.mul, patch_size, 1) + self.clip_embed_dim))  
            vpt = torch.empty((vpt_depth, vpt_width, clip_embed_dim))
            nn.init.uniform_(vpt, -val, val)
            self.visual_prompt = nn.Parameter(vpt)
            
    def forward(self, image):
        """
        input: image: [B, 3, 224, 224]
        """
        x = self.vit.conv1(image)  # shape = [*, width, grid, grid]  ([32, 768, 14, 14])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  ([32, 768, 196]) 
        # x = image.reshape(image.shape[0], image.shape[1], -1)  # shape = [*, width, grid ** 2]  ([32, 768, 196]) 
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]   ([32, 196, 768]) 
        img_patches = x     #([32, 196, 768])
        

        x = torch.cat([self.vit.class_embedding.to(x.dtype) + \
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                        x], dim=1)  # shape = [*, grid ** 2 + 1, width]   ([32, 197, 768])
        x = x + self.vit.positional_embedding.to(x.dtype)  #([32, 197, 768])

        if self.use_vpt:
            vpts = einops.repeat(self.visual_prompt[0,...], 'n d -> b n d', b=x.shape[0])  #([32, 20, 768])
            x = torch.cat([x[:, :1, :],
                            self.vpt_dropout(self.vpt_proj(vpts)),
                            x[:,1:,:]], dim=1)  # shape = [*, grid ** 2 + 1 + n_vpt, width]

        x = self.vit.ln_pre(x)   # ([32, 217, 768])

        x = x.permute(1, 0, 2)  # NLD -> LND  # ([217, 32, 768])
        if (not self.use_vpt) or self.vpt_depth == 1 :
            x = self.vit.transformer(x)  #([217, 32, 768])


        if self.use_vpt and self.vpt_depth > 1:
            x = self.deep_vpt_forward(x) #([197, 32, 768])
        x = x.permute(1, 0, 2)  # LND -> NLD ([32, 197, 768])

        x_cls = x[:, :1, :]  # [CLS] token ([32, 1, 768])
        x_cls = self.vit.ln_post(x_cls)  #([32, 1, 512])
        x_cls = x_cls @ self.vit.proj  #([32, 1, 512])
        return img_patches, x_cls, x  #img_patches=([32, 196, 768]) x_cls=([32, 1, 512]) x=([32, 197, 768])
    

    def deep_vpt_forward(self, embedding_output, out_last = False):
        B = embedding_output.shape[1]
        transformer = self.vit.transformer
        assert self.vpt_depth < transformer.layers , "vpt_depth should be smaller than the number of layers in the transformer"
        for i in range(transformer.layers):
            if i == 0:
                hidden_states = transformer.resblocks[i](embedding_output)
            elif i < self.vpt_depth:
                deep_prompt_emb = self.vpt_dropout(self.vpt_proj(self.visual_prompt[i-1,...]).expand(B, -1, -1)).permute(1, 0, 2)
                # B, L, 768

                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.vpt_width):, :, :]
                ), dim=0)

                hidden_states = transformer.resblocks[i](hidden_states)
            elif i == self.vpt_depth:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[(1+self.vpt_width):, :, :]
                ), dim=0)
                hidden_states = transformer.resblocks[i](hidden_states)
            else:
                hidden_states = transformer.resblocks[i](hidden_states)
            
            if i == (transformer.layers-1): #11
                before_last_feats = self.vpt_norm(hidden_states)

        encoded = self.vpt_norm(hidden_states)
        if out_last:
            return before_last_feats, encoded
        else:
            return encoded
    
class CLIPTextTransformer(nn.Module):
    """
    Transfromer encoder (text) for CLIP
    """
    def __init__(self, clip_model, use_coop:bool, n_ctx:int = 2) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.learnable_context = None
        self.use_coop = use_coop #global context for all classes
        if use_coop:
            self.n_ctx = n_ctx
            context_vectors = torch.empty(self.n_ctx, self.clip_model.ln_final.weight.shape[0])
            torch.nn.init.normal_(context_vectors, std=.02)
            self.learnable_context = nn.Parameter(context_vectors) # [n_ctx, 512]

    def forward(self, text):
        """
        Input:
            text: tokenized text, shape = [batch_size, n_ctx]
        """
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        if self.use_coop:
            sos_token = x[:, 0, :].unsqueeze(1)  # [batch_size, 1, d_model]
            suffix_tokens = x[:, 1:-self.n_ctx, :] # class tokens + [EOS] token
            ctx = einops.repeat(self.learnable_context, 'n d -> b n d', b=x.shape[0])
            x = torch.cat([sos_token, ctx, suffix_tokens], dim=1)
        

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.visual.conv1.weight.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        x = x.unsqueeze(1)  # [batch_size, 1, transformer.width]
        return x

def condv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class DensityDecoder(nn.Module):
    def __init__(self, in_dim:int, target_hw:int, use_hiearachy:bool = False) -> None:
        super().__init__()
        # Density map regresssion module
        self.n_levels = 4 if use_hiearachy else 2
        self.target_hw = [target_hw, target_hw]
        convs = []
        crt_dim = in_dim # number of feature channels
        for i in range(self.n_levels):
            decode_head = nn.Sequential(
                nn.Conv2d(crt_dim, crt_dim//2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, crt_dim//2),
                nn.GELU()
            )
            convs.append(decode_head)
            crt_dim = crt_dim//2

        self.convs = nn.ModuleList(convs)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(crt_dim, 1, kernel_size=1, stride=1)
        )
        #initialize weights
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
        self.pyradim_conv = None # the conv to squeeze the fine multimodel features
        if use_hiearachy:
            self.pyradim_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.GroupNorm(8, 256),
                nn.GELU()
            )

        n_resblocks = 2
        m_body = [ResBlock(condv, 512, 3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) 
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body)
        # Tail
        m_tail = [ condv(128, 64, 3),nn.ReLU(True),condv(64, 1, 3), nn.Tanh()]
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x): # [4, 512, 14, 14]
        # target_hw = [imgs_vi.shape[2], imgs_vi.shape[3]]
        x = self.body(x)
        # x = self.tail(x)

        for i in range(self.n_levels): #2

            x = self.convs[i](x)
            # x = self.up_convs[i](x)
            # if i < self.n_levels-1:
            #     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # else:
            #     x = F.interpolate(x, size = target_hw, mode = 'bilinear', align_corners = False)
        # x = self.final_conv(x)
        x = self.tail(x) # [1, 1, 14, 14]
        
        # x = F.sigmoid(x)
        # x = einops.rearrange(x, 'n 1 h w -> n h w')
        return x

    def forward_hierarchical(self, xs):
        """
        xs: [14,14,512], [28,28,512]
        """
        x0, x1= xs[0], xs[1]  #([32, 512, 14, 14]) ([32, 512, 28, 28])
        x = x0
        for i in range(self.n_levels): # 4
            if i == 1:
                x = x + self.pyradim_conv(x1)

            x = self.convs[i](x)
            if i < self.n_levels-1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # update to use subpixel
            else:
                x = F.interpolate(x, size = (384, 384), mode = 'bilinear', align_corners = False)
        x = self.final_conv(x) #([32, 1, 384, 384])
        
        x = F.sigmoid(x)
        x = einops.rearrange(x, 'n 1 h w -> n h w') #([32, 384, 384])
        return x

class MetaAdaptor(nn.Module):
    def __init__(self, text_dim=512, vision_dim=512, query_length=4):
        super(MetaAdaptor, self).__init__()

        self.query = nn.Parameter(torch.randn(query_length, vision_dim))
        self.cross_attention = CrossAttentionBlock(
            vision_dim, num_heads=8, qkv_bias=True, qk_scale=None, drop=0.1, drop_path=0.1)

    def forward(self, text_information):
        attended_output = self.cross_attention(self.query.unsqueeze(0).expand(text_information.size(0), -1, -1),
                                               text_information)

        return attended_output

##########################################################################
## ------ Spatial Attention --------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
##---------- Dual Attention Unit ----------
class DualAttention(nn.Module):
    def __init__(
            self,infeat, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualAttention, self).__init__()
        modules_body = [conv(infeat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res


class GCLIPCount(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.metaadaptor = MetaAdaptor(query_length=query_length)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion_vi = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.fusion_ir = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.cla = ImageTextFusionModule(64, 64,64)
        self.att1 = DualAttention(64, 64, 3)
        self.att2 = DualAttention(64, 64, 3)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    
    

    def forward(self, imgs_vi, imgs_ir, text_vi, text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        # cls_token_vi, img_feat_patches_vi = self.forward_visual_encoder(imgs_vi) # [4, 1, 512] ([32, 197, 768])
        # cls_token_ir, img_feat_patches_ir = self.forward_visual_encoder(imgs_ir)
        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 
        img_feat_patches_vi = self.att1(img_feat_patches_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.att2(img_feat_patches_ir) 

        patch_feat_vi = img_feat_patches_vi
        patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]


        # patch_feat_ir = img_feat_patches_ir[:,1:,:]
        
        # patch_feat_ir = img_feat_patches_ir[:,1:,:]
        patch_feat_ir = img_feat_patches_ir
        patch_feat_ir = self._2d_to_seq(patch_feat_ir)
        
        text_embedding = text.unsqueeze(1) # [32, 1, 512]

        img_feat_patches =  patch_feat_vi + patch_feat_ir # [32, 197, 768] [3, 65536, 64]

        pred_density, extra_out = self.forward_decoder(img_feat_patches, text_embedding, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x

#######################################W/o fusion text###############################################################
class WO_fusiontext(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.metaadaptor = MetaAdaptor(query_length=query_length)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.att1 = DualAttention(64, 64, 3)
        self.att2 = DualAttention(64, 64, 3)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    
    def notext_decoder(self, img_feat_patches, imgs_vi): # [4, 197, 768] [4, 1, 512]
        """

        """

        extra_out = {}
        
        patch_feat = img_feat_patches   #([32, 196, 768]) [3, 65536, 64]
        patch_embedding = self.patch_feat_proj(patch_feat)  # [32, 196, 512]
        extra_out['patch_embedding'] = patch_embedding
        # patch_embedding_contrast = self.patch_feat_proj_contrast(patch_feat)
        # extra_out['patch_embedding_contrast'] = patch_embedding_contrast
        x = patch_embedding    #([32, 196, 512]),([3, 65536, 512])
        # x = x + self.patch_emb_pos_embed # ([32, 196, 512])
        
        x = self.decoder_norm(x)    #([4, 784, 512])
        
        # Density map regression
        x = self.seq_2_2d(x,imgs_vi)  #[32, 512, 14, 14]
        feature =x
        # x = self.att(x) # [32, 512, 14, 14])

        extra_out['pixel_text_matching_map'] = x
        if self.use_mixed_fim:
            pred_density = self.density_decoder.forward_hierarchical(xs)  #([32, 384, 384])
        else:
            pred_density = self.density_decoder(x)

        return pred_density, extra_out, feature

    def forward(self, imgs_vi, imgs_ir, text_vi, text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 
        img_feat_patches_vi = self.att1(img_feat_patches_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.att2(img_feat_patches_ir) 

        patch_feat_vi = img_feat_patches_vi
        patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]

        patch_feat_ir = img_feat_patches_ir
        patch_feat_ir = self._2d_to_seq(patch_feat_ir)

        text_embedding_vi = text_vi.unsqueeze(1) # [1, 1, 512]
        text_embedding_ir = text_ir.unsqueeze(1) 
        # text_embedding = text.unsqueeze(1) # [32, 1, 512]
        

        # if self.g2t:
        text_img_vi = self.g2t[1]((patch_feat_vi), self.g2t[0](text_embedding_vi)) # ([32, 197, 768]) 
        text_img_ir = self.g2i[1]((patch_feat_ir), self.g2i[0](text_embedding_ir)) # ([32, 197, 768])

        ir_feature = self.seq_2_2d(text_img_ir, imgs_vi)
        vi_feature = self.seq_2_2d(text_img_vi, imgs_vi)
        
        img_feat_patches = text_img_vi + text_img_ir # [32, 197, 768] [3, 65536, 64]


        pred_density, extra_out, feature = self.notext_decoder(img_feat_patches, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density,vi_feature ,ir_feature, feature

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x

class CLIPfusion(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.metaadaptor = MetaAdaptor(query_length=query_length)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.att1 = DualAttention(64, 64, 3)
        self.att2 = DualAttention(64, 64, 3)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    
    

    def forward(self, imgs_vi, imgs_ir, text_vi, text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 
        img_feat_patches_vi = self.att1(img_feat_patches_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.att2(img_feat_patches_ir) 

        patch_feat_vi = img_feat_patches_vi
        patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]

        patch_feat_ir = img_feat_patches_ir
        patch_feat_ir = self._2d_to_seq(patch_feat_ir)

        text_embedding_vi = text_vi.unsqueeze(1) # [1, 1, 512]
        text_embedding_ir = text_ir.unsqueeze(1) 
        text_embedding = text.unsqueeze(1) # [32, 1, 512]

        # if self.g2t:
        text_img_vi = self.g2t[1]((patch_feat_vi), self.g2t[0](text_embedding_vi)) # ([32, 197, 768]) 
        text_img_ir = self.g2i[1]((patch_feat_ir), self.g2i[0](text_embedding_ir)) # ([32, 197, 768])

        ir_feature = self.seq_2_2d(text_img_ir, imgs_vi)
        vi_feature = self.seq_2_2d(text_img_vi, imgs_vi)
        
        img_feat_patches = text_img_vi + text_img_ir # [32, 197, 768] [3, 65536, 64]


        pred_density, extra_out, feature = self.forward_decoder(img_feat_patches, text_embedding, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x
    def seq_2_2d(self,x,imgs_vi):
        n, hw, c = x.shape
        h,w = imgs_vi.shape[2], imgs_vi.shape[3]
        # h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w) 
        return x
    

##############################################新增文本交互模块######################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 4, 1, 0)

    def forward(self, h):
        h = self.conv2(self.act(self.conv1(h)))
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.main = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(in_dim, in_dim)),
            ('relu1',nn.LeakyReLU(0.2, inplace=True)),
            ('linear2',nn.Linear(in_dim, out_dim)),
            ]))

    def forward(self, h):
        h = self.main(h)
        return h

class CWP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CWP, self).__init__()
        self.img_enc = ConvBlock(in_ch, out_ch)
        self.ln = nn.Linear(in_features=512, out_features=out_ch)
        self.txt_enc = MLP(out_ch, out_ch)
        self.fuse = MLP(16386560, 256*2)
        self.fc = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256*4, 256*4)),
            ('relu1',nn.LeakyReLU(0.2,inplace=True)),
        ]))

    def forward(self, v, t):
        img_emb = self.img_enc(v).view(v.size(0),-1)
        t = self.ln(t).view(t.size(0), -1)
        txt_emb = self.txt_enc(t)
        ti = self.fuse(torch.cat((img_emb, txt_emb), dim=1))
        tid = ti[:,:256] - ti[:,256:]
        tim = ti[:,:256] * ti[:,256:]
        ti = self.fc(torch.cat((tid, tim, img_emb, txt_emb), dim=1))
        return ti

class CLIPfusion2(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.cwp = CWP(64, 256)
        self.cwp2 = CWP(64, 256)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.att1 = DualAttention(64, 64, 3)
        self.att2 = DualAttention(64, 64, 3)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    
    

    def forward(self, imgs_vi, imgs_ir, text_vi, text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 
        img_feat_patches_vi = self.att1(img_feat_patches_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.att2(img_feat_patches_ir) 

        patch_feat_vi = img_feat_patches_vi
        # patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]

        patch_feat_ir = img_feat_patches_ir
        # patch_feat_ir = self._2d_to_seq(patch_feat_ir)

        text_embedding_vi = text_vi.unsqueeze(1) # [1, 1, 512]
        text_embedding_ir = text_ir.unsqueeze(1) 
        text_embedding = text.unsqueeze(1) # [32, 1, 512]

        # if self.g2t:
        # text_img_vi = self.g2t[1]((patch_feat_vi), self.g2t[0](text_embedding_vi)) # ([32, 197, 768]) 
        # text_img_ir = self.g2i[1]((patch_feat_ir), self.g2i[0](text_embedding_ir)) # ([32, 197, 768])
        text_img_vi = self.cwp(img_feat_patches_vi, text_embedding_vi)
        text_img_ir = self.cwp2(img_feat_patches_ir, text_embedding_ir)
        
        img_feat_patches = text_img_vi + text_img_ir # [32, 197, 768] [3, 65536, 64]


        pred_density, extra_out = self.forward_decoder(img_feat_patches, text_embedding, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x

#####################################去掉ir text#################################################################
class WO_ir(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.metaadaptor = MetaAdaptor(query_length=query_length)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.att1 = DualAttention(64, 64, 3)
        self.att2 = DualAttention(64, 64, 3)
        self.irtrans = nn.Linear(in_features=64, out_features=768)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    
    

    def forward(self, imgs_vi, imgs_ir, text_vi,text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 
        img_feat_patches_vi = self.att1(img_feat_patches_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.att2(img_feat_patches_ir) 

        patch_feat_vi = img_feat_patches_vi
        patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]

        patch_feat_ir = img_feat_patches_ir
        patch_feat_ir = self._2d_to_seq(patch_feat_ir)

        text_embedding_vi = text_vi.unsqueeze(1) # [1, 1, 512]
        text_embedding = text.unsqueeze(1) # [32, 1, 512]

        # if self.g2t:
        text_img_vi = self.g2t[1]((patch_feat_vi), self.g2t[0](text_embedding_vi)) # ([32, 197, 768]) 
        # ir_feature = self.seq_2_2d(text_img_ir, imgs_vi)
        vi_feature = self.seq_2_2d(text_img_vi, imgs_vi)

        
        img_feat_patches = text_img_vi + patch_feat_ir # [32, 197, 768] [3, 65536, 64]
        pred_density, extra_out,feature = self.forward_decoder(img_feat_patches, text_embedding, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density, vi_feature, img_feat_patches_ir, feature

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x
    
    def seq_2_2d(self,x,imgs_vi):
        n, hw, c = x.shape
        h,w = imgs_vi.shape[2], imgs_vi.shape[3]
        # h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w) 
        return x


#####################################去掉vis text#################################################################
class WO_vis(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.metaadaptor = MetaAdaptor(query_length=query_length)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.att1 = DualAttention(64, 64, 3)
        self.att2 = DualAttention(64, 64, 3)
        self.irtrans = nn.Linear(in_features=64, out_features=768)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    
    

    def forward(self, imgs_vi, imgs_ir, text_vi,text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 
        img_feat_patches_vi = self.att1(img_feat_patches_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.att2(img_feat_patches_ir) 

        patch_feat_vi = img_feat_patches_vi
        patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]

        patch_feat_ir = img_feat_patches_ir
        patch_feat_ir = self._2d_to_seq(patch_feat_ir)

        text_embedding_ir = text_ir.unsqueeze(1) # [1, 1, 512]
        text_embedding = text.unsqueeze(1) # [32, 1, 512]

        # if self.g2t:
        text_img_ir = self.g2i[1]((patch_feat_ir), self.g2t[0](text_embedding_ir)) # ([32, 197, 768]) 

        ir_feature = self.seq_2_2d(text_img_ir, imgs_vi)
        # vi_feature = self.seq_2_2d(text_img_vi, imgs_vi)
        
        img_feat_patches = text_img_ir + patch_feat_vi # [32, 197, 768] [3, 65536, 64]
        pred_density, extra_out, feature = self.forward_decoder(img_feat_patches, text_embedding, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out
        return pred_density, img_feat_patches_vi, ir_feature, feature

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x
    
    def seq_2_2d(self,x,imgs_vi):
        n, hw, c = x.shape
        h,w = imgs_vi.shape[2], imgs_vi.shape[3]
        # h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w) 
        return x
######################################################(W/o Dual Attention)####################################################################################################
class wo_DA(CLIPCount):
    def __init__(self, *args, fim_num_heads=8, mlp_ratio: float = 4., norm_layer=nn.LayerNorm, query_length=4, act=nn.PReLU(), **kwargs):
        super().__init__(*args, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, **kwargs)
        self.g2i = True 
        self.g2t = True 
        self.metaadaptor = MetaAdaptor(query_length=query_length)
        
        modules_body = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head =  nn.Sequential(*modules_body)

        modules_body2 = [conv(1, 64, 3, bias=False), act, conv(64, 64, 3, bias=False)]
        self.head2 =  nn.Sequential(*modules_body2)

        # if self.g2i:
        self.g2i = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        # if self.g2t:
        self.g2t = self._create_g2_module(in_features=512, clip_out_dim=64, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.fusion = self._create_g2_module(in_features=512, clip_out_dim=512, fim_num_heads=fim_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer)

    def _create_g2_module(self, in_features, clip_out_dim, fim_num_heads, mlp_ratio, norm_layer):
        return nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=clip_out_dim),
            AttBlock(clip_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop=0.1, drop_path=0.1)
        ])
    

    def forward(self, imgs_vi, imgs_ir, text_vi, text_ir, text, return_extra: bool = False, coop_require_grad: bool = False): # [4, 1, 256, 256], [4, 512]

        img_feat_patches_vi = self.head(imgs_vi)  # [32, 64, 256, 256]
        img_feat_patches_ir = self.head2(imgs_ir) 

        patch_feat_vi = img_feat_patches_vi
        patch_feat_vi = self._2d_to_seq(patch_feat_vi) # [1, 196, 64]

        patch_feat_ir = img_feat_patches_ir
        patch_feat_ir = self._2d_to_seq(patch_feat_ir)

        text_embedding_vi = text_vi.unsqueeze(1) # [1, 1, 512]
        text_embedding_ir = text_ir.unsqueeze(1) 
        text_embedding = text.unsqueeze(1) # [32, 1, 512]

        # if self.g2t:
        text_img_vi = self.g2t[1]((patch_feat_vi), self.g2t[0](text_embedding_vi)) # ([32, 197, 768]) 
        text_img_ir = self.g2i[1]((patch_feat_ir), self.g2i[0](text_embedding_ir)) # ([32, 197, 768])

        ir_feature = self.seq_2_2d(text_img_ir, imgs_vi)
        vi_feature = self.seq_2_2d(text_img_vi, imgs_vi)
        
        img_feat_patches = text_img_vi + text_img_ir # [32, 197, 768] [3, 65536, 64]


        pred_density, extra_out, feature = self.forward_decoder(img_feat_patches, text_embedding, imgs_vi)  # [N, 384, 384]
        
        if return_extra:
            return pred_density, extra_out,vi_feature, ir_feature, feature
        return pred_density, vi_feature, ir_feature, feature

    def _get_text_embedding(self, text_token, coop_require_grad):
        with torch.set_grad_enabled(coop_require_grad):
            text_embedding = self.text_encoder(text_token).float()
        return text_embedding
    
    
    def _2d_to_seq(self,x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h*w).transpose(1, 2)
        return x
    
    def seq_2_2d(self,x,imgs_vi):
        n, hw, c = x.shape
        h,w = imgs_vi.shape[2], imgs_vi.shape[3]
        # h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w) 
        return x
    

##---------- Spatial Attention ----------

class Inter_Att(torch.nn.Module):
    def __init__(self, channels):
        super(Inter_Att, self).__init__()

        self.sigmoid = nn.Sigmoid()  # 更正为正确的命名
        self.ca_avg = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channels, channels // 2, kernel_size=1),
                nn.PReLU(),
                nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.ca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
        )
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(2 * channels, channels, 1, 1)

    def forward(self, ir, vis):
        w_ir_avg = self.ca_avg(ir)
        w_ir_max = self.ca_max(ir)
        w_ir = torch.cat([w_ir_avg, w_ir_max], dim=1)
        w_ir = self.conv1(w_ir)
        w_ir_f = self.sigmoid(w_ir)  # 使用修正后的命名

        w_vis_avg = self.ca_avg(vis)
        w_vis_max = self.ca_max(vis)
        w_vis = torch.cat([w_vis_avg, w_vis_max], dim=1)
        w_vis = self.conv1(w_vis)
        w_vis_f = self.sigmoid(w_vis)  # 使用修正后的命名

        EPSILON = 1e-10
        mask_ir = torch.exp(w_ir_f) / (torch.exp(w_ir_f) + torch.exp(w_vis_f) + EPSILON)
        mask_vis = torch.exp(w_vis_f) / (torch.exp(w_ir_f) + torch.exp(w_vis_f) + EPSILON)
        out_ir = mask_ir * ir
        out_vis = mask_vis * vis

        avgout_ir = torch.mean(out_ir, dim=1, keepdim=True)
        maxout_ir, _ = torch.max(out_ir, dim=1, keepdim=True)
        x_ir = torch.cat([avgout_ir, maxout_ir], dim=1)
        x1_ir = self.conv(x_ir)
        x2_ir = self.sigmoid(x1_ir)  # 使用修正后的命名

        avgout_vis = torch.mean(out_vis, dim=1, keepdim=True)
        maxout_vis, _ = torch.max(out_vis, dim=1, keepdim=True)
        x_vis = torch.cat([avgout_vis, maxout_vis], dim=1)
        x1_vis = self.conv(x_vis)
        x2_vis = self.sigmoid(x1_vis)  # 使用修正后的命名

        mask_ir_sa = torch.exp(x2_ir) / (torch.exp(x2_ir) + torch.exp(x2_vis) + EPSILON)
        mask_vis_sa = torch.exp(x2_vis) / (torch.exp(x2_ir) + torch.exp(x2_vis) + EPSILON)

        output_ir = mask_ir_sa * out_ir
        output_vis = mask_vis_sa * out_vis

        output = torch.cat([output_ir, output_vis], dim=1)

        return output
        


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

# if __name__ == "__main__":
#     clip_count = CLIPCount()
if __name__ == '__main__':
    x = torch.randn(1, 1, 640, 480).cuda()
    y = torch.randn(1, 1, 640, 480).cuda()
    u = torch.randn(1, 512).cuda()
    v = torch.randn(1, 512).cuda()
    z = torch.randn(1, 512).cuda()
    model = CLIPfusion().cuda()
    outputs = model(x, y)
    model.eval()
    print("Params(M): %.3f" % (params_count(model) / (1000 ** 2)))

    # from thop import profile
    # flops, params = profile(model, inputs=[x, y])
    # print("Params(M): %.2f" % (params / 1e6))
    # print("FLOPs(G): %.4f" % (flops / 1e9))

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # model = FSFNet(32).cuda()
    # flops = FlopCountAnalysis(model, (x, y))
    # print("FLOPs(G): %.4f" % (flops.total()/1e9))
    # print(parameter_count_table(model))

    import time
    #
    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = model(x, x)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                out = model(x, x)
            torch.cuda.synchronize()
            result.append((time.time() - st)/N)
        print("Running Time: {:.3f}s\n".format(np.mean(result)))