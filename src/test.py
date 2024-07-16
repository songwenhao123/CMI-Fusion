import os
import cv2
import torch
import clip
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn.functional as F
from sys import platform
from PIL import Image
from torchvision.io.image import read_image, ImageReadMode
from config import args as args_config
from einops import rearrange
# from models import Encoder, Decoder
from ClipCount.model import  clip_count
from template import imagenet_templates
from templates_ir import imagenet_templates_ir
from templates_vi import imagenet_templates_vi
from m3fds import M3fdMix



def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def rgb_y(im):
    im_ra = rearrange(im, 'c h w -> h w c').numpy()
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:,:,0]).unsqueeze(0)  
    return im_y 

def clip_norm(im):      
    DEV = im.device   
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im_re = F.interpolate(im.repeat(1,3,1,1) if im.shape[1]==1 else im, size=224, mode='bilinear', align_corners=False)
    im_norm = (im_re - mean) / std
    return im_norm

def to_rgb(im_rgb, im_y):
    im_rgb_ra = rearrange(im_rgb.squeeze(0), 'c h w -> h w c').cpu().numpy()
    im_y_ra = rearrange(im_y.squeeze(0), 'c h w -> h w c').cpu().numpy()
    y = np.expand_dims(im_y_ra[:,:,0], -1)
    crcb = cv2.cvtColor(im_rgb_ra, cv2.COLOR_RGB2YCrCb)[:,:,1:]
    ycrcb = np.concatenate((y, crcb), -1)
    ir_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return rearrange(torch.from_numpy(ir_rgb), 'h w c -> c h w').unsqueeze(0).to(device=im_y.device)

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

def load(pt_path, load_epoch, device='cuda:1'):

    # MODEL = clip_count.GCLIPCount(
    #                     fim_depth=2,
    #                     fim_num_heads=8,
    #                     use_coop=True, 
    #                     use_vpt=True,
    #                     coop_width=2,
    #                     vpt_width=20,
    #                     vpt_depth= 10,
    #                     backbone = "b16",
    #                     use_fim = True,
    #                     use_mixed_fim = False,
    #                     unfreeze_vit = False,
    #                     query_length = 4,
    #                     ).to(device=device).eval()
    
    MODEL = clip_count.CLIPfusion(
                        fim_depth=2,
                        fim_num_heads=8,
                        use_coop=True, 
                        use_vpt=True,
                        coop_width=2,
                        vpt_width=20,
                        vpt_depth= 10,
                        backbone = "b16",
                        use_fim = True,
                        use_mixed_fim = False,
                        unfreeze_vit = False,
                        query_length = 4,
                        ).to(device=device).eval()
    print("Params(M): %.3f" % (params_count(MODEL) / (1000 ** 2)))

    MODEL_w = torch.load(str(pt_path + f'/model_{load_epoch:05d}.pt'), map_location=device)
    MODEL.load_state_dict(MODEL_w)


    print('=== Pretrained models load done ===')
    
    return MODEL.float()

def t4d_save(t4d, epoch, save_path, save_file_name, if_print=False):
    C = t4d.shape[1]
    if C == 1:
        im = 255 * t4d.cpu().squeeze(0).squeeze(0).clamp(0,1)
        im = Image.fromarray(im.numpy().astype('uint8'))
    else:
        im = 255 * t4d.cpu().squeeze(0).clamp(0,1)
        im = Image.fromarray(rearrange(im, 'c h w -> h w c').numpy().astype('uint8'))
    
    if not os.path.exists(save_path + f'single_{epoch}/'):
            os.makedirs(save_path + f'single_{epoch}/')
            if if_print:
                print(f"Directory {save_path} was created.")

    # if epoch == 'null':
    #     save_file = save_path + f'single_{epoch}/' + save_file_name
    #     im.save(save_file, quality=100)
    # else:
    #     save_file = save_path + str(epoch) + '_' + save_file_name
    #     im.save(save_file, quality=100)
    save_file = save_path + f'single_{epoch}/' + save_file_name
    im.save(save_file, quality=100)    
    if if_print: print(f'Saved: {save_file}')


def test(dev='cuda:0', data_type=2, epoch=1, pt_folder=args_config.save_dir): 
    if_template = False
    MODEL = load(pt_folder, epoch, dev) # 导入模型
    clip_model, _ = clip.load("ViT-B/32")

    if platform == 'win32':
        data_folder = ['./test_imgs/TNO_test', 
                       './test_imgs/RoadScene_test', 
                       'D:/dataset/FusionData/M3FD/M3FD_Fusion']  
    elif platform == 'linux':
        data_folder = ['/shares/image_fusion/IVIF_datasets/test/test_TNO_25', 
                       '/shares/image_fusion/IVIF_datasets/test/test_NIR_Country', 
                       '/shares/image_fusion/IVIF_datasets/test/test_MSRS',
                       '/scratch/wenhao/MyDatasets/CT-MRI/test',
                       '/scratch/wenhao/MyDatasets/PET-MRI/test',
                       '/scratch/wenhao/MyDatasets/SPECT-MRI/test',
                       '/shares/image_fusion/IVIF_datasets/FMB/test',]  
    
    save_path = ['/home/wenhao/projects/LDFusion/self_results/TNO/three_text/', 
                 '/home/wenhao/projects/LDFusion/self_results/NIR_Country/three_text/', 
                 '/home/wenhao/projects/LDFusion/self_results/MSRS/clip_loss100/',
                 '/home/wenhao/projects/LDFusion/self_results/CT_MRI/three_text/',
                 '/home/wenhao/projects/LDFusion/self_results/PET_MRI/three_text/',
                 '/home/wenhao/projects/LDFusion/self_results/SPECT_MRI/three_text/',
                 '/home/wenhao/projects/CLIPFusion/self_results/FMB/',]
    
    ir_folder = ['ir', 'ir', 'ir','MRI','MRI','MRI','Infrared']
    vis_folder = ['vi', 'vi', 'vi','CT', 'PET','SPECT','Visible']


    # vi_text = ['a visible light gray image with sharp details and clear background']
    # ir_text = ['a infrared image showcasing temperature variations and highlighting the thermal of the objects']
    # prompt = ['a vivid image with clear background and obvious objects']

    vi_text = ['a visible light gray image, highlighting the the objects. ']
    ir_text = ['a infrared image with clear background.']
    prompt = ['a vivid image with clear background and obvious objects']

    # prompt = ['a image showcasing sharp details and vivid contrast, temperature variations along with the objects']
    # vi_text = ['a CT image highlighting the dense structures like bones and tissues with clear contrast']
    # ir_text = ['an MRI image showcasing soft tissues, brain or internal organs with detailed contrast and textures']
    # prompt = ['a fused image combining the detailed contrast of dense and soft tissues, providing a comprehensive view of the internal anatomy']

    # vi_text = ['a PET image']
    # ir_text = ['an MRI image']
    # prompt = ['a fused PET-MRI image illustrating both metabolic activities and detailed anatomical structures']

    # vi_text = ['a SPECT image showcasing functional information and distribution of radiotracers within the body']
    # ir_text = ['an MRI image showcasing soft tissues, brain or internal organs with detailed contrast and textures']
    # prompt = ['a fused SPECT-MRI image illustrating the detailed anatomical structures along with the functional and radiotracer distribution information, providing a comprehensive view of both functional and anatomical aspects']
    vit_template = compose_text_with_templates(vi_text, imagenet_templates_vi)        
    vit_token = clip.tokenize(vit_template if if_template else vi_text).to(dev)
    vit_feature = clip_model.encode_text(vit_token).detach().to(torch.float)
    vit_feature = vit_feature.mean(dim=0, keepdim=True)

    irt_template = compose_text_with_templates(ir_text, imagenet_templates_ir)
    irt_token = clip.tokenize(irt_template if if_template else ir_text).to(dev)
    irt_feature = clip_model.encode_text(irt_token).detach().to(torch.float)
    irt_feature = irt_feature.mean(dim=0, keepdim=True)

    prompt_template = compose_text_with_templates(prompt, imagenet_templates)
    prompt_token = clip.tokenize(prompt_template if if_template else prompt).to(dev)
    prompt_feature = clip_model.encode_text(prompt_token).detach().to(torch.float)
    prompt_feature = prompt_feature.mean(dim=0, keepdim=True)

    ir_path = data_folder[data_type] + '/' + ir_folder[data_type] + '/'
    vis_path = data_folder[data_type] + '/' + vis_folder[data_type] + '/'

    file_list = os.listdir(ir_path)

    # clip    
    print(f'Testing ... ')

    with torch.no_grad():
        for i in file_list:
        # for i in ["1.png"]:
            vis = read_image(vis_path + i, ImageReadMode.RGB) / 255.
            ir = read_image(ir_path + i, ImageReadMode.RGB) / 255.

            vis_y = rgb_y(vis).unsqueeze(0).to(device=dev)
            ir_y = rgb_y(ir).unsqueeze(0).to(device=dev)

            B, C, H, W = vis_y.shape

            fu = MODEL(vis_y, ir_y, vit_feature, irt_feature, prompt_feature)

            fu = fu[:,:,:H,:W]

            if data_type == 1 or data_type == 2 or data_type == 4 or data_type == 5 or data_type == 6:
                vis_y_r = vis_y.repeat(1,3,1,1)
                ir_y_r = ir_y.repeat(1,3,1,1)
                vis_rgb = vis.unsqueeze(0).to(device=dev)
                fu_rgb = to_rgb(vis_rgb, fu)
                cat = torch.cat([torch.cat([vis_y_r, ir_y_r], dim=3), torch.cat([vis_rgb, fu_rgb], dim=3)], dim=2)
                # t4d_save(cat, epoch, save_path[data_type], 'rgb_'+i, if_print=True)
                # t4d_save(fu_rgb, 'null', save_path[data_type], i, if_print=True)
                t4d_save(fu_rgb, epoch, save_path[data_type], i, if_print=True)
            else:
                cat = torch.cat([vis_y, ir_y, fu], dim=3)
                # t4d_save(cat, epoch, save_path[data_type], 'fu_'+i, if_print=True)
                # t4d_save(fu, 'null', save_path[data_type], i, if_print=True)
                t4d_save(fu, epoch, save_path[data_type], i, if_print=True)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = 1
    # config
    args = args_config

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':',  getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')

    print('\n')
    test(dev='cuda', 
        data_type=6, 
        epoch=100, 
        pt_folder='/home/wenhao/projects/CLIPFusion/experiments/240317_203818_three_text')


