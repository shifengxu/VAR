################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


def prepare_var(device):
    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f"vae_ckpt: {vae_ckpt}")
    print(f"var_ckpt: {var_ckpt}")
    print(f'VAR model prepare finished.')
    return var

def manual_seed():
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def main():
    ############################# 2. Sample with classifier-free guidance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    var = prepare_var(device)
    seed = manual_seed()
    print(f"device: {device}")
    print(f"seed  : {seed}")

    # set args
    cfg = 4                     #@param {type:"slider", min:1, max:10, step:0.1}
    # class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
    class_labels = (980, 980)   #@param {type:"raw"}
    more_smooth = False # True for more smooth output

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # sample
    bs = len(class_labels)  # batch size
    print(f"labels: {class_labels}")
    print(f"B     : {bs}")
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device, dtype=torch.long)
    with torch.inference_mode():
        # using bfloat16 can be faster
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_b3hw = var.autoregressive_infer_cfg(B=bs, label_B=label_B, cfg=cfg, top_k=900,
                                                      top_p=0.95, g_seed=seed, more_smooth=more_smooth)

    chw = torchvision.utils.make_grid(recon_b3hw, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    img_path = f"image_generated_seed{seed}.png"
    chw.save(img_path)
    print(f"saved: {img_path}")

if __name__ == '__main__':
    main()
