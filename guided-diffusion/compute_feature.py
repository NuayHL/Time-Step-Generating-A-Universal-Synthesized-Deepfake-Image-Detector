"""
This file is modified from the following sources:

1. `guided-diffusion/scripts/image_sample.py`
   Original repository: https://github.com/openai/guided-diffusion

2. `guided-diffusion/compute_dire.py`
   Original repository: https://github.com/ZhendongWang6/DIRE

"""

import argparse
import os
import torch

import sys
import cv2
from mpi4py import MPI
import os

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th

from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.1,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res

def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main():
    args = create_argparser().parse_args()

    if not args.num_samples:
        args.num_samples = len(os.listdir(args.images_dir)) + 10

    print(f'Num of Images: {args.num_samples}')

    logger.configure(dir=args.recons_dir)

    # create recons_dir
    os.makedirs(args.recons_dir, exist_ok=True)
    logger.log(str(args))

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to('cuda:0')
    logger.log("have created model and diffusion")
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )

    t = int(args.time_step)

    logger.log("have created data loader")

    logger.log("computing Features ...")
    have_finished_images = 0
    while have_finished_images < args.num_samples:
        if (have_finished_images + MPI.COMM_WORLD.size * args.batch_size) > args.num_samples and (
            args.num_samples - have_finished_images
        ) % MPI.COMM_WORLD.size == 0:
            batch_size = (args.num_samples - have_finished_images) // MPI.COMM_WORLD.size
        else:
            batch_size = args.batch_size

        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]
        imgs = imgs.to('cuda:0')
        imgs = reshape_image(imgs, args.image_size)

        """
        Select t for different time step
        """
        recons = model(imgs, torch.tensor([t] * args.batch_size, device='cuda:0'))


        recons, _ = th.split(recons, 3, dim=1)

        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1)
        recons = recons.contiguous()

        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.contiguous()

        have_finished_images +=  batch_size

        recons = recons.cpu().numpy()
        for i in range(len(recons)):
            if args.has_subfolder:
                recons_save_dir = os.path.join(args.recons_dir, paths[i].split("/")[-2])
            else:
                recons_save_dir = args.recons_dir
            fn_save = os.path.basename(paths[i])
            os.makedirs(recons_save_dir, exist_ok=True)
            cv2.imwrite(f"{recons_save_dir}/{fn_save}", cv2.cvtColor(recons[i].astype(np.uint8), cv2.COLOR_RGB2BGR))
        logger.log(f"have finished {have_finished_images} samples")

    logger.log("finish computing feature!")


def create_argparser():
    defaults = dict(
        images_dir="data/adm/imagenet_ai_0508_adm/val/ai",
        recons_dir="data/val/adm/1_fake",
        time_step=0,
        clip_denoised=True,
        num_samples='',
        batch_size=5,
        use_ddim=False,
        model_path="checkpoint/256x256_diffusion_uncond.pt",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
