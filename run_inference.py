#!/usr/bin/env python3
"""UltraVSR inference — accepts video files (via PyAV) or frame directories."""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.append(os.getcwd())

import argparse
import glob
import math
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.ram.models.ram_lora import ram
from models.ram import inference_ram as inference
from models.inference_arch import UltraVSR_test
from wavelet_color_fix import adain_color_fix, wavelet_color_fix

# ---------------------------------------------------------------------------
# PyAV frame extraction
# ---------------------------------------------------------------------------

# Map CLI names to Pillow resampling filters (used for chroma upsampling)
RESAMPLE_METHODS = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}

# Map CLI names to PyAV Interpolation enum names (uppercase)
SWS_METHODS = {
    "lanczos": "LANCZOS",
    "bicubic": "BICUBIC",
    "bilinear": "BILINEAR",
    "nearest": "POINT",
    "spline": "SPLINE",
    "area": "AREA",
}


def extract_frames_av(video_path: str, chroma_resampling: str = "lanczos"):
    """Decode every video frame to an RGB PIL Image using PyAV.

    PyAV's reformatting uses libswscale under the hood.  We pick the
    best available scaler so that chroma planes (typically 4:2:0 YUV)
    are upsampled with a high-quality filter before conversion to RGB.
    """
    sws_flag = SWS_METHODS.get(chroma_resampling, "lanczos")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            rgb_frame = frame.reformat(format="rgb24", interpolation=sws_flag)
            img = Image.fromarray(rgb_frame.to_ndarray())
            frames.append(img)
    container.close()
    return frames


def load_frames_from_directory(frame_dir: str):
    """Load PNG/JPG frames from a directory, sorted by name."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(frame_dir, ext)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No image files found in {frame_dir}")
    return [Image.open(f).convert("RGB") for f in files], [os.path.basename(f) for f in files]


# ---------------------------------------------------------------------------
# Helpers (same logic as original inference.py)
# ---------------------------------------------------------------------------

tensor_transforms = transforms.Compose([transforms.ToTensor()])
ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_validation_prompt(image, dape_model, weight_dtype, device="cuda"):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, dape_model)
    return captions[0], lq


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="UltraVSR video super-resolution inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / Output ────────────────────────────────────────────────
    io = p.add_argument_group("Input / Output")
    io.add_argument(
        "-i", "--input", required=True,
        help="Path to a video file (.mp4, .mkv, …) or a directory of "
             "frame directories (original format).",
    )
    io.add_argument(
        "-o", "--output_dir", default="./results",
        help="Directory where upscaled frames are saved.",
    )

    # ── Model weights ─────────────────────────────────────────────────
    wt = p.add_argument_group("Model weights")
    wt.add_argument(
        "--model_path", default="./pretrained/model_vsr.pkl",
        help="Path to the UltraVSR checkpoint (.pkl).",
    )
    wt.add_argument(
        "--diffusion_path", default="./weights/stable-diffusion-2-1-base/",
        help="Path to the Stable Diffusion 2.1-base model directory.",
    )
    wt.add_argument(
        "--ram_path", default="./weights/ram_swin_large_14m.pth",
        help="Path to the RAM (Recognize Anything Model) weights.",
    )
    wt.add_argument(
        "--dape_path", default="./weights/DAPE.pth",
        help="Path to the DAPE (RAM fine-tuned) weights.",
    )

    # ── Processing ────────────────────────────────────────────────────
    proc = p.add_argument_group("Processing")
    proc.add_argument("--upscale", type=int, default=4, help="Upscale factor.")
    proc.add_argument("--process_size", type=int, default=512,
                       help="Minimum dimension for processing.")
    proc.add_argument("--max_process_size", type=int, default=1536,
                       help="Maximum dimension for processing.")
    proc.add_argument("--max_process_size_per_interval", type=int, default=1024,
                       help="Max spatial size before sequence splitting.")
    proc.add_argument("--max_process_interval", type=int, default=15,
                       help="Max frames per interval for high-res inputs.")
    proc.add_argument("--seed", type=int, default=42, help="Random seed.")
    proc.add_argument(
        "--align_method", choices=["wavelet", "adain", "nofix"], default="wavelet",
        help="Colour alignment method applied to output frames.",
    )
    proc.add_argument(
        "--no_resize_back", action="store_true",
        help="Do NOT resize the output back to upscale × original size.",
    )

    # ── Chroma upsampling ─────────────────────────────────────────────
    chroma = p.add_argument_group("Chroma upsampling (YUV → RGB)")
    chroma.add_argument(
        "--chroma_resampling", choices=list(SWS_METHODS.keys()),
        default="lanczos",
        help="Resampling filter used when converting YUV chroma planes to RGB.",
    )

    # ── Precision ─────────────────────────────────────────────────────
    prec = p.add_argument_group("Precision")
    prec.add_argument(
        "--precision", choices=["fp16", "fp32"], default="fp16",
        help="Mixed-precision mode (fp16 is faster, fp32 is more accurate).",
    )
    prec.add_argument(
        "--merge_lora", action="store_true", default=True,
        help="Merge and unload LoRA weights (saves VRAM and speeds up "
             "inference). Use --no_merge_lora to disable.",
    )
    prec.add_argument(
        "--no_merge_lora", action="store_false", dest="merge_lora",
        help="Keep LoRA weights separate (uses more VRAM, slower).",
    )

    # ── Tiling / VRAM ─────────────────────────────────────────────────
    tile = p.add_argument_group("Tiling / VRAM (tuned for RTX 4090 24GB)")
    tile.add_argument("--chunk_size", type=int, default=6,
                       help="Number of frames to load and process at a time. "
                            "Controls peak CPU/GPU memory. Lower = less RAM. "
                            "Keep ≥3 for temporal coherence.")
    tile.add_argument("--vae_encoder_tiled_size", type=int, default=1600,
                       help="Encoder skips tiling when H*W ≤ this². Set high "
                            "enough to avoid tiling seams at your resolution.")
    tile.add_argument("--vae_encoder_overlap", type=int, default=32)
    tile.add_argument("--vae_encoder_num_frames_per_batch", type=int, default=1)
    tile.add_argument("--vae_decoder_tiled_size", type=int, default=200,
                       help="Decoder skips tiling when latent H*W ≤ this². "
                            "Latent is input÷8, e.g. 1536→192.")
    tile.add_argument("--vae_decoder_overlap", type=int, default=8)
    tile.add_argument("--vae_decoder_num_frames_per_batch", type=int, default=1)
    tile.add_argument("--latent_tiled_size", type=int, default=200,
                       help="UNet skips tiling when latent H*W ≤ this².")
    tile.add_argument("--latent_tiled_overlap", type=int, default=16)
    tile.add_argument("--latent_num_frames_per_batch", type=int, default=1)

    return p


# ---------------------------------------------------------------------------
# Adapt parsed args so the existing UltraVSR_test class sees what it expects
# ---------------------------------------------------------------------------

def adapt_args(args):
    """Map our friendlier CLI names to the attribute names UltraVSR_test expects."""
    args.UltraVSR_weight_path = args.model_path
    args.pretrained_diffusion_model_path = args.diffusion_path
    args.ram_ft_path = args.dape_path
    args.mixed_precision = args.precision
    args.merge_and_unload_lora = args.merge_lora
    args.resize_back = not args.no_resize_back
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_sequence(frames, frame_names, args, model, dape_model, weight_dtype, output_subdir):
    """Run UltraVSR on a list of PIL frames and save results.

    Frames are processed in chunks of ``args.chunk_size`` to avoid loading
    all frames into GPU memory at once (critical for long videos).
    """
    os.makedirs(output_subdir, exist_ok=True)
    total = len(frames)
    chunk_size = args.chunk_size
    resample_filter = RESAMPLE_METHODS[args.chroma_resampling]

    print(f"Processing {total} frames in chunks of {chunk_size} → {output_subdir}")

    # --- Compute degradation factor from a small sample of frames ----
    sample_indices = np.linspace(0, total - 1, num=min(50, total), dtype=int)
    sample_lqs = []
    for si in sample_indices:
        img = frames[si]
        lq = tensor_transforms(img).unsqueeze(0).to("cuda")
        sample_lqs.append(lq)
        if len(sample_lqs) >= 5:
            break
    sample_lqs = torch.cat(sample_lqs, 0)
    with torch.no_grad():
        degra_factor = model.metric_model(sample_lqs)
        degra_factor = torch.mean(degra_factor) ** 2
    del sample_lqs
    torch.cuda.empty_cache()
    print(f"Degradation factor: {degra_factor.item():.6f}")

    # --- Process frames in chunks ------------------------------------
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_frames = frames[chunk_start:chunk_end]
        chunk_names = frame_names[chunk_start:chunk_end]
        print(f"\n  Chunk [{chunk_start+1}–{chunk_end}] / {total}")

        lqs, input_images, validation_prompts, bnames = [], [], [], []

        for idx, input_image in enumerate(chunk_frames):
            ori_width, ori_height = input_image.size

            if ori_width < args.process_size // args.upscale or ori_height < args.process_size // args.upscale:
                scale = args.process_size / min(ori_width, ori_height)
                input_image = input_image.resize(
                    (int(scale * ori_width), int(scale * ori_height)), resample_filter)
                resize_flag = True
            elif ori_width > args.max_process_size // args.upscale or ori_height > args.max_process_size // args.upscale:
                scale = args.max_process_size / max(ori_width, ori_height)
                input_image = input_image.resize(
                    (int(scale * ori_width), int(scale * ori_height)), resample_filter)
                resize_flag = True
            else:
                input_image = input_image.resize(
                    (ori_width * args.upscale, ori_height * args.upscale), resample_filter)
                resize_flag = False

            new_w = input_image.width - input_image.width % 8
            new_h = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)

            input_images.append(input_image)
            bnames.append(chunk_names[idx])

            prompt, lq = get_validation_prompt(input_image, dape_model, weight_dtype)
            validation_prompts.append(prompt)
            lqs.append(lq * 2 - 1)

        # Pad to multiple of latent_num_frames_per_batch
        num_real = len(lqs)
        if num_real % args.latent_num_frames_per_batch > 0:
            res_num = args.latent_num_frames_per_batch - num_real % args.latent_num_frames_per_batch
            for _ in range(res_num):
                lqs.append(lqs[-1])
                validation_prompts.append(validation_prompts[-1])

        lqs = torch.cat(lqs, 0)

        # Inference
        with torch.no_grad():
            output_images = model(lqs, degra_factor, prompts=validation_prompts)

            for i in range(num_real):
                output_pil = transforms.ToPILImage()(output_images[i].cpu() * 0.5 + 0.5)
                output_pil = _apply_color_fix(output_pil, input_images[i], args.align_method)
                if args.resize_back and resize_flag:
                    output_pil = output_pil.resize(
                        (int(args.upscale * ori_width), int(args.upscale * ori_height)),
                        resample_filter)
                output_pil.save(os.path.join(output_subdir, bnames[i]))

        del lqs, output_images, input_images
        torch.cuda.empty_cache()


def _apply_color_fix(target, source, method):
    if method == "adain":
        return adain_color_fix(target=target, source=source)
    elif method == "wavelet":
        return wavelet_color_fix(target=target, source=source)
    return target


def main():
    parser = build_parser()
    args = adapt_args(parser.parse_args())
    input_path = Path(args.input)

    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    # ── Load models ───────────────────────────────────────────────────
    print("Loading UltraVSR model …")
    model = UltraVSR_test(args)

    print("Loading DAPE / RAM …")
    dape_model = ram(
        pretrained=args.ram_path,
        pretrained_condition=args.dape_path,
        image_size=384,
        vit="swin_l",
    )
    dape_model.eval()
    dape_model.to("cuda")

    weight_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    dape_model = dape_model.to(dtype=weight_dtype)

    # ── Determine input type ──────────────────────────────────────────
    if input_path.is_file():
        # Single video file
        print(f"Extracting frames from {input_path} (chroma resampling: {args.chroma_resampling}) …")
        frames = extract_frames_av(str(input_path), args.chroma_resampling)
        frame_names = [f"{i:06d}.png" for i in range(len(frames))]
        seq_name = input_path.stem
        process_sequence(
            frames, frame_names, args, model, dape_model, weight_dtype,
            os.path.join(args.output_dir, seq_name),
        )
    elif input_path.is_dir():
        VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts")
        IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

        def _find_videos(d):
            return sorted(f for f in d.iterdir()
                          if f.is_file() and f.suffix.lower() in VIDEO_EXTS)

        def _has_images(d):
            return any(f.suffix.lower() in IMAGE_EXTS
                       for f in d.iterdir() if f.is_file())

        subdirs = sorted(d for d in input_path.iterdir() if d.is_dir())
        top_videos = _find_videos(input_path)

        if subdirs:
            print(f"Found {len(subdirs)} subdirectory(ies) in {input_path}")
            for subdir in subdirs:
                # Each subdir may contain images (frame dir) or videos
                sub_videos = _find_videos(subdir)
                if _has_images(subdir):
                    frames, frame_names = load_frames_from_directory(str(subdir))
                    process_sequence(
                        frames, frame_names, args, model, dape_model, weight_dtype,
                        os.path.join(args.output_dir, subdir.name),
                    )
                elif sub_videos:
                    for vf in sub_videos:
                        print(f"\n{'='*60}\n{vf.name}\n{'='*60}")
                        frames = extract_frames_av(str(vf), args.chroma_resampling)
                        frame_names = [f"{i:06d}.png" for i in range(len(frames))]
                        process_sequence(
                            frames, frame_names, args, model, dape_model, weight_dtype,
                            os.path.join(args.output_dir, vf.stem),
                        )
                else:
                    print(f"  Skipping {subdir.name} (no images or videos found)")
        elif top_videos:
            # Directory of video files
            print(f"Found {len(top_videos)} video(s) in {input_path}")
            for vf in top_videos:
                print(f"\n{'='*60}\n{vf.name}\n{'='*60}")
                frames = extract_frames_av(str(vf), args.chroma_resampling)
                frame_names = [f"{i:06d}.png" for i in range(len(frames))]
                process_sequence(
                    frames, frame_names, args, model, dape_model, weight_dtype,
                    os.path.join(args.output_dir, vf.stem),
                )
        elif _has_images(input_path):
            # Flat directory of images
            frames, frame_names = load_frames_from_directory(str(input_path))
            process_sequence(
                frames, frame_names, args, model, dape_model, weight_dtype,
                os.path.join(args.output_dir, input_path.name),
            )
        else:
            parser.error(f"No images or videos found in {input_path}")
    else:
        parser.error(f"Input is not a file or directory: {input_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
