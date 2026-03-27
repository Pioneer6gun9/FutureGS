import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
from arguments import ModelParams, get_combined_args


def read_images(renders_dir, gt_dir):
    """Read predicted images and ground truth images"""
    renders = []
    gts = []
    image_names = []

    gt_files = {}
    render_files = {}

    for f in os.listdir(gt_dir):
        if f.startswith("gt_cam") and f.endswith(".png"):
            try:
                cam_id = f.split("gt_cam")[1].split(".png")[0]
                gt_files[cam_id] = f
            except:
                continue

    for f in os.listdir(renders_dir):
        if f.startswith("pred_cam") and f.endswith(".png"):
            try:
                cam_id = f.split("pred_cam")[1].split(".png")[0]
                render_files[cam_id] = f
            except:
                continue

    common_ids = set(gt_files.keys()) & set(render_files.keys())
    if not common_ids:
        raise ValueError("No matching camera IDs found")

    sorted_ids = sorted(common_ids, key=lambda x: int(x))

    for cam_id in sorted_ids:
        gt_path = Path(gt_dir) / gt_files[cam_id]
        render_path = Path(renders_dir) / render_files[cam_id]

        try:
            gt_img = Image.open(gt_path).convert("RGB")
            render_img = Image.open(render_path).convert("RGB")

            gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
            render_tensor = tf.to_tensor(render_img).unsqueeze(0)[:, :3, :, :].cuda()

            gts.append(gt_tensor)
            renders.append(render_tensor)
            image_names.append(f"cam_{cam_id}")

        except Exception as e:
            print(f"Error loading camera {cam_id}: {e}")
            continue

    print(f"Loaded {len(image_names)} image pairs")
    return renders, gts, image_names


def evaluate(model_paths):
    """Compute temporal prediction metrics"""
    device = torch.device("cuda:0")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    for model_path in model_paths:
        try:

            base_dir = Path(model_path) / "temporal" / "predictions"
            renders_dir = base_dir / "renders"
            gt_dir = base_dir / "gt"

            renders, gts, image_names = read_images(renders_dir, gt_dir)

            if len(renders) == 0:
                raise ValueError("No valid images found")

            psnrs = []
            ssims = []
            lpipss = []
            image_metrics = {}

            for idx in tqdm(range(len(renders)), desc="Calculating metrics"):
                p = psnr(renders[idx], gts[idx])
                s = ssim(renders[idx], gts[idx])
                l = lpips_fn(renders[idx], gts[idx]).detach()

                psnrs.append(p)
                ssims.append(s)
                lpipss.append(l)

                image_metrics[image_names[idx]] = {
                    "PSNR": round(p.item(), 4),
                    "SSIM": round(s.item(), 4),
                    "LPIPS": round(l.item(), 4)
                }

            avg_ssim = torch.stack(ssims).mean().item()
            avg_psnr = torch.stack(psnrs).mean().item()
            avg_lpips = torch.stack(lpipss).mean().item()

            print(f"\nFinal Metrics:")
            print(f"  PSNR:  {avg_psnr:.4f}")
            print(f"  SSIM:  {avg_ssim:.4f}")
            print(f"  LPIPS: {avg_lpips:.4f}")

            # Save results
            result = {
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "LPIPS": avg_lpips
            }

            result_file = base_dir / "results.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            per_view_file = base_dir / "per_view.json"
            with open(per_view_file, 'w') as fp:
                fp.write('{\n')
                entries = []
                for img_name, metrics_dict in image_metrics.items():
                    entry = f'  "{img_name}": {json.dumps(metrics_dict, separators=(",", ":"))}'
                    entries.append(entry)
                fp.write(',\n'.join(entries))
                fp.write('\n}')

            print(f"Results saved to {base_dir}")

        except Exception as e:
            print(f"\nError processing {model_path}:")
            print(f"  {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate temporal predictions")
    model = ModelParams(parser)

    parser.add_argument("--model_paths", nargs="+", type=str, required=True)

    args = parser.parse_args()

    evaluate(args.model_paths)
