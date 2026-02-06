from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch

import json
from functools import partial

from cadrille import Cadrille, optimize_model_memory
from transformers import AutoProcessor

# dataset without labels
from dataset import CadrilleSTLDataset
import argparse
from cadrille import collate
from metrics_async import init_pool, get_metrics_from_texts

TEMPERATURE = 0.1
TOP_P = 0.99
TOP_K = None
DO_SAMPLE = True
MAX_NEW_TOKENS = 768


def evaluate_model_mm(model, processor, eval_examples, collate_fn, batch_size=8, normalize="fixed"):
    model.eval()
    print("\n" + "=" * 50)
    print("EVALUATION ON", len(eval_examples), "EXAMPLES")
    print("=" * 50)

    dataloader = DataLoader(
        eval_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    ious, cds = [], []
    n_incorrect, n_failed_intersect = 0, 0

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            generated_ids = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                point_clouds=batch["point_clouds"].to(model.device),
                is_pc=batch["is_pc"].to(model.device),
                is_img=batch["is_img"].to(model.device),
                pixel_values_videos=(batch["pixel_values_videos"].to(model.device)
                                    if batch.get("pixel_values_videos") is not None else None),
                video_grid_thw=(batch["video_grid_thw"].to(model.device)
                                if batch.get("video_grid_thw") is not None else None),
                max_new_tokens=MAX_NEW_TOKENS,
                bad_words_ids=[[model.config.video_token_id]],
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                top_k=TOP_K,
                early_stopping=False,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch["input_ids"], generated_ids)
            ]
            decoded_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            pred_metrics = get_metrics_from_texts(
                decoded_texts, batch["mesh_path"], var_name='r', normalize=normalize
            )

            for m in pred_metrics:
                if m is None or m.get("iou") is None or m.get("cd") is None:
                    n_incorrect += 1
                    continue
                if m["iou"] < 0:
                    n_failed_intersect += 1
                    continue
                ious.append(m["iou"])
                cds.append(m["cd"])

    def _mn(x): return float(np.mean(x)) if len(x) else float("nan")
    def _md(x): return float(np.median(x)) if len(x) else float("nan")

    print(f"IoU mean {_mn(ious)}, median {_md(ious)}")
    print(f"CD mean {_mn(cds)}, median {_md(cds)}")
    print(f"Invalid generations fraction: {n_incorrect / len(eval_examples):.6f}")
    print(f"Intersect failure fraction: {n_failed_intersect / len(eval_examples):.6f}")
    print("=" * 50)

    model.train()
    return {
        "ious": ious,
        "cds": cds,
        "invalid_frac": n_incorrect / len(eval_examples),
        "intersect_fail_frac": n_failed_intersect / len(eval_examples),
    }



def main(model_path: str, normalize: str = "fixed"):
    print("Starting evaluation")

    rank = 0
    torch.cuda.set_device(rank)
    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None

    model = Cadrille.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=rank
    ).train().to(device=torch.device(f"cuda:{rank}"))

    model = optimize_model_memory(model)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left",
                                              )

    eval_data_deepcad = CadrilleSTLDataset(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_test', file_name='test.pkl', n_points=256, size=1000)
    eval_data_fusion = CadrilleSTLDataset(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/fusion360_test', file_name='test.pkl', n_points=256, size=1000)


    print(f"Rank {rank}: Initializing datasets")

    collate_fn = partial(collate, processor=processor, n_points=256, eval=True)
    print("Initializing worker pool")
    init_pool(20)


    # ---- IMG ----
    eval_data_deepcad.mode = 'img'
    eval_data_fusion.mode = 'img'
    res_img_deepcad = evaluate_model_mm(
        model, processor, eval_data_deepcad, collate_fn,
        batch_size=500, normalize=normalize
    )
    res_img_fusion = evaluate_model_mm(
        model, processor, eval_data_fusion, collate_fn,
        batch_size=500, normalize=normalize
    )

    # ---- PC ----
    eval_data_deepcad.mode = 'pc'
    eval_data_fusion.mode = 'pc'
    res_pc_deepcad = evaluate_model_mm(
        model, processor, eval_data_deepcad, collate_fn,
        batch_size=500, normalize=normalize
    )
    res_pc_fusion = evaluate_model_mm(
        model, processor, eval_data_fusion, collate_fn,
        batch_size=500, normalize=normalize
    )

    def mn(x): return float(np.mean(x)) if len(x) else float("nan")
    def md(x): return float(np.median(x)) if len(x) else float("nan")

    metrics = {
        # Point‐cloud DeepCAD
        "eval/pc/DeepCAD/IoU mean":   mn(res_pc_deepcad["ious"]),
        "eval/pc/DeepCAD/CD mean":    mn(res_pc_deepcad["cds"]),
        "eval/pc/DeepCAD/IoU median": md(res_pc_deepcad["ious"]),
        "eval/pc/DeepCAD/CD median":  md(res_pc_deepcad["cds"]),
        "eval/pc/DeepCAD/Invalid frac":  res_pc_deepcad["invalid_frac"],
        "eval/pc/DeepCAD/Intersect fail frac": res_pc_deepcad["intersect_fail_frac"],

        # Point‐cloud Fusion360
        "eval/pc/Fusion360/IoU mean":   mn(res_pc_fusion["ious"]),
        "eval/pc/Fusion360/CD mean":    mn(res_pc_fusion["cds"]),
        "eval/pc/Fusion360/IoU median": md(res_pc_fusion["ious"]),
        "eval/pc/Fusion360/CD median":  md(res_pc_fusion["cds"]),
        "eval/pc/Fusion360/Invalid frac":  res_pc_fusion["invalid_frac"],
        "eval/pc/Fusion360/Intersect fail frac": res_pc_fusion["intersect_fail_frac"],

        # Image DeepCAD
        "eval/img/DeepCAD/IoU mean":   mn(res_img_deepcad["ious"]),
        "eval/img/DeepCAD/CD mean":    mn(res_img_deepcad["cds"]),
        "eval/img/DeepCAD/IoU median": md(res_img_deepcad["ious"]),
        "eval/img/DeepCAD/CD median":  md(res_img_deepcad["cds"]),
        "eval/img/DeepCAD/Invalid frac":  res_img_deepcad["invalid_frac"],
        "eval/img/DeepCAD/Intersect fail frac": res_img_deepcad["intersect_fail_frac"],

        # Image Fusion360
        "eval/img/Fusion360/IoU mean":   mn(res_img_fusion["ious"]),
        "eval/img/Fusion360/CD mean":    mn(res_img_fusion["cds"]),
        "eval/img/Fusion360/IoU median": md(res_img_fusion["ious"]),
        "eval/img/Fusion360/CD median":  md(res_img_fusion["cds"]),
        "eval/img/Fusion360/Invalid frac":  res_img_fusion["invalid_frac"],
        "eval/img/Fusion360/Intersect fail frac": res_img_fusion["intersect_fail_frac"],
    }

    print("\n==== FINAL METRICS ====")
    print(json.dumps(metrics, indent=2))
    print("=======================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path or hub id for Cadrille weights")
    parser.add_argument("--normalize", type=str, default="fixed", help="Normalization of stl meshes, can be 'fixed' or 'mesh_extents'")
    args = parser.parse_args()
    main(args.model_path, args.normalize)