import os
import trimesh
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
from rl_dataset import CadrilleSTLDataset
from metrics_async import transform_real_mesh
from qwen_vl_utils import smart_resize


#### instead of what is done inside process_vision_info of qwen vl utils to match image processing-------
IMAGE_FACTOR = 28
MIN_PIXELS   = 4 * 28 * 28
MAX_PIXELS   = 16384 * 28 * 28

def qwen_align(img):
    h, w = img.height, img.width
    h2, w2 = smart_resize(h, w, factor=IMAGE_FACTOR,
                        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if (h2, w2) != (h, w):
        img = img.resize((w2, h2))
    return img

#### --------------------------------------------------

def render_one(ex):  
    stl_path = ex["mesh_path"]
    try:
        mesh = trimesh.load_mesh(stl_path)
        mesh = transform_real_mesh(mesh)
        out = train_data.get_img(mesh)
        video = out["video"]
        video = [qwen_align(frame) for frame in video]
        return {"video": video}
    except Exception as e:
        print(f"Render failed for {stl_path}: {e}")
        return {"video": None}
        

def convert_to_hf_dataset(train_data, destination_path):
    #train_data = train_combined
    INSTR = "Generate cadquery code"
    mesh_paths = [os.path.join(train_data.path, train_data.annotations[a]['mesh_path']) for a in train_data.annotations]
    base = Dataset.from_list(
        [{"mesh_path": mp, "instruction": INSTR} for mp in mesh_paths],
        features=Features({
            "mesh_path": Value("string"),
            "instruction": Value("string"),
        })
    )

    features = Features({
        "mesh_path": Value("string"),
        "instruction": Value("string"),
        "video": Sequence(HFImage())
    })

    ds = base.map(
        render_one,
        num_proc=156,
        features=features,
        writer_batch_size=64,
        #keep_in_memory=True
    ).filter(lambda ex: ex["video"] is not None)

    ds.save_to_disk(destination_path)

if __name__ == "__main__":
    train_data = CadrilleSTLDataset(path=f'/home/jovyan/users/zhemchuzhnikov/tarasov/data/deepcad_fusion_train', file_name="train_small.pkl", n_points=256, mode="img", noise_scale_pc=0.01, size=None)
    destination_path = "./rendered_1view_deepcad_all"
    convert_to_hf_dataset(train_data, destination_path)