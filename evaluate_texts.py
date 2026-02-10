import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

from metrics_async import init_pool, get_metrics_from_texts


def run(gt_mesh_path, pred_py_path, n_points):
    best_names_path = os.path.join(os.path.dirname(pred_py_path), 'tmp.txt')
    init_pool(20)

    # compute chamfer distance and iou for each sample
    py_file_names = os.listdir(pred_py_path)
    decoded_texts, mesh_paths, file_names, ids = [], [], [], []
    for py_file_name in py_file_names:
        eval_file_name = py_file_name[:py_file_name.rfind('+')]
        index = py_file_name[len(eval_file_name) + 1:-3]
        file_names.append(eval_file_name)
        ids.append(index)
        with open(os.path.join(pred_py_path, py_file_name), 'r') as f:
            decoded_texts.append(f.read())
        mesh_paths.append(os.path.join(gt_mesh_path, eval_file_name + '.stl'))

    pred_metrics = get_metrics_from_texts(
        decoded_texts, mesh_paths, max_workers=16, var_name="r"
    )
    # aggregate metrics per eval_file_name
    py_metrics = [
        dict(file_name=fn, id=idx,
            cd=(m.get("cd") if m else None),
            iou=(m.get("iou") if m else None))
        for fn, idx, m in zip(file_names, ids, pred_metrics)
    ]

    
    # select best metrics per eval_file_name
    ir_cd, ir_iou, cd, iou, best_names = 0, 0, list(), list(), list()
    for key, value in metrics.items():
        if len(value['cd']):
            argmin = np.argmin(value['cd'])
            cd.append(value['cd'][argmin])
            index = value['id'][argmin]
            best_names.append(f'{key}+{index}.py')
        else:
            ir_cd += 1
        
        if len(value['iou']):
            iou.append(np.max(value['iou']))
        else:
            ir_iou += 1

    with open(best_names_path, 'w') as f:
        f.writelines([line + '\n' for line in best_names])

    print(f'mean iou: {np.mean(iou):.3f}',
          f'median cd: {np.median(cd) * 1000:.3f}')

    cd = sorted(cd)
    for i in range(5):
        print(f'skip: {i} ir: {(ir_cd + i) / len(metrics) * 100:.2f}',
              f'mean cd: {np.mean(cd[:len(cd) - i]) * 1000:.3f}')


# The Pool is tweaked to support non-daemon processes that can
# call one more nested process.
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt-mesh-path', type=str, default='/home/jovyan/users/zhemchuzhnikov/kolodiazhnyi/data/deepcad_test_mesh')
    parser.add_argument('--pred-py-path', type=str, default='/home/jovyan/users/zhemchuzhnikov/kolodiazhnyi/work_dirs/tmp/tmp_py')
    parser.add_argument('--n-points', type=int, default=8192)
    args = parser.parse_args()
    run(args.gt_mesh_path, args.pred_py_path, args.n_points)
