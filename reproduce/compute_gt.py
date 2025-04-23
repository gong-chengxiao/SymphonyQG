import numpy as np
from utils.preprocess import normalize
from utils.io import fvecs_read, ivecs_write
from settings import datasets_dir
from multiprocessing import Pool, shared_memory
import os

datasets = ["gist", "msong"]
dis_type = "l2"

def compute_gt_single_query(args):
    q, shm_name, base_shape, base_dtype = args
    shm = shared_memory.SharedMemory(name=shm_name)
    base = np.ndarray(base_shape, dtype=base_dtype, buffer=shm.buf)
    distances = np.linalg.norm(base - q, axis=1)
    shm.close()
    return np.argsort(distances)[:1000]

if __name__ == "__main__":
    for DATASET in datasets:
        print(f"Computing groundtruth for {DATASET}")
        base = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_query.fvecs")

        if dis_type == "angular":
            base = normalize(base)
            query = normalize(query)
        
        try:
            shm = shared_memory.SharedMemory(create=True, size=base.nbytes)
            base_shared = np.ndarray(base.shape, dtype=base.dtype, buffer=shm.buf)
            base_shared[:] = base[:]

            gt = []
            num_processes = 16
            print(f"Using {num_processes} processes")
            with Pool(processes=num_processes) as pool:
                query_args = [(q, shm.name, base.shape, base.dtype) for q in query]
                gt = pool.map(compute_gt_single_query, query_args)

            gt = np.array(gt)

        finally:
            shm.close()
            shm.unlink()

        ivecs_write(f"{datasets_dir}/{DATASET}/{DATASET}_groundtruth.ivecs", gt)
