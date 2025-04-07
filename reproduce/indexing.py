from utils.io import fvecs_read, ivecs_read
from utils.preprocess import normalize
import symphonyqg
from time import time
from settings import EF, datasets, degrees, iter, datasets_dir

if __name__ == "__main__":

    for DATASET in datasets.keys():
        DISTANCE = datasets[DATASET]
        ITER = iter[DATASET]

        base = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"{datasets_dir}/{DATASET}/{DATASET}_query.fvecs")

        N, D = base.shape

        if DISTANCE == "angular":
            base = normalize(base)
            query = normalize(query)

        for DEGREE in degrees[DATASET]:
            index = symphonyqg.Index(
                index_type="QG",
                metric="L2",
                num_elements=N,
                dimension=D,
                degree_bound=DEGREE,
            )
            t1 = time()
            index.build_index(base, EF, num_iter=ITER, num_thread=32)
            t2 = time()
            print(f"The construction time for {DATASET}{DEGREE} is {t2-t1}")

            index_path = f"{datasets_dir}/{DATASET}/symphonyqg_{DEGREE}.index"

            index.save(index_path)
