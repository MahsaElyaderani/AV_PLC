import h5py
import numpy as np

def compute_landmark_stats(h5_file_paths, atol=0.0, verbose=False):
    zero_count = 0          # sequences whose landmarks are all (near-)zero
    empty_count = 0         # sequences with an empty landmarks array
    nan_only_count = 0      # sequences where all entries are NaN
    total = 0               # total sequences that had a landmarks dataset

    for h5_file_path in h5_file_paths:
        with h5py.File(h5_file_path, 'r+') as f:
            for key in f.keys():
                print(f[key].keys())
                dpath = f"{key}/landmarks"
                if dpath not in f:
                    # no landmarks for this item; skip or count separately if you want
                    continue

                x = f[dpath][()]  # read the whole dataset
                total += 1

                if x.size == 0:
                    empty_count += 1
                    if verbose:
                        print(f"[EMPTY] {h5_file_path} :: {key}")
                    continue

                # Handle NaNs explicitly
                if np.isnan(x).all():
                    nan_only_count += 1
                    if verbose:
                        print(f"[ALL-NAN] {h5_file_path} :: {key}")
                    continue

                if atol == 0.0:
                    is_zero = (x == 0).all()

                if is_zero:
                    #delete the entire data with that key is it is all zero
                    #del f[key]
                    zero_count += 1
                    if verbose:
                        print(f"[ALL-ZERO] {h5_file_path} :: {key}")

    return {
        "zero_count": zero_count,
        "empty_count": empty_count,
        "nan_only_count": nan_only_count,
        "total": total,
        "zero_pct": 100.0 * zero_count / total if total else 0.0
    }

if __name__ == "__main__":
    import glob
    h5_files = glob.glob("/home/amin/Projects/Mahsa/datasets/lrs2/lrs2_test_features_chunk*.h5")
    stats = compute_landmark_stats(h5_files, atol=0.0, verbose=False)
    print(
        f"Total sequences with landmarks: {stats['total']}\n"
        f"All-zero sequences: {stats['zero_count']} ({stats['zero_pct']:.6f}%)\n"
        f"Empty arrays: {stats['empty_count']}\n"
        f"All-NaN arrays: {stats['nan_only_count']}"
    )
