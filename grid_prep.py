import os
import shutil


def reduce_name(root):
    for folder in os.listdir(root):
        old_path = os.path.join(root, folder)

        # Check if it's a directory and matches the pattern
        if os.path.isdir(old_path) and folder.endswith("_processed") and folder.startswith("s"):
            new_name = folder.replace("_processed", "", 1)  # remove only the first occurrence
            new_path = os.path.join(root, new_name)

            print(f"Renaming: {old_path} -> {new_path}")
            os.rename(old_path, new_path)


def split_dataset(root):
    dry_run = False  # set True to preview without moving

    splits = {
        "val": ['s26', 's27', 's29', 's31'],
        "test": ['s30', 's32', 's33', 's34'],
        "train": ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                  's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                  's18', 's19', 's20', 's22', 's23', 's24', 's25', 's28'],
    }

    # Make split folders
    for subset in ("train", "val", "test"):
        os.makedirs(os.path.join(root, subset), exist_ok=True)

    def move_or_merge_dir(src_path, dst_path):
        """
        Move a whole directory. If destination exists, merge contents.
        """
        if not os.path.exists(dst_path):
            print(f"MOVE  {src_path}  ->  {dst_path}")
            if not dry_run:
                shutil.move(src_path, dst_path)
            return

        # Destination exists: merge contents (files and subfolders)
        print(f"MERGE {src_path}  ->  {dst_path}")
        if not dry_run:
            for name in os.listdir(src_path):
                s = os.path.join(src_path, name)
                d = os.path.join(dst_path, name)
                if os.path.exists(d):
                    # If conflict: skip or handle as you wish (here we skip)
                    print(f"  SKIP (exists): {d}")
                    continue
                shutil.move(s, d)
            # Try to remove the now-empty source dir
            try:
                os.rmdir(src_path)
            except OSError:
                pass  # not empty or some issue; leave it

    for subset, speakers in splits.items():
        subset_dir = os.path.join(root, subset)
        for speaker in speakers:
            src = os.path.join(root, speaker)
            dst = os.path.join(subset_dir, speaker)

            if not os.path.isdir(src):
                print(f"MISS  {src} (not found or not a directory) — skipping")
                continue

            move_or_merge_dir(src, dst)


if __name__ == "__main__":
    root = '/Users/kadkhodm/PycharmProjects/speech_inpainting/datasets/grid'
    reduce_name(root)
    split_dataset(root)
