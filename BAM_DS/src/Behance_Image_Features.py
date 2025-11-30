import struct
import pandas as pd
import os
from Behance_appreciate_1M import dir_path

RAW = fr"{dir_path}\data_raw\Behance_Image_Features.b"
OUT = fr"{dir_path}\data_processed\image_features.parquet"

BATCH_SIZE = 1500

def run():
    print("Reading large image feature file... streaming mode active")

    out_dir = os.path.dirname(OUT)
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(OUT):
        os.remove(OUT)

    f = open(RAW, "rb")

    item_ids = []
    features = []
    batch_num = 0
    first_write = True

    while True:

        # Read item ID
        item_id_raw = f.read(8)
        if not item_id_raw:
            break

        item_id = item_id_raw.decode("utf-8", errors="ignore")

        # Read 4096 float bytes
        raw = f.read(4 * 4096)

        # FIX: if fewer bytes remain → end of file
        if len(raw) < 4 * 4096:
            break

        feature_vector = struct.unpack("f" * 4096, raw)

        item_ids.append(item_id)
        features.append(feature_vector)

        if len(item_ids) == BATCH_SIZE:

            df = pd.DataFrame(features)
            df.insert(0, "item_id", item_ids)
            df.columns = ["item_id"] + [f"f_{i}" for i in range(4096)]

            if first_write:
                df.to_parquet(OUT)
                first_write = False
            else:
                df.to_parquet(OUT, append=True)

            print(f"Written batch {batch_num}")

            batch_num += 1
            item_ids = []
            features = []

    # Write leftovers
    if item_ids:
        df = pd.DataFrame(features)
        df.insert(0, "item_id", item_ids)
        df.columns = ["item_id"] + [f"f_{i}" for i in range(4096)]

        if first_write:
            df.to_parquet(OUT)
        else:
            df.to_parquet(OUT, append=True)

        print(f"Written final batch {batch_num}")

    f.close()
    print("DONE — Features successfully extracted.")

if __name__ == "__main__":
    run()
