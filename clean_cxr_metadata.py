import os
import pandas as pd
from pathlib import Path

meta_data_path = '/home/aamer98/scratch/data/subpopbench/MIMIC-CXR-JPG/physionet.org/files/mimic-cxr-jpg/2.0.0/subpop_bench_meta/metadata_multisite.csv'

df = pd.read_csv(meta_data_path)

for index, row in df.iterrows():
    image_dir = Path(row['filename'])
    if not image_dir.exists():
        print(f"Image {image_dir} does not exist")
        df.drop(index, inplace=True)

# breakpoint()
df.to_csv(meta_data_path, index=False)