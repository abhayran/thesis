import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pyhist import utility_functions, parser_input
from pyhist.slide import PySlide, TileGenerator


ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"
METHOD = "graph"
PATCH_SIZE = "256"
OUTPUT_DOWNSAMPLE = "2"
CONTENT_THRESHOLD = "0.05"


def main():
    # read in dataframes
    dfs = {
        key: pd.concat([
            pd.read_csv(os.path.join(ORTHOPEDIA_DIR, "csv_files", f"{key}_{split}.csv")) 
            for split in ["train", "val", "test"]
        ])
        for key in {"infect", "noinfect"}
    }
    print(f"# infect: {len(dfs['infect'])}, # noinfect: {len(dfs['noinfect'])}")

    # create tiles from the images and save
    for folder in {"infect", "noinfect"}:
        fallnummer_list = dfs[folder]["Fallnummer"].astype(str).to_list()
        image_id_list = dfs[folder]["imageID"].astype(int).astype(str).to_list() 
        for fallnummer, image_id in tqdm(zip(fallnummer_list, image_id_list)):
            fallnummer_path = os.path.join(ORTHOPEDIA_DIR, f"{folder}_tiles", fallnummer)
            if not os.path.exists(fallnummer_path):
                os.mkdir(fallnummer_path)
            image_id_path = os.path.join(fallnummer_path, image_id)
            if os.path.exists(image_id_path):
                continue
            
            os.mkdir(image_id_path)
            parser = parser_input.build_parser()
            args = parser.parse_args([
                "--method",
                METHOD,
                "--patch-size",
                PATCH_SIZE,
                "--output-downsample",
                OUTPUT_DOWNSAMPLE,
                "--content-threshold",
                CONTENT_THRESHOLD,
                os.path.join(ORTHOPEDIA_DIR, folder, f"{image_id}.svs"),
            ])
            slide = PySlide(vars(args))
            tile_extractor = TileGenerator(slide)
            tiles = tile_extractor.execute_and_return()
            utility_functions.clean(slide)
            for idx, tile in enumerate(tiles):
                im = Image.fromarray(tile).save(os.path.join(image_id_path, f"tile_{idx}.png"))
                
            
if __name__ == "__main__":
    main()
