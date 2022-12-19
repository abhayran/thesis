from pyhist import utility_functions, parser_input
from pyhist.slide import PySlide, TileGenerator
from PIL import Image
import os


METHOD = "graph"
PATCH_SIZE = "256"
OUTPUT_DOWNSAMPLE = "2"
CONTENT_THRESHOLD = "0.05"
ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"


def main():
    for folder in {"infect", "noinfect"}:
        os.mkdir(os.path.join(ORTHOPEDIA_DIR, f"{folder}_tiles"))
        for f in os.listdir(os.path.join(ORTHOPEDIA_DIR, folder)):
            new_folder_path = os.path.join(ORTHOPEDIA_DIR, f"{folder}_tiles", f.replace(".svs", ""))
            os.mkdir(new_folder_path)
            
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
                os.path.join(ORTHOPEDIA_DIR, folder, f),
            ])
            slide = PySlide(vars(args))
            tile_extractor = TileGenerator(slide)
            tiles = tile_extractor.execute_and_return()
            for idx, tile in enumerate(tiles):
                im = Image.fromarray(tile).save(os.path.join(new_folder_path, f"tile_{idx}.png"))
            utility_functions.clean(slide)
            
            # TODO: remove
            break
    
            
if __name__ == "__main__":
    main()
