from torch.utils.data import Dataset


class MILDataset(Dataset):
    def __init__(self, orthopedia_dir: str) -> None:
        super().__init__()
        

        for key_folder in ["infect", "noinfect"]:
            key_path = os.path.join(orthopedia_dir, f"{key_folder}_tiles")
            for fn_folder in os.listdir(key_path):
                fn_path = os.path.join(key_path, fn_folder)
                for img_folder in os.listdir(fn_path):
                    img_path = os.path.join(fn_path, img_folder)                
                    img_files = os.listdir(img_path)

                    num_iters = len(img_files) // BATCH_SIZE + (len(img_files) % BATCH_SIZE > 0)
                    num_neutrophils = list()
                    for i in range(num_iters):
                        num_neutrophils.extend([len(preds) for preds in inference([
                            read_png_into_numpy(os.path.join(img_path, f)) for f in img_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]])
                        ])
                    np.save(os.path.join(fn_path.replace("tiles", "inference"), f"{img_folder}.npy"), np.array(num_neutrophils))
