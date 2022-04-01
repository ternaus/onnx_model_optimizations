import argparse
from pathlib import Path

import cv2
import numpy as np
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline
from openvino.tools.pot import load_model, save_model


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=str, help="Path to the images and masks.", required=True)
    return parser.parse_args()


class ImageLoader(DataLoader):
    def __init__(self, input_path: str):
        file_paths = sorted(Path(input_path).glob("*.*"))
        self.image_paths = [x for x in file_paths if "mask" not in x.as_posix()]
        self.mask_paths = [x for x in file_paths if "mask" in x.as_posix()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = cv2.resize(cv2.imread(str(self.image_paths[index])), (768, 768))
        mask = cv2.resize(cv2.imread(str(self.mask_paths[index]), 0), (768, 768)) > 0

        image = np.expand_dims(image.transpose(2, 0, 1), 0) / 255
        mask = np.expand_dims(mask, axis=(0, 1))

        return {"image": image * mask, "mask": (1 - mask)}, None


def main():
    args = get_args()

    model_config = {
        "model_name": "model",
        "model": "weights/lama-regular.xml",
        "weights": "weights/lama-regular.bin"
    }

    engine_config = {"device": "CPU"}

    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "ANY",
                "stat_subset_size": 300
            },
        }
    ]

    data_loader = ImageLoader(args.image_path)
    model = load_model(model_config=model_config)
    engine = IEEngine(config=engine_config, data_loader=data_loader)
    pipeline = create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(model=model)
    compress_model_weights(compressed_model)
    save_model(
        model=compressed_model,
        save_path="optimized_model",
        model_name="optimized_model",
    )


if __name__ == "__main__":
    main()
