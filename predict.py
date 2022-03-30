import argparse
import time

import cv2
import numpy as np
from openvino.inference_engine import IECore

ie = IECore()


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image", type=str, help="Path to the masked image.", required=True)
    arg("-m", "--mask", type=str, help="Path to mask.", required=True)
    arg("-w", "--weights", type=str, help="Path to weights.", required=True)
    return parser.parse_args()


class Inpainter:
    def __init__(self, weights_path: str):
        net_onnx = ie.read_network(model=weights_path)
        self.exec_net_onnx = ie.load_network(network=net_onnx, device_name="CPU")
        self.output_layer_onnx = next(iter(self.exec_net_onnx.outputs))

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        res_onnx = self.exec_net_onnx.infer(inputs={"image": image, "mask": mask})
        return res_onnx[self.output_layer_onnx][0]


def main():
    args = get_args()

    model = Inpainter(args.weights)

    image = np.expand_dims(cv2.imread(args.image).transpose(2, 0, 1), 0) / 255
    mask = np.expand_dims(cv2.imread(args.mask, 0) > 0, 0)

    start_time = time.time()
    result = (np.clip(model(image, mask) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0))
    print(f"Inference performed in {time.time() - start_time} seconds")

    cv2.imwrite("result.jpg", result)


if __name__ == "__main__":
    main()
