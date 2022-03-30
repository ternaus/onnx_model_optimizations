1. **Prepare weights:** `cat weights/lama-regular.onnx_split* > weights/lama-regular.onnx`
2. **Run inference:** `python infer.py -i images/image.jpg -m mask.png -w weights/lama-regular.onnx`