# ONNX
1. **Prepare weights:** `cat weights/lama-regular.onnx_split* > weights/lama-regular.onnx`
2. **Run inference:** `python src.predict.py -i images/masked_image.jpg -m images/mask.png -w weights/lama-regular.onnx`

## Convert to IR

```
mo --input_model weights/lama-regular.onnx --output_dir weights
```

```
python src/to_openvino.py 
```