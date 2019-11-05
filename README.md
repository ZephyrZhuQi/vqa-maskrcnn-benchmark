# Ground Truth Bounding Box extracted image feature
This repository is a tool to extract object feature maps from images.
If you want to use ground truth bounding boxes(e.g. Open Images bounding box) instead of region proposals(generated in rpn in Mask R-CNN), you will find this project helpful.
Only several small modifications have been added to the original vqa-maskrcnn-benchmark.

# Changed files
1. ./maskrcnn_benchmark/structures/image_list.py
2. ./maskrcnn_benchmark/modeling/detector/generalized_rcnn.py
3. ./extract_features_vmb.py
4. ./maskrcnn_benchmark/modeling/roi_heads/box_head/box_head.py


# Instructions
0. set up the environment
first pip install torch and torchvison

```bash
python setup.py build
python setup.py develop
```
```bash
pip install opencv-python yacs tqdm
```
1. download detectron model and configuration file
```
- "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
- "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
```
put them in ./model/detectron_model.pth and ./model/detectron_model.yaml
2. download ground truth bounding box and store them in ./bbox/bbox.json file(a dictionary), with file format:
```
{
  "000c5171b38d4bb0": [
    [
      "0.742582",
      "0.779320",
      "0.518804",
      "0.597472"
    ]
  ],
  "00183df6ffe09093": [
    [
      "0.432475",
      "0.480983",
      "0.774041",
      "0.832049"
    ]
}
000c5171b38d4bb0 is image id and the four numbers are xmin, xmax, ymin, ymax(after normalization)
```
3. put images in ./data/images
4. generating feature maps
```bash
CUDA_VISIBLE_DEVICES=x nohup python -u  extract_features_vmb.py --image_dir ./data/images --model_file ./model/detectron_model.pth --config_file ./model/detectron_model.yaml --output_folder ./output --batch_size 2 --bbox_json ./bbox/bbox.json > generate_feat.log 2>&1 &
```

# Reminders
1. the images have to be jpg file