# YOLOv5-Powered Self-Driving Cars — Object Detection

## Setup
```bash
pip install -r requirements.txt
```
(First run of `detect.py` auto-downloads YOLOv5 + pretrained weights via torch.hub — internet required once.)

## 1. Detect on a static image
```bash
python detect.py --source data/images/bus.jpg --weights yolov5s --conf 0.25
```

## 2. Detect on a live webcam
```bash
python detect.py --source 0 --weights yolov5s --conf 0.25
```
Press `q` in the preview window to stop.

## 3. Detect on a video file
```bash
python detect.py --source street.mp4 --weights yolov5s --conf 0.25
```

All annotated results are saved to the `outputs/` folder.

## 4. Train on a custom dataset (e.g. autorickshaws)
1. Put labeled images/labels into:
   ```
   custom_dataset/images/train, custom_dataset/images/val
   custom_dataset/labels/train, custom_dataset/labels/val
   ```
   (Label with LabelImg or Roboflow, YOLO txt format: `class_id x_center y_center width height`, normalized.)
2. Run:
   ```bash
   python train_custom.py --dataset_dir custom_dataset --classes car bike autorickshaw --epochs 50 --batch 16 --img 640
   ```
3. Use your trained weights for detection:
   ```bash
   python detect.py --source test_image.jpg --weights yolov5/runs/train/my_custom_model/weights/best.pt
   ```

## Submission
Zip this whole folder (excluding large `yolov5/` clone and `outputs/` if huge) before uploading, per the assignment guidelines.
