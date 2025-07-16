[![DOI](https://zenodo.org/badge/1019373997.svg)](https://doi.org/10.5281/zenodo.15877632)
This code is directly related to the manuscript “Enhancing Solar Photovoltaic Defect Detection An Innovative Regression Loss Function Approach” we are currently submitting to The Visual Computer.
This project is an improvement based on YOLOv8, so you can run it in the same way as the standard YOLOv8. We provide three .yaml files in the project to correspond to different modules and their combinations mentioned in the paper. By default, the project uses CIoU. If you wish to use our proposed IdIoU or KIoE, simply replace 'CIoU' with 'IdIoU' or 'KIoE' in the line of code:
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).
