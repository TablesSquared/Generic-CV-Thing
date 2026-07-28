#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

MODEL    = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"
CONF     = 0.5
IOU      = 0.65
MAX_DETS = 10

imx500     = IMX500(MODEL)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = "object detection"
labels     = intrinsics.labels

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12
)
imx500.show_network_fw_progress_bar()
picam2.configure(config)

last_detections = []

def draw_detections(request):
    with MappedArray(request, "main") as m:
        for box, cat, conf in last_detections:
            x, y, w, h = box.astype(int)
            name = labels[cat] if labels else str(cat)
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(m.array, f"{name} {conf:.0%}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

picam2.pre_callback = draw_detections
picam2.start()

print("Running — Ctrl+C to stop")
try:
    while True:
        metadata = picam2.capture_metadata()
        outputs = imx500.get_outputs(metadata)

        if outputs is not None:
            boxes, scores, categories = postprocess_nanodet_detection(
                outputs,
                conf=CONF,
                iou_thres=IOU,
                max_out_dets=MAX_DETS,
            )

            print("boxes shape:", boxes.shape)
            print("scores shape:", scores.shape)
            print("categories shape:", categories.shape)
            print("boxes sample:", boxes[:2])

            last_detections = [
                (box, int(cat), float(score))
                for box, cat, score in zip(boxes, categories, scores)
                if score >= CONF
            ]
            for box, cat, score in last_detections:
                name = labels[cat] if labels else str(cat)
                print(f"  {name:20s} {score:.0%}  box={box}")

except KeyboardInterrupt:
    print("Stopped.")
finally:
    picam2.stop()
    picam2.stop()
    