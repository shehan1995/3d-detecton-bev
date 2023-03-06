""" Inference Code """

from typing import List
from PIL import Image
import cv2
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from torchvision import models

import re
import torch
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule

from deep_sort.tracker import Tracker
from src.utils import Calib
from src.utils.averages import ClassAverages
from src.utils.Plotting import calc_alpha, plot_3d_box
from src.utils.Math import calc_location, compute_orientaion, recover_angle, translation_constraints
from src.utils.Plotting import calc_theta_ray
from src.utils.Plotting import Plot3DBoxBev

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys
import pyrootutils
import src.utils
from src.utils.utils import KITTIObject

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder

import time

log = src.utils.get_pylogger(__name__)

try:
    import onnxruntime
    import openvino.runtime as ov
except ImportError:
    log.warning("ONNX and OpenVINO not installed")

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def natural_sort_key(s):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', s)]


def predict_path(y_min_values):
    # Assuming y_min_values is a list of the previous 5 y_min values for a vehicle
    # and time_delta is the time elapsed between the last two measurements in seconds
    # y_min_values = [300, 305, 310, 315, 320]
    if len(y_min_values) < 5:
        return np.zeros(1)
    time_delta = 1.0
    n = len(y_min_values)

    # Convert y_min_values to a numpy array and create a corresponding array of time values
    y_min_array = np.array(y_min_values)
    time_array = np.array(range(n)) * time_delta

    # Perform linear regression to fit a line to the points
    slope, intercept = np.polyfit(time_array, y_min_array, 1)

    # Calculate the velocity as the slope of the line
    velocity = slope

    future_time_array = np.arange(n, n+5, time_delta)

    future_y_min_array = slope * future_time_array + intercept

    return future_y_min_array



@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages #TODO: depricated
    class_averages = ClassAverages()

    # =============== Tracker things ==============
    max_cosine_distance = float(0.4)
    nn_budget = None
    nms_max_overlap: float = 1.0
    FPS = 20
    # Define the time interval between frames (in seconds)
    TIME_INTERVAL = 1 / FPS

    encoder = create_box_encoder("./deep_sort/model_weights/mars-small128.pb", batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                       nn_budget)
    tracker = Tracker(metric)

    # time
    avg_time = {
        "initiate_detector": 0,
        "initiate_regressor": 0,
        "detector": 0,
        "regressor": 0,
        "plotting": 0,
    }

    # initialize detector models
    start_detector = time.time()
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)
    avg_time["initiate_detector"] = time.time() - start_detector

    # initialize regressor models
    start_regressor = time.time()

    # pytorch regressor models
    log.info(f"Instantiating regressor <{config.model._target_}>")
    regressor: LightningModule = hydra.utils.instantiate(config.model)
    regressor.load_state_dict(torch.load(config.get("regressor_weights"), map_location='cpu'))
    regressor.eval().to('cpu')

    avg_time["initiate_regressor"] = time.time() - start_regressor

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("output_dir"), exist_ok=True)

    # Initialize an empty dictionary to store tracks
    tracks = {}

    # TODO: inference on video
    # loop thru images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")), key=natural_sort_key)
    for img_path in imgs_path:

        # initialize tracker
        bboxes = []
        scores = []
        classes = []
        cls_names = []

        # Initialize object and plotting modules
        plot3dbev = Plot3DBoxBev(P2)

        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detect object with Detector
        start_detect = time.time()
        frame = img.copy()
        dets = detector(img).crop(save=config.get("save_det2d"))
        # detector(frame).render(labels=False)   # to show 2D detected images
        # pd = detector(img).pandas()
        avg_time["detector"] += time.time() - start_detect

        # dimension averages #TODO: depricated
        DIMENSION = []
        height = frame.shape[0]
        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det["label"].split(" ")[0].capitalize()
            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            box = [box.cpu().numpy() for box in det["box"]]
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = box[0], box[1], box[2], box[3]

            bbox = box
            conf = det["conf"].item()
            cls_name = obj.name
            cls = det["cls"].item()

            x, y, xmax, ymax = bbox
            w, h = xmax - x, ymax - y
            bbox = [x, y, w, h]

            bboxes.append(bbox)
            scores.append(conf)
            classes.append(cls)
            cls_names.append(cls_name)

            # preprocess img with torch.transforms
            crop = preprocess(cv2.resize(det["im"], (224, 224)))
            crop = crop.reshape((1, *crop.shape)).to('cpu')

            start_reg = time.time()

            # regress 2D bbox with Regressor
            [orient, conf, dim] = regressor(crop)
            orient = orient.cpu().detach().numpy()[0, :, :]
            conf = conf.cpu().detach().numpy()[0, :]
            dim = dim.cpu().detach().numpy()[0, :]

            # dimension averages # TODO: depricated
            try:
                dim += class_averages.get_item(class_to_labels(det["cls"].cpu().numpy()))
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]

            obj.alpha = recover_angle(orient, conf, 2)
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

            # output prediction label
            output_line = obj.member_to_list()
            output_line.append(1.0)
            output_line = " ".join([str(i) for i in output_line]) + "\n"

            avg_time["regressor"] += time.time() - start_reg

            # write results
            if config.get("save_txt"):
                with open(f"{config.get('output_dir')}/{img_name}.txt", "a") as f:
                    f.write(output_line)

            if config.get("save_result"):
                start_plot = time.time()
                plot3dbev.plot(
                    img=img,
                    class_object=obj.name.lower(),
                    bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                    dim=[obj.h, obj.w, obj.l],
                    loc=[obj.tx, obj.ty, obj.tz],
                    rot_y=obj.rot_global,
                    file_name=img_name,
                )
                avg_time["plotting"] += time.time() - start_plot

            # save images
            if config.get("save_result"):
                # cv2.imwrite(f'{config.get("output_dir")}/{name}.png', img_draw)
                plot3dbev.save_plot(config.get("output_dir"), img_name)

        # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
        features = encoder(frame,
                           bboxes)  # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, cls_names,
                          features)]  # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b')  # initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),
                          color, -1)
            cv2.putText(frame, class_name + " : " + str(track.track_id), (int(bbox[0]), int(bbox[1] - 11)),
                        0,
                        0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
            #     str(track.track_id), class_name,
            #     (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # Get the track ID and class name
            track_id = str(track.track_id)
            class_name = track.get_class()

            # Get the y_min coordinate of the bounding box
            y_min = int(track.to_tlbr()[3])

            # Add the y_min coordinate to the track
            if track_id in tracks:
                tracks[track_id]['y_min'].append(y_min)
            else:
                tracks[track_id] = {'class': class_name, 'y_min': [y_min]}

            prediction = predict_path(tracks[track_id]['y_min'])
            print("id ",track_id,"y ",y_min,prediction)
            if len(prediction)>1 and prediction[len(prediction)-1]>430:
                print("warning")

        tracker.update(detections)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.line(result, (0, 430), (result.shape[1], 430), (0, 0, 255), 2)

        cv2.imshow("Output Video", result)
        cv2.waitKey(2)

    # ---------------------------------- DeepSORT tacker work ends here ------------------------------------------------------------

    # print time
    for key, value in avg_time.items():
        if key in ["detector", "regressor", "plotting"]:
            avg_time[key] = value / len(imgs_path)
    log.info(f"Average Time: {avg_time}")

    cv2.destroyAllWindows()


def detector_yolov5(model_path: str, cfg_path: str, classes: int, device: str):
    """YOLOv5 detector models"""
    sys.path.append(str(root / "yolov5"))
    # sys.path.insert(0, './yolov5')

    # NOTE: ignore import error
    from models.common import AutoShape
    from models.yolo import Model
    from utils.general import intersect_dicts
    from utils.torch_utils import select_device

    # device = select_device(
    #     ("0" if torch.cuda.is_available() else "cpu") if device is None else device
    # )
    device = 'cpu'

    model = Model(cfg_path, ch=3, nc=classes)
    ckpt = torch.load(model_path, map_location=device)  # load
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if len(ckpt["model"].names) == classes:
        model.names = ckpt["model"].names  # set class names attribute
    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS

    return model.to(device)


def class_to_labels(class_: int, list_labels: List = None):
    if list_labels is None:
        # TODO: change some labels mistakes
        list_labels = ["car", "car", "truck", "pedestrian", "cyclist"]

    return list_labels[int(class_)]


if __name__ == "__main__":
    inference()
