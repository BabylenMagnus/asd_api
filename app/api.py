import warnings

import gc
import tempfile

from scipy.interpolate import interp1d

from io import StringIO
import json

import logging
import torch
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, RedirectResponse

from scenedetect import detect, AdaptiveDetector
import cv2
from datetime import datetime

import os
from .model import S3FD
from .ASD import ASD
import numpy as np
import subprocess
import python_speech_features
import math
from tqdm import tqdm

from skimage.metrics import structural_similarity  as ssim


warnings.simplefilter("ignore", UserWarning)
logging.getLogger().setLevel(logging.INFO)

app = FastAPI()
DET = S3FD(device='cuda')
asd_model = ASD()
pretrain_model = "app/weights/pretrain_AVA_CVPR.model"
asd_model.loadParameters(pretrain_model)
asd_model.eval()
logging.getLogger().setLevel(logging.INFO)


def scene_detect(video_path):
    scene_list = detect(
        video_path, AdaptiveDetector(adaptive_threshold=4, min_content_val=18), show_progress=True, start_in_scene=True
    )
    return scene_list


def inference_video(video_path, facedet_scale, max_sim=.96, skip_fps=1):
    cap = cv2.VideoCapture(video_path)
    dets = []
    i = 0
    ret, last_frame = cap.read()
    bboxes = DET.detect_faces(last_frame, conf_th=0.9, scales=[facedet_scale])
    for bbox in bboxes:
        dets[-1].append(
            {'frame': i, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]}
        )  # dets has the frames info, bbox info, conf info
    last_frame = last_frame.mean(axis=2)

    while True:
        if not i % 5000:
            logging.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + str(i))

        if not i % skip_fps:
            cap.set(1, i)
            _, image = cap.read()

            if image is None:
                break
            try:
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image_np = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            current_frame = image.mean(axis=2)

            if ssim(current_frame, last_frame, data_range=255) < max_sim:
                bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])
                last_frame = current_frame

            dets.append([])

            for bbox in bboxes:
                dets[-1].append(
                    {'frame': i, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]}
                )  # dets has the frames info, bbox info, conf info

        i += 1

    return dets


def bb_intersection_over_union(box_a, box_b, eval_col=False):
    # CPU: IOU Function to calculate overlap between two image
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if eval_col:
        iou = inter_area / float(box_a_area)
    else:
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def track_shot(scene_faces, num_failed_det, min_track, min_face_size):
    iou_threshold = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in frame_faces:
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iou_threshold:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        elif len(track) > min_track:
            frame_num = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frame_i = np.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(frame_num, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    return tracks


def load_audio(file: str, s: float, e: float, sr: int):
    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-vn",
        "-ss", str(round(s, 3)),
        "-to", str(round(e, 3)),
        "-"
    ]
    # fmt: on
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def evaluate_network(all_tracks, filename, sample_rate=16000):
    duration_set = {1, 1, 2, 2, 2, 3, 3, 4, 5, 6}  # Use this line can get more reliable result
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for track_id in tqdm(range(len(all_tracks))):
        start, end = all_tracks[track_id]["frame"][0], all_tracks[track_id]["frame"][-1]
        audio = load_audio(filename, start / fps, end / fps, sample_rate)
        audio_feature = python_speech_features.mfcc(audio, sample_rate, numcep=13, winlen=0.025, winstep=0.010)

        faces = []

        for i in range(start, end + 1):
            cap.set(1, i)
            ret, frame = cap.read()
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
            faces.append(face)

        video_feature = np.array(faces)
        length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0])
        audio_feature = audio_feature[:int(round(length * 100)), :]
        video_feature = video_feature[:int(round(length * 25)), :, :]

        all_score = []
        for duration in tqdm(duration_set):
            batch_size = int(math.ceil(length / duration))

            scores = []
            with torch.no_grad():
                for i in range(batch_size):
                    input_a = torch.FloatTensor(
                        audio_feature[i * duration * 100:(i + 1) * duration * 100, :]
                    ).unsqueeze(0).cuda()
                    input_v = torch.FloatTensor(
                        video_feature[i * duration * 25: (i + 1) * duration * 25, :, :]
                    ).unsqueeze(0).cuda()
                    if not (input_v.shape[1] and input_a.shape[1]):
                        continue
                    embed_a = asd_model.model.forward_audio_frontend(input_a)
                    embed_v = asd_model.model.forward_visual_frontend(input_v)
                    min_shape = min(embed_v.shape[1], embed_a.shape[1])
                    embed_a = embed_a[:, :min_shape]
                    embed_v = embed_v[:, :min_shape]
                    out = asd_model.model.forward_audio_visual_backend(embed_a, embed_v)
                    score = asd_model.lossAV.forward(out, labels=None)
                    scores.extend(score)
            all_score.append(scores)
        all_score = np.round((np.mean(np.array(all_score), axis=0)), 1).astype(float)

        all_tracks[track_id]["scores"] = all_score.tolist()
        all_tracks[track_id]["bbox"] = all_tracks[track_id]["bbox"].tolist()
        all_tracks[track_id]["frame"] = all_tracks[track_id]["frame"].tolist()

    return all_tracks


@app.post("/asd")
async def apply_effects_endpoint(
        video_file: UploadFile = File(...),
        min_track: int = Query(
            default=3
        ),
        num_failed_det: int = Query(
            default=10
        ),
        min_face_size: int = Query(
            default=40
        ),
        facedet_scale: float = Query(
            default=0.25
        ),
        max_sim: float = Query(
            default=0.96
        ),
        skip_fps: int = Query(
            default=5
        ),
):
    filename = tempfile.mktemp(".mp4")
    with open(filename, "wb") as t:
        t.write(video_file.file.read())
    scene = scene_detect(filename)
    logging.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": get scenes")
    faces = inference_video(filename, facedet_scale, max_sim, skip_fps)
    logging.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": get faces")

    all_tracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= min_track:  # Discard the shot frames less than min_track frames
            all_tracks.extend(
                track_shot(
                    faces[shot[0].frame_num:shot[1].frame_num],
                    num_failed_det, min_track, min_face_size
                )
            )
            # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    logging.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": get tracks")

    all_tracks = evaluate_network(all_tracks, filename)
    logging.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": finishing")

    # After processing, remove the uploaded files
    os.remove(filename)

    # Manually run garbage collector and reset PyTorch cache
    gc.collect()
    torch.cuda.empty_cache()
    output_file = StringIO()
    json.dump(all_tracks, output_file)
    output_file.seek(0)

    # Return the file as a response
    return StreamingResponse(
        output_file,
        media_type="text/plain",
        headers={
            'Content-Disposition': f'attachment;'
        }
    )
