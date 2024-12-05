# from os import listdir, path
import numpy as np
import cv2, os
# import json, subprocess, random, string
from tqdm import tqdm
# from glob import glob
import torch
import face_detection
from models import Wav2Lip
# import platform
import traceback
import configparser
import json
import pickle

config = configparser.ConfigParser()
config.read("config.ini")

var_face = config["Wav2Lip"]["avatar_path"]
var_resize_factor = config["Wav2Lip"].getfloat("resize_factor")
var_checkpoint_path=config["Wav2Lip"]["checkpoint_path"]
var_static=config["Wav2Lip"].getboolean("static")
var_pads=json.loads(config["Wav2Lip"]["pads"])
var_face_det_batch_size=config["Wav2Lip"].getint("face_det_batch_size")
var_box=json.loads(config["Wav2Lip"]["box"])
var_nosmooth=config["Wav2Lip"].getboolean("nosmooth")
var_img_size =config["Wav2Lip"].getint("img_size")
var_wav2lip_batch_size = config["Wav2Lip"].getint("wav2lip_batch_size")

# print("pads: ", var_pads)
# print("box: ", var_box)
facepickle = 'fd_results/' + config["Wav2Lip"]["avatar_path"].split('/')[-1] + '.pickle'
face_det_results = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.isfile(var_face) and var_face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	var_static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images):
	global facepickle
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device=device)

	batch_size = var_face_det_batch_size
	print('Face detection:')
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = var_pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not var_nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	# fd_pickle[var_face] = results
	print("Write FD to pickle")
	with open(facepickle, "wb") as fp: 
		pickle.dump(results, fp)

	del detector
	return results


def datagen(frames, mels):
	global face_det_results, fd_pickle
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	try:
		if(face_det_results == []):
			print("FD results not available")
			if os.path.isfile(facepickle):
				print("Get FD from pickle")
				with open(facepickle, "rb") as fp: 
					face_det_results = pickle.load(fp)
			# else:
			# 	fd_pickle[var_face] = None

			# if(fd_pickle[var_face] and fd_pickle[var_face] != []):
			# 	face_det_results = fd_pickle[var_face]
			else:
				print("Generate new FD results")
				if var_box[0] == -1:
					if not var_static:
						face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
					else:
						face_det_results = face_detect([frames[0]])
				else:
					print('Using the specified bounding box instead of face detection...')
					y1, y2, x1, x2 = var_box
					face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
		print("DATAGEN frames mels", len(frames), len(mels))
		print("DATAGEN fd results", len(face_det_results))
		for i, m in enumerate(mels):
			idx = 0 if var_static else i%len(frames)
			frame_to_save = frames[idx].copy()
			face, coords = face_det_results[idx].copy()

			face = cv2.resize(face, (var_img_size, var_img_size))

			img_batch.append(face)
			mel_batch.append(m)
			frame_batch.append(frame_to_save)
			coords_batch.append(coords)

			if len(img_batch) >= var_wav2lip_batch_size:
				# print(f"img_batch type: {len(img_batch)}, shapes: {[x.shape for x in img_batch]}")
				# print(f"mel_batch type: {len(mel_batch)}, shapes: {[x.shape for x in mel_batch]}")

				img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

				img_masked = img_batch.copy()
				img_masked[:, var_img_size//2:] = 0

				img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
				mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

				yield img_batch, mel_batch, frame_batch, coords_batch
				img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

		if len(img_batch) > 0:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, var_img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
	except Exception as e:
            print(f"Error in infer_model: {e}")
            traceback.print_exc()

# mel_step_size = 16
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint


def load_model(path):
	model = Wav2Lip()
	print("Load model checkpoint from: {}".format(path))
	# traceback.print_stack()
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()
