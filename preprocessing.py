import numpy as np
import pandas as pd
import imageio
import os
import warnings
import glob
import time
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
warnings.filterwarnings("ignore")
import shutil
import face_recognition
import cv2
import subprocess
import dlib
from mvextractor.videocap import VideoCap
from moviepy.editor import VideoFileClip
import librosa
import torch
from PIL import Image
from model import Model
from torchvision.transforms import ToTensor, ToPILImage


DEVNULL = open(os.devnull, "wb")
REF_FPS = 30

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def convertMVToFrameDim(mv, top, left, bottom, right, width=256, height=256):
    ct = 0
    res = torch.zeros((width, height, 2))
    for m in mv:
        i = m[4] - 1
        j = m[3] - 1
        if i >= left and j >= top and i < left + 256 and j < top + 256:
            res[i - left, j - top, 0] = m[6] - m[4]
            res[i - left, j - top, 1] = m[5] - m[3]
            ct += 1
    return res

def extract_mfccs(cmp_vid, cmp_mfcc_path, start, end, df):
    video_clip = VideoFileClip(cmp_vid)

    # Extract the audio
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("out.wav")

    y, sr = librosa.load("out.wav")
    frame_size = 0.04  # Frame size in seconds
    frame_stride = 0.04  # Frame stride in seconds
    n_mfcc = 40  # Number of MFCC coefficients
    n_fft = int(sr * frame_size)  # Number of FFT points
    hop_length = int(sr * frame_stride)

    ind = start
    for i in range(0, len(y) - n_fft, hop_length):
        if ind >= end:
            break
        frame = y[i:i+n_fft]  # Extract frame
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        mfcc_file = os.path.join(cmp_mfcc_path, f"{df.loc[ind, 'frame_no']}.pt")
        df.loc[ind, 'mfcc'] = mfcc_file
        torch.save(torch.from_numpy(mfcc), mfcc_file)
        ind += 1


def find_landmark(img):
    predictor_path =  "./model/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # img = dlib.load_rgb_image(f)

    dets = detector(img, 1)
    t = torch.zeros([256,256])

    if len(dets) == 0 or len(dets) > 1:
        return t

    for k, d in enumerate(dets):
        shape = predictor(img, d)
        for i in range(shape.num_parts):
            x = shape.part(i).x
            y = shape.part(i).y
            if x >= 0 and x <= 255 and y >= 0 and y <= 255:
                t[x, y] = 1
    return t

def detect_and_resize_face(frame, target_size=(256, 256)):
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 1:
        return frame, 0, 0, 0, 0
    if face_locations:  
        top, right, bottom, left = face_locations[0]
        face = frame[top:bottom, left:right]
        resized_face = cv2.resize(face, target_size)
        return resized_face, top, left, bottom, right
    return frame, 0, 0, 0, 0

def preprocess(vid_file):
    # Load Model
    model = Model()
    checkpoint = torch.load('./model/model.pth',map_location=torch.device('cpu')) 
    model.load_state_dict(checkpoint['model_state_dict'])
                
    # Data collection from original
    vc = VideoCap()
    vc.open(vid_file)
    video_clip = VideoFileClip(vid_file)

    # Extract the audio
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("out.wav")
    y, sr = librosa.load("out.wav")
    frame_size = 0.04  # Frame size in seconds
    frame_stride = 0.04  # Frame stride in seconds
    n_mfcc = 40  # Number of MFCC coefficients
    n_fft = int(sr * frame_size)  # Number of FFT points
    hop_length = int(sr * frame_stride)

    # Return enhanced Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as per your requirement
    out = cv2.VideoWriter('./media/enhanced_video.mp4', fourcc, 30, (256, 256))

    to_tensor = ToTensor()
    to_pil = ToPILImage()

    counter = 0
    while vc.grab():
        tup = vc.retrieve()

        # Cropping frame
        frame, t, l, b, r = detect_and_resize_face(tup[1])
        if t == 0 and l == 0 and b == 0 and r == 0:   
            counter += hop_length
            continue

        # Extracting Landmark
        landmark = find_landmark(frame)

        #Extracting MVs
        mv = convertMVToFrameDim(tup[2], t, l, b, r)
        
        # Extracting MFCCs
        segment = y[counter : counter + n_fft]  # Extract frame
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc = torch.from_numpy(mfcc)

        mv = mv.permute(2, 0, 1)
        mfcc = mfcc.T[:2]

        
        # Prepare Output
        if counter == 0:
            prev = to_tensor(frame)
        
        img = to_tensor(frame)
        output = model(img.unsqueeze(0), mfcc.unsqueeze(0), mv.unsqueeze(0), landmark.unsqueeze(0), prev.unsqueeze(0))
        output = output.squeeze(0)
        output = output.detach().cpu()
        outimg = to_pil(output)
        outimg = np.array(outimg)
        # tup[1][t : b, l : r] = outimg
        out.write(outimg)

        prev = img
        counter += hop_length

    out.release()
    vc.release()
    return './media/enhanced_video.mp4'
