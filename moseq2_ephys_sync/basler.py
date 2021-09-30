from datetime import time
import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
from glob import glob
import joblib
import argparse
import moseq2_extract.io.video as moseq_video
import pickle
import decord
from skimage import color

import pdb

import plotting, extract_leds, sync

def basler_workflow(base_path, save_path, num_leds, led_blink_interval, basler_chunk_size=1000, led_rois_from_file=False, overwrite_models=False):
    """
    Workflow to extract led codes from a Basler mp4 file.

    We know the LEDs only change once every (led_blink_interval), so we can just grab a few frames per interval. 
    
    """

    # Set up 
    basler_path = glob('%s/*.mp4' % base_path )[0]
    vr = decord.VideoReader(basler_path, ctx=decord.cpu(0), num_threads=8)
    num_frames = len(vr)
    timestamps = vr.get_frame_timestamp(np.arange(0,num_frames))  # blazing fast. nframes x 2 (beginning,end)
    print('Assuming Basler recorded at 120 fps...')
    timestamps = timestamps*2  # when basler records at 120 fps, timebase is halved :/

    ############### Cycle through the frame chunks to get all LED events: ###############
    
    # Prepare to load video (use frame batches)
    frame_batches = gen_batch_sequence(num_frames, basler_chunk_size, 0, offset=0)
    num_chunks = len(frame_batches)
    basler_led_events = []
    led_roi_list = load_led_rois_from_file(base_path)
    basler_led_events_path = os.path.join(base_path, 'basler_led_events.npz')
    print('num_chunks = ', num_chunks)

    # Do the loading
    if overwrite_models or (not os.path.isfile(basler_led_events_path)):

        for i in tqdm(range(num_chunks)[0:]):
            
            print(frame_batches[i])
            # NO! The skimage functions convert to float all at once, this causes memory issues!
            # NB: just request 100 GB ram :/
            frame_data_chunk = color.rgb2gray(vr.get_batch(list(frame_batches[i])).asnumpy())  
            # rgb_frame_data_chunk = vr.get_batch(list(frame_batches[i])).asnumpy()  # instead, convert one frame at a time. Size of return is N x W x H x 3.
            # frame_data_chunk = np.zeros(rgb_frame_data_chunk.shape[0:-1], dtype='uint8') # N x W x H
            # for j in range(rgb_frame_data_chunk.shape[0]):
            #     frame_data_chunk[j,:,:] = color.rgb2gray(rgb_frame_data_chunk)
            batch_timestamps = timestamps[frame_batches[i], 0]

            if i==0:
                plotting.plot_video_frame(frame_data_chunk.std(axis=0),'%s/basler_frame_std.pdf' % save_path)


            leds = extract_leds.get_led_data_from_rois(frame_data_chunk=frame_data_chunk, 
                                                    led_roi_list=led_roi_list,
                                                    led_thresh=0.5,  # since we converted to gray, all vals betw 0 and 1. "Off is around 0.1, "On" is around 0.8.
                                                    save_path=save_path)

            tmp_event = extract_leds.get_events(leds, batch_timestamps)
            basler_led_events.append(tmp_event)

            # actual_led_nums = np.unique(tmp_event[:,1]) ## i.e. what was found in this chunk
            # if np.all(actual_led_nums == range(num_leds)):    
            # else:
            #     print('Found %d LEDs found in chunk %d. Skipping... ' % (len(actual_led_nums),i))
                
            del frame_data_chunk
                
        basler_led_events = np.concatenate(basler_led_events)
        np.savez(basler_led_events_path,led_events=basler_led_events)
    else:
        basler_led_events = np.load(basler_led_events_path)['led_events']

    ############### Convert the LED events to bit codes ############### 
    basler_led_codes, latencies = sync.events_to_codes(basler_led_events, nchannels=4, minCodeTime=(led_blink_interval-1))  
    basler_led_codes = np.asarray(basler_led_codes)

    return basler_led_codes


### Basler HELPER FUNCTIONS ###

def load_led_rois_from_file(base_path):
    fin = os.path.join(base_path, 'led_rois.pickle')
    with open(fin, 'rb') as f:
        led_roi_list = pickle.load(f)
    return led_roi_list


def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    '''
    Generates batches used to chunk videos prior to extraction.

    Parameters
    ----------
    nframes (int): total number of frames
    chunk_size (int): desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns
    -------
    Yields list of batches
    '''

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out