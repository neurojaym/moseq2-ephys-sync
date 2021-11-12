from datetime import time
import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
import subprocess
from glob import glob
import joblib
import argparse
import json
import moseq2_extract.io.video as moseq_video
import subprocess
import pickle
import pdb

import moseq2_ephys_sync.plotting as plotting
import moseq2_ephys_sync.extract_leds as extract_leds
import moseq2_ephys_sync.sync as sync

def mkv_workflow(base_path, save_path, num_leds, led_blink_interval, mkv_chunk_size=2000, led_loc=None, led_rois_from_file=False, overwrite_mkv_extraction=False):
    """
    Workflow to extract led codes from an MKV file
    
    """

    # Set up paths
    depth_path = glob('%s/*.mkv' % base_path )[0]
    stream_names = get_mkv_stream_names(depth_path) # e.g. {'DEPTH': 0, 'IR': 1}
    info_path = '%s/info.json' % base_path  # make paths for info and timestamps. if they exist, don't recompute:
    timestamp_path = '%s/mkv_timestamps.csv' % base_path 


    # Load timestamps and mkv info if exist, otherwise calculate
    if (os.path.exists(info_path) and os.path.exists(timestamp_path) ):
        print('Loading preexisting mkv timestamps...')

        with open(info_path,'r') as f:
            info = json.load(f)

        timestamps = pd.read_csv(timestamp_path)
        timestamps = timestamps.values[:,1].flatten()

    else:
        print('Loading mkv timestamps de novo...')
        ## get info on the depth file; we'll use this to see how many frames we have
        info,timestamps = get_mkv_info(depth_path,stream=stream_names['DEPTH'])

        ## save info and timestamps:
        timestamps = pd.DataFrame(timestamps)
        timestamps.to_csv(timestamp_path) # save the timestamps
        timestamps = timestamps.values.flatten()
        
        with open(info_path, 'w') as f:
            json.dump(info, f)

    # Debugging
    # print('info = ', info)
    # print('timestamps.shape = ', timestamps.shape)

    
    ############### Cycle through the frame chunks to get all LED events: ###############
    
    # Prepare to load video (use frame batches like in moseq2-extract)
    frame_batches = gen_batch_sequence(info['nframes'], mkv_chunk_size,
                                           0, offset=0)
    num_chunks = len(frame_batches)
    mkv_led_events = []
    print('num_chunks = ', num_chunks)

    if led_rois_from_file:
        led_roi_list = load_led_rois_from_file(base_path)

    mkv_led_events_path = '%s_led_events.npz' % os.path.splitext(depth_path)[0]

    # Do the loading
    if not os.path.isfile(mkv_led_events_path) or overwrite_mkv_extraction:

        for i in tqdm(range(num_chunks)[0:]):
        # for i in [45]:
            
            frame_data_chunk = moseq_video.load_movie_data(depth_path,  # nframes, nrows, ncols
                                           frames=frame_batches[i],
                                           mapping=stream_names['IR'], movie_dtype=">u2", pixel_format="gray16be",
                                          frame_size=info['dims'],timestamps=timestamps,threads=8,
                                                          finfo=info)

            if i==0:
                plotting.plot_video_frame(frame_data_chunk.std(axis=0),'%s/frame_std.pdf' % save_path)

            if led_rois_from_file:
                leds = extract_leds.get_led_data_from_rois(frame_data_chunk=frame_data_chunk, led_roi_list=led_roi_list, save_path=save_path)
            else:
                leds = extract_leds.get_led_data_with_stds(frame_data_chunk=frame_data_chunk,
                                num_leds=num_leds,chunk_num=i, led_loc=led_loc, save_path=save_path)
            
            time_offset = frame_batches[i][0] ## how many frames away from first chunk's  #### frame_chunks[0,i]
            
            tmp_event = extract_leds.get_events(leds,timestamps[frame_batches[i]])

            actual_led_nums = np.unique(tmp_event[:,1]) ## i.e. what was found in this chunk


            if np.all(actual_led_nums == range(num_leds)):
                mkv_led_events.append(tmp_event)
            else:
                print('Found %d LEDs found in chunk %d. Skipping... ' % (len(actual_led_nums),i))
                
                
            
        mkv_led_events = np.concatenate(mkv_led_events)

        ## optional: save the events for further use
        np.savez(mkv_led_events_path,led_events=mkv_led_events)
    
    else:
        mkv_led_events = np.load(mkv_led_events_path)['led_events']

    print('Successfullly extracted mkv leds, converting to codes...')    


    ############### Convert the LED events to bit codes ############### 
    mkv_led_codes, latencies = sync.events_to_codes(mkv_led_events, nchannels=4, minCodeTime=(led_blink_interval-1))
    mkv_led_codes = np.asarray(mkv_led_codes)
    print('Converted.')

    return mkv_led_codes


### MKV HELPER FUNCTIONS ###

def load_led_rois_from_file(base_path):
    fin = os.path.join(base_path, 'led_rois.pickle')
    with open(fin, 'rb') as f:
        led_roi_list = pickle.load(f, pickle.HIGHEST_PROTOCOL)
    return led_roi_list

def get_mkv_info(fileloc, stream=1):
    stream_features = ["width", "height", "r_frame_rate", "pix_fmt"]

    outs = {}
    for _feature in stream_features:
        command = [
            "ffprobe",
            "-select_streams",
            "v:{}".format(int(stream)),
            "-v",
            "fatal",
            "-show_entries",
            "stream={}".format(_feature),
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            fileloc,
            "-sexagesimal",
        ]
        ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        if err:
            print(err)
        outs[_feature] = out.decode("utf-8").rstrip("\n")

    # need to get duration and nframes the old fashioned way
    outs["duration"] = get_mkv_duration(fileloc)
    timestamps = get_mkv_timestamps(fileloc,stream)
    outs["nframes"] = len(timestamps)

    return (
        {
            "file": fileloc,
            "dims": (int(outs["width"]), int(outs["height"])),
            "fps": float(outs["r_frame_rate"].split("/")[0])
            / float(outs["r_frame_rate"].split("/")[1]),
            "duration": outs["duration"],
            "pixel_format": outs["pix_fmt"],
            "nframes": outs["nframes"],
        },
        timestamps,
    )

def get_mkv_duration(fileloc, stream=1):
    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        fileloc,
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    return float(out.decode("utf-8").rstrip("\n"))


def get_mkv_timestamps(fileloc, stream=1,threads=8):
    command = [
        "ffprobe",
        "-select_streams",
        "v:{}".format(int(stream)),
        "-v",
        "fatal",
        "-threads", str(threads),
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "csv=p=0",
        fileloc,
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    timestamps = out.decode("utf-8").rstrip("\n").split("\n")
    timestamps = np.array([float(_) for _ in timestamps])
    return timestamps

def get_mkv_stream_names(fileloc):
    stream_tag = "title"

    outs = {}
    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "stream_tags={}".format(stream_tag),
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        fileloc,
    ]
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    out = out.decode("utf-8").rstrip("\n").split("\n")
    
    
    ## !! changed the key/value order here from what JM had: (i.e. so the string name is the key, the stream is the value)
    return dict(list(zip(out,np.arange(len(out)))))


def get_mkv_stream_tag(fileloc, stream=1, tag="K4A_START_OFFSET_NS"):

    command = [
            "ffprobe",
            "-v",
            "fatal",
            "-show_entries",
            "format_tags={}".format(tag),
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            fileloc,
        ]
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print(err)
    out = out.decode("utf-8").rstrip("\n")
    return out

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