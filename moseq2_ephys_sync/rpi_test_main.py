import os
import numpy as np

import numpy as np
import pandas as pd
import sys,os
from tqdm import tqdm
import subprocess
from glob import glob
import joblib
import argparse
import json

from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

import moseq2_extract.io.video as moseq_video

from moseq2_ephys_sync.video import get_mkv_stream_names, get_mkv_info
from moseq2_ephys_sync.extract_leds import gen_batch_sequence, get_led_data, get_events
from moseq2_ephys_sync.sync import events_to_codes, match_codes
from moseq2_ephys_sync.plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_matches_video_time,plot_video_frame
from moseq2_ephys_sync.rpi_utils import load_rpi_vid, get_rpi_test_rois, extract_rpi_leds, interpolate_missing_timestamps

import pdb

def sync(base_path):

    save_path = '%s/sync/' % base_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_path = glob('%s/*.mkv' % base_path )[0]

    print('Running sync on %s.' % base_path)

    stream_names = get_mkv_stream_names(depth_path) # e.g. {'DEPTH': 0, 'IR': 1}

    ### make paths for info and timestamps. if they exist, don't recompute:
    info_path = '%s/info.json' % base_path
    timestamp_path = '%s/mkv_timestamps.csv' % base_path 

    if (os.path.exists(info_path) and os.path.exists(timestamp_path) ):
        
        with open(info_path,'r') as f:
            info = json.load(f)

        timestamps = pd.read_csv(timestamp_path)
        timestamps = timestamps.values[:,1].flatten() # time in seconds

    else:
        ## get info on the depth file; we'll use this to see how many frames we have
        info,timestamps = get_mkv_info(depth_path,stream=stream_names['DEPTH'])

        ## save info and timestamps:
        timestamps = pd.DataFrame(timestamps)
        timestamps.to_csv(timestamp_path) # save the timestamps
        timestamps = timestamps.values.flatten()
        
        with open(info_path, 'w') as f:
            json.dump(info, f)

        

    ## we'll load the actual frames in chunks of 1000/2000. let's see how many chunks we need:
    nframes = info['nframes']
    chunk_size = 2000

    num_leds = 4

    ## get frame batches like in moseq2-extract:
    frame_batches = gen_batch_sequence(info['nframes'], chunk_size,
                                           0, offset=0)

    print('info = ', info)
    print('timestamps.shape = ', timestamps.shape)

    ############### Cycle through the frame chunks to get all LED events:
    num_chunks = len(frame_batches)
    led_events = []
    print('num_chunks = ', num_chunks)

    led_events_path = '%s_led_events.npz' % os.path.splitext(depth_path)[0]

    if not os.path.isfile(led_events_path):

        for i in tqdm(range(num_chunks)[0:]):
            
            frame_data_chunk = moseq_video.load_movie_data(depth_path,
                                           frames=frame_batches[i],
                                           mapping=stream_names['IR'], movie_dtype=">u2", pixel_format="gray16be",
                                          frame_size=info['dims'],timestamps=timestamps,threads=8,
                                                          finfo=info)

            if i==0:
                plot_video_frame(frame_data_chunk.std(axis=0),'%s/frame_std.pdf' % save_path)

            leds = get_led_data(frame_data_chunk=frame_data_chunk,
                            num_leds=num_leds,chunk_num=i,sort_by='horizontal',save_path=save_path)
            
            time_offset = frame_batches[i][0] ## how many frames away from first chunk's  #### frame_chunks[0,i]
            
            tmp_event = get_events(leds,timestamps[frame_batches[i]],time_offset,num_leds=num_leds)

            actual_led_nums = np.unique(tmp_event[:,1]) ## i.e. what was found in this chunk

            if np.all(actual_led_nums == range(num_leds)):
                led_events.append(tmp_event)
            else:
                print('Found %d LEDs found in chunk %d. Skipping... ' % (len(actual_led_nums),i))

                
            
        led_events = np.concatenate(led_events)

        ## optional: save the events for further use
        np.savez(led_events_path,led_events=led_events)
    
    else:
        led_events = np.load(led_events_path)['led_events']
        

    ################################ Load the rpi video if present #################
    
    rpi_vid_path = glob('%s/*.mp4' % base_path )[0]
    if not rpi_vid_path:
        raise RuntimeError('Expected rpi video as mp4')
    
    # To test the rpi, we record the bucket LEDs in the rpi video. 
    # Extract the LED signals and compare the Rpi timestamps to the open ephys timestamps.

    # Start with LED extraction.
    fps = 30
    led_samples = 10
    sec_per_sample = 5
    nframes = fps*sec_per_sample*led_samples
    led_thresh = 50
    rpi_vid_beginning = load_rpi_vid(rpi_vid_path, num_frames=nframes) # T x M x N x C. IR LEDs are most visible in third channel (BGR maybe?)
    leds_xs, leds_ys, sorting = get_rpi_test_rois(rpi_vid_beginning[:,:,;,2]) 

    # Determine total number of frames and extract signals
    vid_reader = skvideo.io.FFmpegReader(rpi_vid_path)
    (totalFrames, _, _, _) = vid_reader.getShape()
    vid_reader.close()
    rpi_leds = extract_rpi_leds(rpi_vid_path,
                        totalFrames=totalFrames,
                        leds_xs=leds_xs,
                        leds_ys=leds_ys,
                        sorting=sorting,
                        led_thresh=led_thresh)
    
    # Get rpi timestamps
    rpi_ts_path = '%s/rpicamera_video_timestamps.csv' % base_path
    rpi_ts = np.genfromtxt(rpi_ts_path, delimiter=',') # first col is frame times, second col is TTL trigger times
    if not rpi_ts:
        raise RuntimeError('Expected rpi timestamps as csv')
    rpi_frame_ts = (rpi_ts[:,0] - rpi_ts[0,0]) /1e6 # start at 0 and convert to sec
    rpi_ttl_ts = (rpi_ts[:,1] - rpi_ts[0,1]) /1e6 

    # For the purposes of testing, decode the LED pattern with the ground truth times
    rpi_frame_ts = interpolate_missing_timestamps(rpi_frame_ts, fps=30)
    rpi_frame_ts = interpolate_missing_timestamps(rpi_frame_ts, fps=30)
    rpi_frame_ts = rpi_frame_ts.ravel() 
    rpi_led_events = get_events(leds, rpi_frame_ts, time_offset=0, num_leds=4)

    ######### (General case) Fit a linear piecewise model to go from rpi frame to ephys TTL #############
    # First get rpi signal from ephys TTL
    ephys_fs = 3e4
    ephys_ttl_path = glob('%s/**/TTL_*/' % base_path,recursive = True)[0]
    channels = np.load('%s/channel_states.npy' % ephys_ttl_path)
    ephys_timestamps = np.load('%s/timestamps.npy' % ephys_ttl_path) / ephys_fs # in seconds
    print('Assuming rpi ttl is in ttl channel 7...')
    rpi_ttl_bool = np.isin(channels, [-7,7])
    rpi_ttl_events = np.vstack([ephys_timestamps[rpi_ttl_bool], channels[rpi_ttl_bool], np.sign(channels[rpi_ttl_bool])]).T

    # Get rpi's version of its ttl events
    rpi_ttl_ts


    ################################# Load the ephys TTL data #####################################

    ephys_ttl_path = glob('%s/**/TTL_*/' % base_path,recursive = True)[0]
    channels = np.load('%s/channel_states.npy' % ephys_ttl_path)
    ephys_timestamps = np.load('%s/timestamps.npy' % ephys_ttl_path)

    ephys_fs = 3e4

    led_fs = 30

    led_interval = 5 # seconds

    ## convert the LED events to bit codes:
    led_codes, latencies = events_to_codes(led_events,nchannels=4,minCodeTime=led_interval-1)
    led_codes = np.asarray(led_codes)


    ## convert the ephys TTL events to bit codes:
    
    # # DEBUG: try swapping channels?
    # channel_map = {-4:-1, -3:-2, -2:-3, -1:-4, 1:4, 2:3, 3:2, 4:1, -7:-7, 7:7}
    # pdb.set_trace()
    # channels = np.vectorize(channel_map.get)(channels)
    
    print('Assuming LED events in TTL channels 1-4...')
    ttl_channels = [-4,-3,-2,-1,1,2,3,4]
    ttl_bool = np.isin(channels, ttl_channels)
    ephys_events = np.vstack([ephys_timestamps[ttl_bool], abs(channels[ttl_bool])-1, np.sign(channels[ttl_bool])]).T
    ephys_codes, ephys_latencies = events_to_codes(ephys_events, nchannels=4, minCodeTime=(led_interval-1)*ephys_fs)
    ephys_codes = np.asarray(ephys_codes)


    np.savez('%s/codes.npz' % save_path, led_codes=led_codes, ephys_codes=ephys_codes)

    ## visualize a small chunk of the bit codes. do you see a match? 
    plot_code_chunk(ephys_codes,led_codes,ephys_fs,save_path)


    ################### Match the codes! ##########################

    matches = np.asarray(match_codes(ephys_codes[:,0] / ephys_fs,  ## converting the ephys times to seconds for matching (led times already in seconds)
                                  ephys_codes[:,1], 
                                  led_codes[:,0],
                                  led_codes[:,1],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))

    ## plot the matched codes against each other:
    plot_matched_scatter(matches,save_path)

    ####################### Make the models! ####################

    ephys_model = PiecewiseRegressor(verbose=True,
                               binner=KBinsDiscretizer(n_bins=10))
    ephys_model.fit(matches[:,0].reshape(-1, 1), matches[:,1])


    predicted_video_matches = ephys_model.predict(matches[:,0].reshape(-1, 1) ) ## for checking the error

    predicted_video_times = ephys_model.predict(ephys_codes[:,0].reshape(-1, 1) / ephys_fs ) ## for all predicted times

    joblib.dump(ephys_model, '%s/ephys_timebase.p' % save_path)
    print('Saved ephys model')

    ## how big are the differences between the matched ephys and video code times ?
    time_errors = (predicted_video_matches - matches[:,1]) 

    ## plot model errors:
    plot_model_errors(time_errors,save_path)

    ## plot the codes on the same time scale
    plot_matches_video_time(predicted_video_times,ephys_codes,led_codes,save_path)


    #################################

    video_model = PiecewiseRegressor(verbose=True,
                               binner=KBinsDiscretizer(n_bins=10))
    video_model.fit(matches[:,1].reshape(-1, 1), matches[:,0])


    predicted_ephys_matches = video_model.predict(matches[:,1].reshape(-1, 1) )

    predicted_ephys_times = video_model.predict(led_codes[:,0].reshape(-1, 1) )

    joblib.dump(video_model, '%s/video_timebase.p' % save_path)
    print('Saved video model')


    print('Syncing complete. FIN')




if __name__ == "__main__" :

    ## take a config file w/ a list of paths, sync each of those and plot the results in a subfolder called /sync/

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)
    

    settings = parser.parse_args(); 

    base_path = settings.path #'/n/groups/datta/maya/ofa-snr/mj-snr-01/mj_snr_01_2021-03-24_11-06-33/'
    

    sync(base_path)

    
