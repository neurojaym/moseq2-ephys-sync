import numpy as np
import pandas as pd
import sys,os
import copy

from scipy import stats
import datetime
from tqdm import tqdm
import scipy
import subprocess

import shutil
from glob import glob
import joblib
import argparse
import json


from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

import moseq2_extract.io.video as moseq_video

from video import get_mkv_stream_names, get_mkv_info
from extract_leds import gen_batch_sequence, get_led_data, get_events
from sync import events_to_codes, match_codes
from plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_matches_video_time,plot_video_frame


if __name__ == "__main__" :

    ## take a config file w/ a list of paths, sync each of those and plot the results in a subfolder called /sync/

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)

    settings = parser.parse_args(); 

    base_path = settings.path #'/n/groups/datta/maya/ofa-snr/mj-snr-01/mj_snr_01_2021-03-24_11-06-33/'

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
        timestamps = timestamps.values[:,1].flatten()

    else:
        ## get info on the depth file; we'll use this to see how many frames we have
        info,timestamps = get_mkv_info(depth_path,stream=stream_names['DEPTH'])

        ## save info and timestamps:
        timestamps = pd.DataFrame(timestamps)
        timestamps.to_csv(timestamp_path) # save the timestamps
        timestamps = timestamps.values
        
        with open(info_path, 'w') as f:
            json.dump(info, f)

        

    ## we'll load the actual frames in chunks of 1000/2000. let's see how many chunks we need:
    nframes = info['nframes']
    chunk_size = 2000

    ## get frame batches like in moseq2-extract:
    frame_batches = gen_batch_sequence(info['nframes'], chunk_size,
                                           0, offset=0)

    print('info = ', info)
    print('timestamps.shape = ', timestamps.shape)

    ############### Cycle through the frame chunks to get all LED events:
    num_chunks = len(frame_batches)
    led_events = []
    print('num_chunks = ', num_chunks)

    for i in tqdm(range(num_chunks)[0:]):
        
        frame_data_chunk = moseq_video.load_movie_data(depth_path,
                                       frames=frame_batches[i],
                                       mapping=stream_names['IR'] ,movie_dtype=">u2", pixel_format="gray16be",
                                      frame_size=info['dims'],timestamps=timestamps,threads=8,
                                                      finfo=info)

        if i==0:
            plot_video_frame(frame_data_chunk,save_path)

        leds = get_led_data(frame_data_chunk,num_leds=4,sort_by='horizontal')
        
        time_offset = frame_batches[i][0] ## how many frames away from first chunk's  #### frame_chunks[0,i]
        
        led_events.append(get_events(leds,timestamps[frame_chunks[0,i]:frame_chunks[1,i]],time_offset,num_leds=4))
        
        
    led_events = np.concatenate(led_events)

    ## optional: save the events for further use
    np.savez('%s_led_events.npz' % os.path.splitext(depth_path)[0],led_events=led_events)


    ################################# Load the ephys TTL data #####################################

    ephys_ttl_path = glob('%s/*/*/events/Rhythm_FPGA-100.0/TTL_1/' % base_path)[0] 
    channels = np.load('%s/channel_states.npy' % ephys_ttl_path)
    ephys_timestamps = np.load('%s/timestamps.npy' % ephys_ttl_path)



    ephys_fs = 3e4

    led_fs = 30

    led_interval = 5 # seconds


    ## convert the LED events to bit codes:
    led_codes, latencies = events_to_codes(led_events,nchannels=4,minCodeTime=led_interval-1)
    led_codes = np.asarray(led_codes)


    ## convert the ephys TTL events to bit codes:
    ephys_events = np.vstack([ephys_timestamps[abs(channels)!=5],abs(channels[abs(channels)!=5])-1,np.sign(channels[abs(channels)!=5]) ]).T
    ephys_codes, ephys_latencies = events_to_codes(ephys_events,nchannels=4,minCodeTime=(led_interval-1)*ephys_fs)
    ephys_codes = np.asarray(ephys_codes)


    ## visualize a small chunk of the bit codes. do you see a match? 
    plot_code_chunk(ephys_codes,led_codes,ephys_fs,save_path)


    ################### Match the codes! ##########################

    matches = np.asarray(match_codes(ephys_codes[:,0] / ephys_fs,  ## converting the ephys times to seconds for matching (led times already in seconds)
                                  ephys_codes[:,1], 
                                  led_codes[:,0],
                                  led_codes[:,1],
                                  minMatch=10,maxErr=0,remove_duplicates=True ) )


    ## plot the matched codes against each other:
    plot_matched_scatter(matches,save_path)

    ####################### Make the models! ####################

    ephys_model = PiecewiseRegressor(verbose=True,
                               binner=KBinsDiscretizer(n_bins=100))
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
                               binner=KBinsDiscretizer(n_bins=100))
    video_model.fit(matches[:,1].reshape(-1, 1), matches[:,0])


    predicted_ephys_matches = video_model.predict(matches[:,1].reshape(-1, 1) )

    predicted_ephys_times = video_model.predict(led_codes[:,0].reshape(-1, 1) )

    joblib.dump(video_model, '%s/video_timebase.p' % save_path)
    print('Saved video model')


    print('Syncing complete. FIN')
