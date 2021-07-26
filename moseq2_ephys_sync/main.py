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

from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer

import moseq2_extract.io.video as moseq_video

from moseq2_ephys_sync.video import get_mkv_stream_names, get_mkv_info
from moseq2_ephys_sync.extract_leds import gen_batch_sequence, get_led_data, get_events
from moseq2_ephys_sync.sync import events_to_codes, match_codes
from moseq2_ephys_sync.plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_matches_video_time,plot_video_frame
from moseq2_ephys_sync.arduino import load_arduino_data, list_to_events

import pdb

def sync(base_path, second_source='ephys_ttl', led_loc=None):
    """
    Takes timestamps of a 4-bit sequence from two sources and creates a piecewise linear model to sync the timestamps
    ----
    Inputs:
        base_path: path to the .mkv and any other files needed
        second_source: whether to sync the mkv with Open Ephys TTL data ("epys_ttl") or the arduino text data ("arduino_text")
        led_loc: useful for extracting LEDs if there's noise in corners of the bucket other than the LED corner
    """
    print(f'Running sync on {base_path} with {second_source} as second source.')

    ##### Setup #####

    # Built-in params (should make dynamic)
    led_interval = 5 # seconds, converted to num samples later
    mkv_chunk_size = 2000
    num_leds = 4
    ephys_fs = 3e4  # sampling rate in Hz
    arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'yaw', 'roll', 'pitch', 'accx', 'accy', 'accz', 'therm', 'olfled']
    arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']
    # arduino sampling rate calculated empirically below because it's not stable

    # Set up paths
    save_path = '%s/sync/' % base_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    depth_path = glob('%s/*.mkv' % base_path )[0]
    stream_names = get_mkv_stream_names(depth_path) # e.g. {'DEPTH': 0, 'IR': 1}
    info_path = '%s/info.json' % base_path  # make paths for info and timestamps. if they exist, don't recompute:
    timestamp_path = '%s/mkv_timestamps.csv' % base_path 

    # Load timestamps and mkv info
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
        timestamps = timestamps.values.flatten()
        
        with open(info_path, 'w') as f:
            json.dump(info, f)

    # Debugging
    print('info = ', info)
    print('timestamps.shape = ', timestamps.shape)

    
    ############### Cycle through the frame chunks to get all LED events: ###############
    
    # Prepare to load video (use frame batches like in moseq2-extract)
    frame_batches = gen_batch_sequence(info['nframes'], mkv_chunk_size,
                                           0, offset=0)
    num_chunks = len(frame_batches)
    mkv_led_events = []
    print('num_chunks = ', num_chunks)

    mkv_led_events_path = '%s_led_events.npz' % os.path.splitext(depth_path)[0]

    # Do the loading
    if not os.path.isfile(mkv_led_events_path):

        for i in tqdm(range(num_chunks)[0:]):
            
            frame_data_chunk = moseq_video.load_movie_data(depth_path,
                                           frames=frame_batches[i],
                                           mapping=stream_names['IR'], movie_dtype=">u2", pixel_format="gray16be",
                                          frame_size=info['dims'],timestamps=timestamps,threads=8,
                                                          finfo=info)

            if i==0:
                plot_video_frame(frame_data_chunk.std(axis=0),'%s/frame_std.pdf' % save_path)

            leds = get_led_data(frame_data_chunk=frame_data_chunk,
                            num_leds=num_leds,chunk_num=i, led_loc=led_loc, sort_by='horizontal',save_path=save_path)
            
            time_offset = frame_batches[i][0] ## how many frames away from first chunk's  #### frame_chunks[0,i]
            
            tmp_event = get_events(leds,timestamps[frame_batches[i]],time_offset,num_leds=num_leds)

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

        

    ############### Convert the LED events to bit codes ############### 
    mkv_led_codes, latencies = events_to_codes(mkv_led_events,nchannels=4,minCodeTime=led_interval-1)
    mkv_led_codes = np.asarray(mkv_led_codes)


    ################################# Load the second source of data and convert to codes #####################################

    if second_source=='ephys_ttl':
        ephys_ttl_path = glob('%s/**/TTL_*/' % base_path,recursive = True)[0]
        channels = np.load('%s/channel_states.npy' % ephys_ttl_path)
        ephys_timestamps = np.load('%s/timestamps.npy' % ephys_ttl_path)  # these are in sample number
        print('Assuming LED events in TTL channels 1-4...')
        ttl_channels = [-4,-3,-2,-1,1,2,3,4]
        ttl_bool = np.isin(channels, ttl_channels)
        ephys_events = np.vstack([ephys_timestamps[ttl_bool], abs(channels[ttl_bool])-1, np.sign(channels[ttl_bool])]).T
        source2_codes, ephys_latencies = events_to_codes(ephys_events, nchannels=4, minCodeTime=(led_interval-1)*ephys_fs)
        source2_codes = np.asarray(source2_codes)
        source2_fs = ephys_fs
        source2_times_in_seconds = source2_codes[:,0] / ephys_fs

    elif second_source=='arduino_text':
        assert num_leds==4, "Arduino code expects 4 LED channels"
        ino_data = load_arduino_data(base_path, arduino_colnames, arduino_dtypes, file_glob='*.txt')
        ino_timestamps = ino_data.time  # these are in milliseconds
        ino_events = list_to_events(ino_timestamps, ino_data.led1, ino_data.led2, ino_data.led3, ino_data.led4)
        ino_average_fs = 1/(np.mean(np.diff(ino_timestamps)))*1000  # fs = sampling freq in Hz
        source2_codes, _ = events_to_codes(ino_events, nchannels=4, minCodeTime=(led_interval-1)*1000)  # I think as long as the column 'timestamps' in events and the minCodeTime are in the same units, it's fine (for ephys, its nsamples, for arudino, it's ms)
        source2_codes = np.asarray(source2_codes)
        source2_fs = ino_average_fs
        source2_times_in_seconds = source2_codes[:,0] * 1000

    # Save the codes for use later
    np.savez('%s/codes.npz' % save_path, led_codes=mkv_led_codes, source2_codes=source2_codes)

    ## visualize a small chunk of the bit codes. do you see a match? 
    plot_code_chunk(source2_codes,mkv_led_codes,source2_fs,save_path)


    ################### Match the codes! ##########################

    # Returns two columns of matched event times
    matches = np.asarray(match_codes(source2_times_in_seconds,  ## converting the source2 times to seconds for matching (led times already in seconds)
                                  source2_codes[:,1], 
                                  mkv_led_codes[:,0],
                                  mkv_led_codes[:,1],
                                  minMatch=10,maxErr=0,remove_duplicates=True ))

    ## plot the matched codes against each other:
    plot_matched_scatter(matches,save_path)

    ####################### Make the models! ####################

    # Rename for clarity.
    ground_truth_source2_times = matches[:,0]
    ground_truth_video_times = matches[:,1]
    
    # Learn to predict video times from source2
    source2_model = PiecewiseRegressor(verbose=True,
                               binner=KBinsDiscretizer(n_bins=10))
    source2_model.fit(ground_truth_source2_times.reshape(-1, 1),
                     ground_truth_video_times)

    # Verify accuracy of predicted video event times from source2 event times
    predicted_video_matches = source2_model.predict(ground_truth_source2_times.reshape(-1, 1) )
    time_errors = (predicted_video_matches - ground_truth_video_times) 
    plot_model_errors(time_errors,save_path)

    # Verify accuracy of all predicted times
    predicted_video_times = source2_model.predict(source2_times_in_seconds.reshape(-1, 1) )
    plot_matches_video_time(predicted_video_times,source2_codes,mkv_led_codes,save_path)

    # Save
    joblib.dump(source2_model, '%s/ephys_timebase.p' % save_path)
    print('Saved ephys model')


    ### Repeat but vice versa
    video_model = PiecewiseRegressor(verbose=True,
                               binner=KBinsDiscretizer(n_bins=10))
    video_model.fit(ground_truth_video_times.reshape(-1, 1), 
                    ground_truth_source2_times)

    predicted_ephys_matches = video_model.predict(ground_truth_video_times.reshape(-1, 1) )
    predicted_ephys_times = video_model.predict(mkv_led_codes[:,0].reshape(-1, 1) )

    joblib.dump(video_model, '%s/video_timebase.p' % save_path)
    print('Saved video model')


    print('Syncing complete. FIN')



if __name__ == "__main__" :

    ## take a config file w/ a list of paths, sync each of those and plot the results in a subfolder called /sync/

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str)
    parser.add_argument('-second_source', type=str)
    parser.add_argument('-led_loc', type=str)

    settings = parser.parse_args(); 

    base_path = settings.path #'/n/groups/datta/maya/ofa-snr/mj-snr-01/mj_snr_01_2021-03-24_11-06-33/'
    second_source = settings.second_source
    led_loc = settings.led_loc

    sync(base_path, second_source, led_loc)

    
