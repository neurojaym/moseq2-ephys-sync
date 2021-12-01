import pandas as pd
import numpy as np
from glob import glob
import pdb

import moseq2_ephys_sync.sync as sync
import moseq2_ephys_sync.util as util

def arduino_workflow(base_path, save_path, num_leds, led_blink_interval, arduino_spec, timestamp_jump_skip_event_threshhold=100):
    """
    Workflow to get codes from arduino txt file. Note arduino sampling rate is calculated empirically below because it's not stable from datapoint to datapoint.
    
    Inputs:
        base_path (str): path to the .txt file
        num_leds (int): expects 4
        led_blink_interval: sets an upper bound on how fast the sync code can change. Useful for noisy vids.
        arduino_spec (str): Specifies what the column names should be in the data that gets read in. Current options are "fictive_olfaction" or "odor_on_wheel", which are interpreted below.
        timestamp_jump_skip_event_threshhold (int): if there is a jump in timestamps larger than this, skip any artifactual "event" that might arise because of it.
    """
    print('Doing arduino workflow...')
    assert num_leds==4, "Arduino code expects 4 LED channels, other nums of channels not yet supported"
    assert arduino_spec is not None, "Arduino source requires a spec for the column names and datatypes (see arg arduino_spec)"
    arduino_colnames, arduino_dtypes = get_col_info(arduino_spec)
    ino_data = load_arduino_data(base_path, arduino_colnames, arduino_dtypes, file_glob='*.txt')
    ino_timestamps = ino_data.time  # these are in milliseconds
    ino_events = list_to_events(ino_timestamps, ino_data.led1, ino_data.led2, ino_data.led3, ino_data.led4, tskip=timestamp_jump_skip_event_threshhold)
    ino_average_fs = 1/(np.mean(np.diff(ino_timestamps)))*1000  # fs = sampling freq in Hz
    ino_codes, _ = sync.events_to_codes(ino_events, nchannels=4, minCodeTime=(led_blink_interval-1)*1000)  # I think as long as the column 'timestamps' in events and the minCodeTime are in the same units, it's fine (for ephys, its nsamples, for arudino, it's ms)
    ino_codes = np.asarray(ino_codes)
    ino_codes[:,0] = ino_codes[:,0] / 1000  ## convert to seconds

    return ino_codes, ino_average_fs



def get_col_info(spec):
    """
    Given a string specifying the experiment type, return expected list of columns in arudino text file
    """
    if spec == 'old_fictive_olfaction':
        arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'yaw', 'roll', 'pitch', 'accx', 'accy', 'accz', 'therm', 'olfled']
        arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32', 'uint8']
    elif spec == 'fictive_olfaction':
        arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'yaw', 'roll', 'pitch', 'accx', 'accy', 'accz', 'therm', 'olfled', 'pwm']
        arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32', 'uint8', 'uint8']
    elif spec == 'basic_thermistor':
        arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'yaw', 'roll', 'pitch', 'accx', 'accy', 'accz', 'therm']
        arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'float64', 'float64', 'float64', 'float64', 'int32']
    elif spec == 'odor_on_wheel':
        arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'wheel', 'thermistor', 'odor_ttl']
        arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'int64', 'int64', 'uint8']
    elif spec == 'header':  # headers in txt files
        arduino_colnames = None
        arduino_dtypes = None
    return arduino_colnames, arduino_dtypes



def load_arduino_data(base_path, colnames, dtypes, file_glob='*.txt'):

    # Define header data types
    # Do not use unsigned integers!! Otherwise np.diff() will not be able to return negatives.
    header_val_dtypes = {
        'time': 'int64',
        'led1': 'int8',
        'led2': 'int8',
        'led3': 'int8',
        'led4': 'int8',
        'yaw': 'float64',
        'roll': 'float64',
        'pitch': 'float64',
        'accx': 'float64',
        'acc_x': 'float64',  # same as accz  
        'accy': 'float64',  
        'acc_y': 'float64',  # same as accy
        'accz': 'float64',
        'acc_z': 'float64',  # same as acc_z
        'therm': 'int16',
        'thermistor': 'int16',  # same as thermistor
        'olfled': 'int8',
        'ledState': 'int8',  # same as olfled
        'pwm': 'int8',
        'pwmVal': 'int8',  # same as pwm
        'mouseROI': 'int8',
        'odor_ttl': 'int8',
        'wheel': 'int64'    
    }

    # Find file
    arduino_data_path = util.find_file_through_glob_and_symlink(base_path, file_glob)
        
    # Check if header is present
    with open(arduino_data_path, 'r') as f:
        first_row = f.readline().strip('\r\n').split(',')
    if first_row[0] == 'time':
        header = 1
        colnames = first_row
        print('Found header in arduino file, using...')
    else:
        header = 0

    if header:
        dtype_dict = {col: header_val_dtypes[col] for col in colnames}
        data = pd.read_csv(arduino_data_path, header=0, dtype=dtype_dict, index_col=False)  # header=0 means first row
    else:
        dtype_dict = {colname: dtype for colname, dtype in zip(colnames, dtypes)}
        try:
            # Try loading the entire thing first. 
            data = pd.read_csv(arduino_data_path, header=0, names=colnames, dtype=dtype_dict, index_col=False)
        except ValueError:
            try:
                # If needed, try ignoring the last line. This is slower so we don't use as default.
                data = pd.read_csv(arduino_data_path, header=0, names=colnames, dtype=dtype_dict, index_col=False, warn_bad_lines=True, skipfooter=1)
            except:
                raise RuntimeError('Could not load arduino data -- check text file for weirdness. \
                Most common issues text file issues are: \
                -- line that ends with a "-" (minus sign), "." (decima) \
                -- line that begins with a "," (comma) \
                -- usually no more than one issue like this per txt file')
    return data


def list_to_events(time_list, led1, led2, led3, led4, tskip):
    """
    Transforms list of times and led states into list of led change events.
    ---
    Input: pd.Series from arduino text file
        tskip (int): if there is a jump in timestamps larger than this, skip any artifactual "event" that might arise because of it.
    ---
    Output: 
    events : 2d array
        Array of pixel clock events (single channel transitions) where:
            events[:,0] = times
            events[:,1] = channels (0-indexed)
            events[:,2] = directions (1 or -1)
    """
    led_states = [led1, led2, led3, led4]

    # Check for timestamp skips
    time_diffs = np.diff(time_list)
    skip_list = np.asarray(time_diffs >= tskip).nonzero()[0] + 1

    # Get lists of relevant times and events
    times = pd.Series(dtype='int64', name='times')
    channels = pd.Series(dtype='int8', name='channels')
    directions = pd.Series(dtype='int8', name='directions')
    for i in range(4):
        states = led_states[i]  # list of 0s and 1s for this given LED
        diffs = np.diff(states)
        events_idx = np.asarray(diffs != 0).nonzero()[0] + 1  # plus 1, because the event should be the first timepoint where it's different
        events_idx = events_idx[~np.isin(events_idx, skip_list)]  # remove any large time skips because they're not guaranteed to be synchronized
        times = times.append(pd.Series(time_list[events_idx], name='times'), ignore_index=True)
        channels = channels.append(pd.Series(np.repeat(i,len(events_idx)), name='channels'), ignore_index=True)
        directions = directions.append(pd.Series(np.sign(diffs[events_idx-1]), name='directions'), ignore_index=True)
    events = pd.concat([times, channels, directions], axis=1)
    sorting = np.argsort(events.loc[:,'times'])
    events = events.loc[sorting, :]
    assert np.all(np.diff(events.times)>=0), 'Event times are not sorted!'
    return np.array(events)
