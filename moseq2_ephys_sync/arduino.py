import pandas as pd
import numpy as np
from glob import glob
import pdb

from . import sync

def arduino_workflow(base_path, save_path, num_leds, led_blink_interval, arduino_spec):
    """
    Workflow to get codes from arduino txt file. Note arduino sampling rate is calculated empirically below because it's not stable from datapoint to datapoint.
    
    """
    assert num_leds==4, "Arduino code expects 4 LED channels, other nums of channels not yet supported"
    arduino_colnames, arduino_dtypes = get_col_info(arduino_spec)
    ino_data = load_arduino_data(base_path, arduino_colnames, arduino_dtypes, file_glob='*.txt')
    ino_timestamps = ino_data.time  # these are in milliseconds
    ino_events = list_to_events(ino_timestamps, ino_data.led1, ino_data.led2, ino_data.led3, ino_data.led4)
    ino_average_fs = 1/(np.mean(np.diff(ino_timestamps)))*1000  # fs = sampling freq in Hz
    ino_codes, _ = sync.events_to_codes(ino_events, nchannels=4, minCodeTime=(led_blink_interval-1)*1000)  # I think as long as the column 'timestamps' in events and the minCodeTime are in the same units, it's fine (for ephys, its nsamples, for arudino, it's ms)
    ino_codes = np.asarray(ino_codes)
    ino_codes[:,0] = ino_codes[:,0] / 1000  ## convert to seconds

    return ino_codes, ino_average_fs



def get_col_info(spec):
    """
    Given a string specifying the experiment type, return expected list of columns in arudino text file
    """
    if spec == "fictive_olfaction":
        arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'yaw', 'roll', 'pitch', 'accx', 'accy', 'accz', 'therm', 'olfled']
        arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']
    elif spec == "odor_on_wheel":
        arduino_colnames = ['time', 'led1', 'led2', 'led3', 'led4', 'wheel']
        arduino_dtypes = ['int64', 'int64', 'int64', 'int64','int64', 'int64']
    return arduino_colnames, arduino_dtypes



def load_arduino_data(base_path, colnames, dtypes, file_glob='*.txt'):
    arduino_data = glob(f'{base_path}/{file_glob}')[0]
    dtype_dict = {colname: dtype for colname, dtype in zip(colnames, dtypes)}
    try:
        # Try loading the entire thing first. 
        data = pd.read_csv(arduino_data, header=0, names=colnames, dtype=dtype_dict, error_bad_lines=False)
    except ValueError:
        try:
            # If needed, try ignoring the last line. This is slower so we don't use as default.
            data = pd.read_csv(arduino_data, header=0, names=colnames, dtype=dtype_dict, error_bad_lines=False, warn_bad_lines=True, skipfooter=1)
        except:
            raise RuntimeError('Could not load arduino data -- check text file for weirdness. \
            Most common issues text file issues are: \
            -- line that ends with a "-" (minus sign), "." (decima) \
            -- line that begins with a "," (comma) \
            -- usually no more than one issue like this per txt file')
    return data


def list_to_events(time_list, led1, led2, led3, led4):
    """
    Transforms list of times and led states into list of led change events.
    ---
    Input: pd.Series from arduino text file
    ---
    Output: 
    events : 2d array
        Array of pixel clock events (single channel transitions) where:
            events[:,0] = times
            events[:,1] = channels (0-indexed)
            events[:,2] = directions (1 or -1)
    """
    led_states = [led1, led2, led3, led4]

    # Get lists of relevant times and events
    times = pd.Series(dtype='int64', name='times')
    channels = pd.Series(dtype='int8', name='channels')
    directions = pd.Series(dtype='int8', name='directions')
    for i in range(4):
        states = led_states[i]
        diffs = np.diff(states)
        events_idx = np.asarray(diffs != 0).nonzero()[0] + 1  # plus 1, because the event should be the first timepoint where it's different
        times = times.append(pd.Series(time_list[events_idx], name='times'), ignore_index=True)
        channels = channels.append(pd.Series(np.repeat(i,len(events_idx)), name='channels'), ignore_index=True)
        directions = directions.append(pd.Series(np.sign(diffs[events_idx-1]), name='directions'), ignore_index=True)
    events = pd.concat([times, channels, directions], axis=1)
    sorting = np.argsort(events.loc[:,'times'])
    events = events.loc[sorting, :]
    assert np.all(np.diff(events.times)>=0), 'Event times are not sorted!'
    return np.array(events)
