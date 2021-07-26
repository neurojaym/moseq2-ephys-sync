import pandas as pd
import numpy as np
from glob import glob

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
            events[:,2] = directions
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
        directions = directions.append(pd.Series(np.sign(diffs[events_idx]), name='directions'), ignore_index=True)
    events = pd.concat([times, channels, directions], axis=1)
    sorting = np.argsort(events.loc[:,'times'])
    events = events.loc[sorting, :]
    assert np.all(np.diff(events.times)>=0), 'Event times are not sorted!'
    return np.array(events)
