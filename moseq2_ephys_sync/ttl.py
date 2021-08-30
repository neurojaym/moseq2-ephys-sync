import numpy as np

from glob import glob
import sync

def ttl_workflow(base_path, save_path, num_leds, led_blink_interval, ephys_fs):
    """
    
    """
    assert num_leds==4, "TTL code expects 4 LED channels, other nums of channels not yet supported"

    ephys_ttl_path = glob('%s/**/TTL_*/' % base_path,recursive = True)[0]
    channels = np.load('%s/channel_states.npy' % ephys_ttl_path)
    ephys_timestamps = np.load('%s/timestamps.npy' % ephys_ttl_path)  # these are in sample number
    ttl_channels = [-4,-3,-2,-1,1,2,3,4]
    ttl_bool = np.isin(channels, ttl_channels)
    ephys_events = np.vstack([ephys_timestamps[ttl_bool], abs(channels[ttl_bool])-1, np.sign(channels[ttl_bool])]).T
    codes, ephys_latencies = sync.events_to_codes(ephys_events, nchannels=num_leds, minCodeTime=(led_blink_interval-1)*ephys_fs)
    codes = np.asarray(codes)

    return codes