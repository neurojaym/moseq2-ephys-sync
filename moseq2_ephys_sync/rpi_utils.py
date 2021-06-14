
import skvideo
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import canny
import numpy as np
import pdb
import os

def load_rpi_vid(fname, num_frames=1500):
    return skvideo.io.vread(fname, num_frames=num_frames)


def get_rpi_test_rois(clip):
    """
    Can't load chunks of an MP4 efficiently, so I structure the analysis a bit differently.
    Find LED ROIs upfront, and then process each frame.
    This function finds the ROIs.
    
    Returns:
        leds_xs: list of x coords of the middle (mean) of each led 
        leds_ys: list of x coords of the middle (mean) of each led 
        sorting: what order the LEDs should be interpreted in. I manually update outside this function.
    """


    stdpx = np.std(clip, axis=0)
    
    ## threshold the image to get rid of edge noise:
    thresh = threshold_otsu(stdpx)
    thresh_px = np.copy(stdpx)
    thresh_px[thresh_px<thresh] = 0

    edges = canny(thresh_px/255.) ## find the edges
    filled_image = ndi.binary_fill_holes(edges) ## fill its edges
    labeled_leds, num_features = ndi.label(filled_image) ## get the clusters

    ## assign labels to the LEDs
    labels = [label for label in np.unique(labeled_leds) if label > 0 ]
            
    ## get LED x and y positions for sorting
    leds_xs = [np.where(labeled_leds==i)[1].mean() for i in labels] 
    leds_ys = [np.where(labeled_leds==i)[0].mean() for i in labels]  

    # sort by x pos in the test vid
    sorting = np.argsort(leds_xs)

    return leds_xs, leds_ys, sorting


def surrounding_slice(num):
    return slice(num-1, num+2)


def extract_rpi_leds(rpi_vid_path, totalFrames, leds_xs, leds_ys, sorting, base_path, redo_led_extract):
    """

    Returns:
        leds: 4 x T numpy array of led signals over the movie
    """
    rpi_led_path = '%s/rpi_leds.npy' % base_path
    if os.path.exists(rpi_led_path):
        with open(rpi_led_path,'rb') as f:
            led_vals = np.load(f)
    else:
        # For each frame in rpi vid, get led values
        leds_xs = np.round(np.array(leds_xs)).astype('int')
        leds_ys = np.round(np.array(leds_ys)).astype('int')
        videogen = skvideo.io.vreader(rpi_vid_path)
        led_vals = np.zeros((totalFrames, 4))
        for i,frame in enumerate(videogen):
            for j in range(len(sorting)):
                led_num = sorting[j]
                # led_vals[i,j] = frame[leds_ys[led_num], leds_xs[led_num], 2]
                ch3 = frame[surrounding_slice(leds_ys[led_num]), surrounding_slice(leds_xs[led_num]), 2]
                ch2 = frame[surrounding_slice(leds_ys[led_num]), surrounding_slice(leds_xs[led_num]), 1]
                led_vals[i,j] = np.mean(ch2) + np.mean(ch3)
        with open(rpi_led_path, 'wb') as f:
            np.save(f, led_vals, allow_pickle=False)

    # Extract on/off signals
    leds = []
    for i in range(4):
        led_vec = np.zeros(totalFrames)
        signal = led_vals[:,i]
        factor = 5
        convolved = np.convolve(signal, np.ones((factor,)), mode='same')/factor
        on_or_off = (convolved > np.mean(convolved)).astype('int') # 0 or 1
        led_on = np.where(np.diff(on_or_off) > 0.5)[0]   #rise indices
        led_off = np.where(np.diff(on_or_off) < -0.5)[0]   #fall indic
        led_vec[led_on] = 1
        led_vec[led_off] = -1
        leds.append(led_vec)
    leds = np.vstack(leds) #spiky differenced signals to extract times  
    return leds


def interpolate_missing_timestamps(ts, fps=30):
    """simple interpolation of missing timestamps (not multiple in a row)
        Modified from Arne's github:
         https://github.com/arnefmeyer/RPiCameraPlugin/blob/master/Python/rpicamera/util.py
         His version has a weird thing with subtracting off deltas...don't understand what that's up to
    """

    dt = 1. / fps
    missing = np.where(ts < 0)[0]

    for i in missing:

        if i == 0:
            ts[i] = ts[i+1] - dt
        else:
            ts[i] = ts[i-1] + dt

    return ts