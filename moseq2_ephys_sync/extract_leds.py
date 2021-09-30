'''
Tools for extracting LED states from video files
'''

import os
import numpy as np
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from moseq2_ephys_sync.plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_matches_video_time,plot_video_frame
import pdb


def get_led_data_from_rois(frame_data_chunk, led_roi_list, led_thresh=2e4, save_path=None):
    """
    Given pre-determined rois for LEDs, return sequences of ons and offs
    Inputs:
        frame_data_chunk: array-like of video data (typically from moseq_video.load_movie_data() but could be any)
        rois (list): ordered list of rois [{specify ROI format, eg x1 x2 y1 y2}] to get LED data from
        led_thresh (int): value above which LEDs are considered on. Default 2e4. In the k4a recorder, LEDs that are on register as 65535 = 6.5e4, off ones are roughly 1000.
        save_path (str): where to save plots for debugging if desired
    Returns:
        leds (np.array): (num leds) x (num frames) array of 0s and 1s, indicating if LED is above or below threshold (ie, on or off)
    """

    leds = []

    for i in range(len(led_roi_list)):


        pts = led_roi_list[i]

        led = frame_data_chunk[:, pts[0], pts[1]].mean(axis=1) #on/off block signals  (slice returns a nframes x npts array, then you get mean of all the pts at each time)

        led_on = np.where(np.diff(led) > led_thresh)[0]   #rise indices
        led_off = np.where(np.diff(led) < -led_thresh)[0]   #fall indices


        led_vec = np.zeros(frame_data_chunk.shape[0])
        led_vec[led_on] = 1
        led_vec[led_off] = -1

        leds.append(led_vec)

    leds = np.vstack(leds) #spiky differenced signals to extract times 

    return leds




def get_led_data_with_stds(frame_data_chunk, num_leds = 4, chunk_num=0, led_loc=None,
    flip_horizontal=False, flip_vertical=False, sort_by=None, save_path=None):
    
    
    if flip_horizontal:
        print('Flipping image horizontally')
        frame_data_chunk = frame_data_chunk[:,:,::-1]
    if flip_vertical:
        print('Flipping image vertically')
        frame_data_chunk = frame_data_chunk[:,::-1,:]
    
    frame_uint8 = np.asarray(frame_data_chunk / frame_data_chunk.max() * 255, dtype='uint8')
    

    std_px = frame_uint8.std(axis=0)    
    mean_px = frame_uint8.mean(axis=0)
    vary_px = std_px if np.std(std_px) < np.std(mean_px) else mean_px # pick the one with the lower variance

    # Initial thresholding
    thresh = threshold_otsu(vary_px)
    thresh_px = np.copy(vary_px)
    thresh_px[thresh_px<thresh] = 0

    # Initial regions
    edges = canny(thresh_px/255.) ## find the edges
    filled_image = ndi.binary_fill_holes(edges) ## fill its edges
    labeled_leds, num_features = ndi.label(filled_image) ## get the clusters
    
    
    # If too many features, try a series of steps


    # 
    # first try thresholding again, excluding very low values
    # if num_features > num_leds:
    #     print('Too many features, using second thresholding step...')
    #     thresh2 = threshold_otsu(thresh_px[thresh_px > 5])
    #     thresh_px[thresh_px < thresh2] = 0
    #     edges = canny(thresh_px/255.) ## find the edges
    #     filled_image = ndi.binary_fill_holes(edges) ## fill its edges
    #     labeled_leds, num_features = ndi.label(filled_image) ## get the clusters
    #     # plot_video_frame(labeled_leds,'%s/frame_%d_led_labels_secondThreshold.png' % (save_path,chunk_num))

    # If still too many features, check for location parameter and filter by it
    if (num_features > num_leds) and led_loc:
        print('Too many features, using provided LED position...')
        centers_of_mass = ndi.measurements.center_of_mass(filled_image, labeled_leds, range(1, np.unique(labeled_leds)[-1] + 1))  # exclude 0, which is background
        centers_of_mass = [(x/filled_image.shape[0], y/filled_image.shape[1]) for (x,y) in centers_of_mass]  # normalize
        # x is flipped, y is not
        if led_loc == 'topright':
            idx = np.asarray([((x < 0.5) and (y > 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
        elif led_loc == 'topleft':
            idx = np.asarray([((x > 0.5) and (y > 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
        elif led_loc == 'bottomleft':
            idx = np.asarray([((x > 0.5) and (y < 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
        elif led_loc == 'bottomright':
            idx = np.asarray([((x < 0.5) and (y < 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
        else:
            RuntimeError('led_loc not recognized')
        
        # Add back one to account for background
        idx = idx+1

        # Remove non-LED labels
        labeled_leds[~np.isin(labeled_leds, idx)] = 0
        num_features = len(idx)

        # Ensure LEDs have labels 1,2,3,4
        if not np.all(idx == np.array([1,2,3,4])):
            # pdb.set_trace()
            for i,val in enumerate([1,2,3,4]):
                labeled_leds[labeled_leds==idx[i]] = val

    # If still too many features, remove small ones
    if num_features > num_leds:
        print('OoOOoOooOooOops! Number of features (%d) did not match the number of LEDs (%d)' % (num_features,num_leds))
        
        ## erase extra labels:
        if num_features > num_leds:
            
            
            led_size_thresh = 20
            labels_to_erase = [label for label in np.unique(labeled_leds) if (len(np.where(labeled_leds==label)[0]) < led_size_thresh and label > 0) ]
            
            for erase in labels_to_erase:
                print('Erasing extraneous label #%d' % erase)
                labeled_leds[labeled_leds==erase] = 0

    ## assign labels to the LEDs
    labels = [label for label in np.unique(labeled_leds) if label > 0 ]
            
    
    ## get LED x and y positions for sorting
    leds_xs = [np.where(labeled_leds==i)[1].mean() for i in labels] 
    leds_ys = [np.where(labeled_leds==i)[0].mean() for i in labels]  
    

    if sort_by == None: ## if not specified, sort by where there's most variance    
        print('Sorting LEDs by variance...if no matches found, check LED sorting!')
        if np.std(leds_xs) > np.std(leds_ys): # sort leds by the x coord:
            sorting = np.argsort(leds_xs)
        else:
            sorting = np.argsort(leds_ys)
    elif sort_by == 'vertical':
          sorting = np.argsort(leds_ys)
    elif sort_by == 'horizontal':
        sorting = np.argsort(leds_xs)
    else:
        sorting = range(num_leds)
        print('Choose how to sort LEDs: vertical, horizontal, or by variance (None)')
    
    # Show led labels for debugging
    image_to_show = np.copy(labeled_leds)
    for i in range(1,5):
        image_to_show[labeled_leds==(sorting[i-1]+1)] = i
    plot_video_frame(image_to_show,'%s/frame_%d_led_labels.png' % (save_path,chunk_num) )

    
    led_thresh = 2e4

    leds = []

    for i in range(len(sorting)):

        led_x = np.where(labeled_leds==sorting[i]+1)[0]
        led_y = np.where(labeled_leds==sorting[i]+1)[1]

        led = frame_data_chunk[:,led_x,led_y].mean(axis=1) #on/off block signals

        led_on = np.where(np.diff(led) > led_thresh)[0]   #rise indices
        led_off = np.where(np.diff(led) < -led_thresh)[0]   #fall indices


        led_vec = np.zeros(frame_data_chunk.shape[0])
        led_vec[led_on] = 1
        led_vec[led_off] = -1

        leds.append(led_vec)

    leds = np.vstack(leds) #spiky differenced signals to extract times   
    
    
    return leds

def get_events(leds, timestamps):
    """
    Convert list of led ons/offs + timestamps into list of ordered events

    Inputs:
        leds(np.array): num leds x num frames
        timestamps (np.array): 1 x num frames
    """
    ## e.g. [123,1,-1  ] time was 123rd frame, channel 1 changed from on to off... 

    times = []
    directions = []
    channels = []

    direction_signs = [1, -1]
    led_channels = range(leds.shape[0]) ## actual number of leds in case one is missing in this particular chunk. # range(num_leds)
    
    for channel in led_channels:

        for direction_sign in direction_signs:

            times_of_dir = timestamps[np.where(leds[channel,:] == direction_sign)]  #np.where(leds[channel,:] == direction_sign)[0] + time_offset ## turned off or on
                        
            times.append(times_of_dir)
            channels.append(np.repeat(channel,times_of_dir.shape[0]))
            directions.append(np.repeat(direction_sign,times_of_dir.shape[0] ))


    times = np.hstack(times)
    channels = np.hstack(channels).astype('int')
    directions = np.hstack(directions).astype('int')
    sorting = np.argsort(times)
    events = np.vstack([times[sorting],channels[sorting],directions[sorting]]).T
      
    return events