'''
Tools for extracting LED states from video files
'''

import os
import numpy as np
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_multiotsu
from moseq2_ephys_sync.plotting import plot_code_chunk, plot_matched_scatter, plot_model_errors, plot_video_frame
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



def relabel_labeled_leds(labeled_led_img):
    """Relabels arbitrary non-zero labels to be 1,2,3,4... (0 is background)
    """
    vals = np.unique(labeled_led_img)
    for i,val in enumerate(vals):
        if i == 0:
            continue
        else:
            labeled_led_img[labeled_led_img == val] = i
    return labeled_led_img

def extract_initial_labeled_image(frames_uint8, movie_type, top_down=False):
    """Use std (usually) of frames to find LED ROIS. Since avi's recorded in top-down configuration are 16-bit, no OTSU thresholding needed.:
    """

    std_px = frames_uint8.std(axis=0)    
    mean_px = frames_uint8.mean(axis=0)
    vary_px = std_px if np.std(std_px) < np.std(mean_px) else mean_px 
    if top_down:    
    	# Initial regions from mask
    	edges = canny(vary_px/255.) ## find the edges
    	filled_image = ndi.binary_fill_holes(edges) ## fill its edges
    	labeled_led_img, num_features = ndi.label(filled_image) ## get the clusters
    else:
	# Get threshold for LEDs
        if movie_type == 'mkv':
            thresh = threshold_otsu(vary_px)
        elif movie_type == 'avi':
            thresh = threshold_multiotsu(vary_px,5)[-1]  # take highest threshold from multiple
        # Get mask
        thresh_px = np.copy(vary_px)
        thresh_px[thresh_px<thresh] = 0

        edges = canny(thresh_px/255.)
        filled_image = ndi.binary_fill_holes(edges) ## fill its edges
        labeled_led_img, num_features = ndi.label(filled_image) ## get the clusters
    
    return num_features, filled_image, labeled_led_img


def clean_by_location(filled_image, labeled_led_img, led_loc):
    """Take labeled led image, and a location, and remove labeled regions not in that loc
    led_loc (str): Location of LEDs in an plt.imshow(labeled_leds)-oriented plot. Options are topright, topleft, bottomleft, or bottomright.
    """
    centers_of_mass = ndi.measurements.center_of_mass(filled_image, labeled_led_img, range(1, np.unique(labeled_led_img)[-1] + 1))  # exclude 0, which is background
    centers_of_mass = [(x/filled_image.shape[0], y/filled_image.shape[1]) for (x,y) in centers_of_mass]  # normalize
    # Imshow orientation: x is the vertical axis of the image and runs top to bottom; y is horizontal and runs left to right. (0,0 is top-left)
    if led_loc == 'topright':
        idx = np.asarray([((x < 0.5) and (y > 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'topleft':
        idx = np.asarray([((x < 0.5) and (y < 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'bottomleft':
        idx = np.asarray([((x > 0.5) and (y < 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'bottomright':
        idx = np.asarray([((x > 0.5) and (y > 0.5)) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'bottomquarter':
        idx = np.asarray([(x > 0.75) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'topquarter':
        idx = np.asarray([(x < 0.25) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'right_quarter':
        idx = np.asarray([(y > 0.75) for (x,y) in centers_of_mass]).nonzero()[0]
    elif led_loc == 'leftquarter':
        idx = np.asarray([(y < 0.25) for (x,y) in centers_of_mass]).nonzero()[0]
    else:
        RuntimeError('led_loc not recognized')
    
    # Add back one to account for background. Ie, if 3rd center of mass was in correct loc, this corresponds to label 4 in labeled_leds
    idx = idx+1
    
    # Remove non-LED labels
    labeled_led_img[~np.isin(labeled_led_img, idx)] = 0
    
    # Relabel 
    labeled_led_img = relabel_labeled_leds(labeled_led_img)

    return labeled_led_img


def clean_by_size(labeled_led_img, led_size_thresh):
    
    ## erase small rois:
    labels_to_erase = [label for label in np.unique(labeled_led_img) if (len(np.where(labeled_led_img==label)[0]) < led_size_thresh[0] and label > 0) ]
    for erase in labels_to_erase:
        print('Erasing extraneous label #%d based on too small size...' % erase)
        labeled_led_img[labeled_led_img==erase] = 0
    
    labels_to_erase = [label for label in np.unique(labeled_led_img) if (len(np.where(labeled_led_img==label)[0]) > led_size_thresh[1] and label > 0) ]
    for erase in labels_to_erase:
        print('Erasing extraneous label #%d based on too large size...' % erase)
        labeled_led_img[labeled_led_img==erase] = 0

    # Relabel 
    labeled_led_img = relabel_labeled_leds(labeled_led_img)

    return labeled_led_img

def get_roi_sorting(labeled_led_img, led_labels, sort_by):
    ## get LED x and y positions for sorting
    leds_xs = [np.where(labeled_led_img==i)[1].mean() for i in led_labels] 
    leds_ys = [np.where(labeled_led_img==i)[0].mean() for i in led_labels]  
    
    # LEDs are numbered 1-4; figure out how to order them
    # If not specified, sort by where there's most variance    
    if sort_by is None: 
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
        Warning('Argument to sort_by not recognized, using variance')
        if np.std(leds_xs) > np.std(leds_ys): # sort leds by the x coord:
            sorting = np.argsort(leds_xs)
        else:
            sorting = np.argsort(leds_ys)
    
    return sorting

def extract_roi_events(labeled_led_img, led_labels, sorting, frame_data_chunk, movie_type,top_down=True):
    
    # List to hold events by frame
    leds = []
    
    for i in range(len(sorting)):
        led_x = np.where(labeled_led_img==led_labels[sorting[i]])[0]
        led_y = np.where(labeled_led_img==led_labels[sorting[i]])[1]
        led = frame_data_chunk[:,led_x,led_y].mean(axis=1) #on/off block signals
    
        # If using avi(8bit), the range is pretty small, so use otsu to pick a good dividing number, then simplify to 0 or 1.
        if movie_type == 'avi' and not top_down:
            led_on_thresh = threshold_otsu(led)
            detection_vals = (led > led_on_thresh).astype('int')  # 0 or 1 --> diff is -1 or 1
            led_event_thresh = 0
        elif movie_type == 'mkv':
            detection_vals = led
            led_event_thresh = 2e4
        elif movie_type == 'avi' and top_down:
            #print('top_down extracting...')
            detection_vals = led
            led_event_thresh = 2e4

        led_on = np.where(np.diff(detection_vals) > led_event_thresh)[0]   #rise indices
        led_off = np.where(np.diff(detection_vals) < (-1*led_event_thresh))[0]   #fall indices
        led_vec = np.zeros(frame_data_chunk.shape[0])
        led_vec[led_on] = 1
        led_vec[led_off] = -1
        leds.append(led_vec)
    
    leds = np.vstack(leds) # (nLEDs x nFrames), spiky differenced signals to extract times   

    return leds


def check_led_order(leds, num_leds):
    reverse = 0
    num_events_per_led = np.sum(leds!=0, axis=1)
    max_event_idx = np.where(num_events_per_led == np.max(num_events_per_led))[0]
    if max_event_idx == 0:
        reverse = 1
    elif max_event_idx == (num_leds-1):
        pass
    elif len(max_event_idx) > 1:
        if (0 in max_event_idx) and not ((num_leds-1) in max_event_idx):
            reverse = 1
        elif ((num_leds-1) in max_event_idx) and not (0 in max_event_idx):
            pass
        else:
            Warning('Multiple max events in LEDs and was not first or last in sort!')
    return reverse


def get_led_data_with_stds(frame_data_chunk, movie_type, num_leds = 4, chunk_num=0, led_loc=None,
    flip_horizontal=False, flip_vertical=False, sort_by=None, save_path=None):
    """
    Uses std across frames + otsu + cleaning + knowledge about the event sequence to find LED ROIs in a chunk of frames.
    In AVIs, since they're clipped to int8, cleaning is harder. Might be able to solve this by casting to int16 and bumping any value above 250 to 2^16, but it's risky.
    Note that AVI's collected in top-down configuration are still 16bit, and tehrefore do not need OTSU thresholding for LED detection.  
    frame_data_chunk: nframes, nrows, ncols
    movie_type (str): 'mkv' or 'avi'. Will adjust thresholding beacuse currently avi's with caleb's clipping (uint8) don't have strong enough std
    
    """

    # Set up threshold for MKV
    if movie_type == 'mkv':
        led_thresh = 2e4
    elif movie_type == 'avi':
        led_thresh = None  # dynamically determine below with otsu
    
    # Flip frames if requested
    if flip_horizontal:
        print('Flipping image horizontally')
        frame_data_chunk = frame_data_chunk[:,:,::-1]
    if flip_vertical:
        print('Flipping image vertically')
        frame_data_chunk = frame_data_chunk[:,::-1,:]
    
    print ('Processing Chunk #', chunk_num)
    # Convert to uint8
    frames_uint8 = np.asarray(frame_data_chunk / frame_data_chunk.max() * 255, dtype='uint8')
    
    # for top-down configuration, get rid of azure reflections in the center of image
    cutout_window = 60
    frames_uint8[:,int(frames_uint8.shape[1]/2)-cutout_window:int(frames_uint8.shape[1]/2)+cutout_window,
            int(frames_uint8.shape[2]/2)-cutout_window:int(frames_uint8.shape[2]/2)+cutout_window]=0 

	
    # Get initial labeled image
    num_features, filled_image, labeled_led_img = extract_initial_labeled_image(frames_uint8, movie_type, top_down=True)
    
    # If too many features, try a series of cleaning steps. Labeled_leds has 0 for background, then 1,2,3...for ROIs of interest

    # If still too many features, check for location parameter and filter by it
    if (num_features > num_leds) and led_loc:
        print('Too many features, using provided LED position...')
        labeled_led_img = clean_by_location(filled_image, labeled_led_img, led_loc)

    # Recompute num features (minus 1 for background)
    num_features = len(np.unique(labeled_led_img)) - 1

    # If still too many features, remove small ones
    if (num_features > num_leds):
        print('Oops! Number of features (%d) did not match the number of LEDs (%d)' % (num_features,num_leds))
        #size_thresh = (40,100)  # min,max
        size_thresh = (30,100)
        labeled_led_img = clean_by_size(labeled_led_img, size_thresh)
    

    # Show led labels for debugging
    image_to_show = np.copy(labeled_led_img)
    # for i in range(1,5):
    #     image_to_show[labeled_leds==(sorting[i-1]+1)] = i
    plot_video_frame(image_to_show, 200, '%s/frame_%d_led_labels_preEvents.png' % (save_path,chunk_num) )

    led_labels = [label for label in np.unique(labeled_led_img) if label > 0 ]
    assert led_labels == sorted(led_labels)  # note that these labels aren't guaranteed only the correct ROIs yet... but the labels should be strictly sorted at this point.
    print(f'Found {len(led_labels)} LED ROIs after size- and location-based cleaning...')        
    if len(led_labels)==0:
        print('no LEDs found...skipping chunk')
        return [] 
    # At this point, sometimes still weird spots, but they're roughly LED sized.
    # So, to distinguish, get event data and then look for things that don't 
    # look like syncing LEDs in that data.

    # We use led_labels to extract events for each ROI.
    # sorting will be a nLEDs-length list, zero-indexed sort based on ROI horizontal or vertical position.
    # leds wil be an np.array of size (nLEDs, nFrames) with values 1 (on) and -1 (off) for events,
        #  and row index is the sort value.
    # So led_labels[sorting[0]] is the label of the ROI the script thinks belongs to LED #1,
        # and leds[sorting[0]] is the sequence of ONs and OFFs for that LED.
    sorting = get_roi_sorting(labeled_led_img, led_labels, sort_by)
    leds = extract_roi_events(labeled_led_img, led_labels, sorting, frame_data_chunk, movie_type)


    # In the ideal case, there are 4 ROIs, extract events, double check LED 4 is switching each time, and we're done.

    # Re-plot labeled led img, with remaining four led labels mapped to their sort order.
    # Use tmp because if you remap, say, 2 --> 3 before looking for 3, then when you look for 3, you'll also find 2.
    image_to_show = np.copy(labeled_led_img)
    tmp_idx_to_update = []
    for i in range(len(sorting)):
        tmp_idx_to_update.append(image_to_show == led_labels[sorting[i]])
    for i in range(len(sorting)):
        image_to_show[tmp_idx_to_update[i]] = (i+1)
    plot_video_frame(image_to_show, 200, '%s/frame_%d_sort_order_postEvents.png' % (save_path,chunk_num) )

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
