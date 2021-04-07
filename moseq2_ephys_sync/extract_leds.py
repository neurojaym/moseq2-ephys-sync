'''
Tools for extracting LED states from mkv files
'''
import os
import numpy as np
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    '''
    Generates batches used to chunk videos prior to extraction.

    Parameters
    ----------
    nframes (int): total number of frames
    chunk_size (int): desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns
    -------
    Yields list of batches
    '''

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out


def get_led_data(frame_data_chunk,num_leds = 4,flip_horizontal=False,flip_vertical=False,sort_by=None):
    
    
    ## cropping:
    #frame_data_chunk = frame_data_chunk[:,:,:-100]
    
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

    ## threshold the image to get rid of edge noise:
    thresh = threshold_otsu(vary_px)
    thresh_px = np.copy(vary_px)
    thresh_px[thresh_px<thresh] = 0

    
    edges = canny(thresh_px/255.) ## find the edges
    filled_image = ndi.binary_fill_holes(edges) ## fill its edges
    labeled_leds, num_features = ndi.label(filled_image) ## get the clusters
    

    
    if num_features != num_leds:
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

def get_events(leds,timestamps,time_offset=0,num_leds=2):

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
    channels = np.hstack(channels)
    directions = np.hstack(directions)
    
    sorting = np.argsort(times)
    
    
    events = np.vstack([times[sorting],channels[sorting],directions[sorting]]).T
      
    
    return events