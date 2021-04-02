'''
Tools for creating bit codes from LED and ephys TTLs, and matching them
'''
import os


def events_to_codes(events, nchannels, minCodeTime): # swap_12_codes = 1,swap_03_codes=0
    """
    Parameters
    ----------
    events : 2d array
        Array of pixel clock events (single channel transitions) where:
            events[:,0] = times
            events[:,1] = channels
            events[:,2] = directions
    nchannels : int
        Number of pixel clock channels
    minCodeTime : int
        Minimum time (in samples) for a given code change
    
    Return
    ------
    codes : 2d array
        Array of reconstructed pixel clock codes where:
            codes[:,0] = time
            codes[:,1] = code
            codes[:,2] = trigger channel
        These codes are NOT offset for latencies of the triggered channel
    latencies : nchannels x nchannels list of lists
        List of channel-to-channel latencies measured from events.
        Should be used with offset_codes to correct code times for the triggered channel
    """
    assert len(events) > 0, "Events cannot be 0 length"
    assert len(events[0]) == 3, "Each event should be of length 3 not %i" % len(events[0])

    evts = np.array(copy.deepcopy(events))
    evts = evts[evts[:,0].argsort(),:] # sort events
    # get initial state by looking at first transitions
    state = []
    for i in range(nchannels): ## was range
        d = evts[np.where(evts[:,1] == i)[0][0],2]
        if d == 1:
            state.append(0)
        else:
            state.append(1)

    trigTime = evts[0,0]
    trigChannel = int(evts[0,1])
    trigDirection = int(evts[0,2])
    codes = []
    
    latencies = [ [ [] for x in range(nchannels)] for y in range(nchannels)]
    for ev in evts:
        if abs(ev[0] - trigTime) > minCodeTime:
            # new event
            code = state_to_code(state)
            

            codes.append((trigTime, code, trigChannel))
            trigTime = ev[0]
            trigChannel = int(abs(ev[1]))
            trigDirection = int(ev[2])
        # update state
        ch = int(abs(ev[1])) ## the channel on which this event happened. [0 or 1]
        state[ch] += int(ev[2]) ## += [-1 or +1]... 

        #print(state[ch],ch,int(ev[2]) )
        
        if not (state[ch] in [0,1]):
            #logging.debug("Invalid state found[%s] at %i, truncating" % (str(state), ev[0]))
            state[ch] = max(0,min(1,state[ch]))
        else:
            # only store on valid states and + transitions
            # TODO look at transition direction of trigger, did it go up or down? only look for THOSE events
            if (trigDirection == 1) and (int(ev[2]) == 1): latencies[trigChannel][ch].append(ev[0] - trigTime)
    
    # assume last code was complete
    code = state_to_code(state)

    if code != codes[-1][1]:
        codes.append((trigTime, code, trigChannel))
    


    return codes, latencies

def state_to_code(state):
    """
    Convert a pixel clock state list to a code
    
    Parameters
    ----------
    state : list
        List of 0s and 1s indicating the state of the pixel clock channels.
        state[0] = LSB, state[1] = MSB
    
    Returns
    -------
    code : int
        Reconstructed pixel clock code
    """
    return sum([state[i] << i for i in range(len(state))]) # << = bit shift operator


def match_codes(auTimes, auCodes, mwTimes, mwCodes, minMatch = 5, maxErr = 0,remove_duplicates=0):
    """
    Find times of matching periods in two code sequences
    
    Parameters
    -s---------
    audioTimes : list
        Times of audio codes
    audioCodes : list
        List of audio codes to match
    mwTimes : list
        Times of mworks codes
    mwCodes : list
        List of mworks codes
    minMatch : int
        Minimum match length (starting at the first index)
    maxErr : int
        Maximum number of matching errors
    
    Returns
    -------
    matches : 2d list
        List of matching times where:
            matches[:,0] = audioTimes
            matches[:,1] = mwTimes
    Notes
    -----
    Repeats in the code sequence will result in offset errors
    Setting maxErr > 0 will result in offset errors
    """
    auTimes = np.array(auTimes)
    auCodes = np.array(auCodes)
    mwTimes = np.array(mwTimes)
    mwCodes = np.array(mwCodes)
    
    # remove all duplicate audioCodes?
    if remove_duplicates:
        dels = np.where(np.diff(auCodes) == 0)[0] + 1
        auCodes = np.delete(auCodes,dels)
        auTimes = np.delete(auTimes,dels)
    
        dels = np.where(np.diff(mwCodes) == 0)[0] + 1
        mwCodes = np.delete(mwCodes,dels)
        mwTimes = np.delete(mwTimes,dels)
    
    # create lookup lists for each audioCode
    codes = np.unique(mwCodes)
    lookup = {}
    for c in codes:
        lookup[c] = np.where(auCodes == c)[0]
    
    # step through mwCodes, looking for audioTimes
    auI = -1
    matches = []
    for mwI in range(len(mwCodes)):
        matchFound = False
        code = mwCodes[mwI]
        for aui in lookup[code][np.where(lookup[code] > auI)[0]]:
            if match_test(auCodes[aui:],mwCodes[mwI:], minMatch, maxErr) and\
                    (auCodes[aui] == mwCodes[mwI]):
                matches.append((auTimes[aui], mwTimes[mwI]))
                offset = auTimes[aui] - mwTimes[mwI]
                auI = aui
                break
        # warn that code wasn't found?
    return matches

def match_test(au, mw, minMatch, maxErr):
    """
    Test if two codes sequeces match based on a minimum match length and maximum error
    
    Parameters
    ----------
    au : list
        List of audio codes to match
    mw : list
        List of mworks codes to match
    minMatch : int
        Minimum match length (starting at the first index)
    maxErr : int
        Maximum number of matching errors
    
    Returns
    -------
    match : bool
        True if codes matched, false if otherwise
    """
    mwI = 0
    auI = 0
    if len(mw) < minMatch or len(au) < minMatch:
        return False
    err = 0
    matchLen = 0
    while (mwI < len(mw)) and (auI < len(au)):
        if mw[mwI] == au[auI]:
            matchLen += 1
            if matchLen >= minMatch:
                # print "match!"
                return True
            mwI += 1
            auI += 1
        else:
            auI += 1
            err += 1
            if err > maxErr:
                # print "err"
                return False
    return False