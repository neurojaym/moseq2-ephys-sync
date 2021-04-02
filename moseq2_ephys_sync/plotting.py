import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
plt.rcParams['pdf.fonttype'] = 'truetype'
import numpy as np

## visualize a small chunk of the bit codes. do you see a match? 
def plot_code_chunk(ephys_codes,led_codes,ephys_fs,save_path):

    f,axarr = plt.subplots(1,2,dpi=600,sharex=True)

    axarr[0].plot(ephys_codes[:,0]/ephys_fs,ephys_codes[:,1],label='ephys bit codes')

    axarr[1].plot(led_codes[:,0],led_codes[:,1],label='video bit codes')

    plt.xlim([0,200])

    plt.xlabel('time (sec)')
    plt.ylabel('bit code')
    plt.legend()

    f.savefig('%s/bit_code_chunk.pdf' % save_path)

    plt.close(f)


## plot the matched codes against each other:
def plot_matched_scatter(matches,save_path):

    f = plt.figure(dpi=600)

    plt.plot([0,3600],[0,3600],c='k',lw=0.5)

    plt.scatter(matches[:,0],matches[:,1],s=1)

    plt.title('Found %d matches' % len(matches))

    plt.xlabel('time of ephys codes')
    plt.ylabel('time of video codes')

    f.savefig('%s/matched_codes_scatter.pdf' % save_path)

    plt.close(f)

## plot model errors:
def plot_model_errors(time_errors,save_path):

    f = plt.figure(dpi=600)
    ax = plt.hist(time_errors)

    plt.title('%.2f sec. mean abs. error in Ephys Code Times' % np.abs(np.mean(time_errors)))
    plt.xlabel('Predicted - actual matched video code times')

    f.savefig('%s/ephys_model_errors.pdf' % save_path)

    plt.close(f)


## plot the codes on the same time scale
def plot_matches_video_time(predicted_video_times,ephys_codes,led_codes,save_path):
    f = plt.figure(dpi=600)

    start,stop =  0,100
    plt.plot(predicted_video_times[start:stop] , ephys_codes[start:stop,1],lw=2,label='Predicted video times')

    plt.plot(led_codes[start:stop,0], led_codes[start:stop,1],alpha=0.5,lw=1,label='Actual video times')

    plt.legend()

    f.savefig('%s/matched_codes_video_time.pdf' % save_path)

    plt.close(f)


def plot_video_frame(frame,save_path):
    f = plt.figure(dpi=600)

    plt.imshow(frame.std(axis=0))
    plt.colorbar()

    f.savefig('%s/frame_std.pdf' % save_path)

    plt.close(f)

