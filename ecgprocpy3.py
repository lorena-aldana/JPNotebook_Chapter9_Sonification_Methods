from scipy import signal, stats
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets, linear_model
# from sklearn.model_selection import train_test_split
import math
import os
import csv 
dir_path = os.path.dirname(os.path.realpath(__file__))

#Filter functions 

def hilbert_transform(sigfiltered):
    # # Hilbert transform
    ht = abs(signal.hilbert(sigfiltered))
    return ht

def bpIIR_filter(data,fc1,fc2, SR, order=4):
    '''This function filters (bandpass) the ECG signal
    and removes the DC component'''
    # Band pass filter design
    # Remove DC component
    sig_centered= data- np.mean(data) 
    sigarr = sig_centered

    cutoffpass = fc1 / (SR / 2.0);
    cutoffstop = fc2 / (SR / 2.0)  # 5.0 inferior cf, 70.0
    b, a = signal.iirfilter(order, [cutoffpass, cutoffstop], btype='bandpass', analog=False, ftype='butter')
    # Apply High pass filter
    signalfilt = signal.filtfilt(b, a, sigarr[:])
    return signalfilt

def hpIIR_filter(data, fc1, SR, order=2):
    cutoffpass = fc1 / (SR / 2.0);
    b, a = signal.iirfilter(order, cutoffpass, btype='high', analog=False, ftype='butter')
    signalfilt = signal.filtfilt(b, a, data)
    return signalfilt

def lpIIR_filter(data, fc1, SR, order=4):
    cutoffpass = fc1 / (SR / 2.0);
    b, a = signal.iirfilter(order, cutoffpass, btype='low', analog=False, ftype='butter')
    signalfilt = signal.filtfilt(b, a, data, method='gust')
    # signalfilt = signal.lfilter(b, a, data)
    return signalfilt

def bpFIR_filter(data, fil_len, fc1, fc2, SR):
    cutoffpass = fc1 / (SR / 2.0);
    cutoffstop = fc2 / (SR / 2.0)
    b = signal.firwin(fil_len, [cutoffpass, cutoffstop], pass_zero=False, window='hamming')
    signalfilt = signal.lfilter(b,[1.0],data)
    return signalfilt

def notch_filter(data, fc1, SR, Q):
    notchfreq = fc1 / (SR / 2.0);
    print ('filter created')
    b, a= signal.iirnotch(notchfreq, Q)
    print ('coeffs created')
    signalfilt = signal.filtfilt(b, a, data)
    print ('returning')
    return signalfilt

def notch_filter_02(data,fc1,fc2, SR, order=2):
    cutoffpass = fc1 / (SR / 2.0);
    cutoffstop = fc2 / (SR / 2.0)  # 5.0 inferior cf, 70.0
    b, a = signal.iirfilter(order, [cutoffpass, cutoffstop], btype='bandstop', analog=False, ftype='butter')
    # Apply bandstop
    signalfilt = signal.filtfilt(b, a, data)
    return signalfilt

def filtecg_dc_bandpass(sigarr, lp=0.5, hp=50, sr=1000.0, order=2):
    
    sig_centered= sigarr - np.mean(sigarr) #Remove DC component
    sigarr = sig_centered
    cf1 = lp / (sr/2)
    cf2 = hp / (sr/2)
    b, a = signal.iirfilter(order, [cf1, cf2], btype='band', analog=False, ftype='butter')
    return signal.filtfilt(b,a, sigarr[:], method="gust", axis=0) 

def filter_multi_channel(signal, sr=1000.0, fc1=0.5, fc2=50):
    '''This function filters an remove the DC component of a multichannel ECG signal'''
    channels = np.shape(signal)[1]
    filt_matrix = []
    for ch in range(channels):
        filtered = bpIIR_filter(signal[signal.columns[ch]], fc1, fc2, SR=sr)
        filt_matrix.append(filtered)
    return filt_matrix

#R peak detection functions


def R_peaks_detection_onelead(ampdata, timedata, thld):
    
    '''This function chooses one peak in the given time window.
    It is the core function in ecgprocpy3 for calculating R peaks'''

    detected_peaks=[]
    lastPeak = []
    aboveThreshold = False
    for x in range(len(ampdata)):
        lastAboveThreshold = aboveThreshold
        curValue = ampdata[x]
        if curValue > thld:
            aboveThreshold = True
        else:
            aboveThreshold = False

        if aboveThreshold== True:
            if len(lastPeak) == 0 or curValue > lastPeak[1]:
                lastPeak = [timedata[x], ampdata[x]]
        if lastAboveThreshold == True and aboveThreshold == False:
            detected_peaks.append(lastPeak)
            lastPeak = []

        lastAboveThreshold = aboveThreshold

    if len(detected_peaks)>0: #select max peak among peaks found
        peakamp,loc= max([(x[1], i) for i, x in enumerate(detected_peaks)])
        selectedpeak= detected_peaks[loc]
        return selectedpeak


def find_R_peaks_onechannel(sig, sr=1000.0, window=200, thpercentage = 0.65, fc1=0.5, fc2=40.0, plot = False):
    '''This function filters the signal, calculates the time stamps for the peaks function and find
    the R peaks in one ECG lead'''
    #Filter the signal
    sigf= bpIIR_filter(sig, fc1, fc2, SR=sr)
    #Hilbert transform
    ht = hilbert_transform(sigf)
    #Set threshold for peaks detection
    maxamp = np.max(ht)
    minamp = np.min(ht)
    thld = (np.abs(maxamp-minamp)*thpercentage) #Threshold is 75 of the amplitude in the signal
    #
    originalsigtoplot= sig 
    sig = ht #now the sig variable contains the hilbert transform 
    #Calculate time stamps
    tref=0.0
    tlist=[] #Initialize list for timestamps
    dtsamples = (1 / sr)  # Time between samples
    peaks=[]
    peaks_final_sel=[]
    size_an_window=window
    for count, element in enumerate(sig,1): # Start counting from 1
        if count % size_an_window == 0: 
            segment= sig[count-size_an_window:count] #Only calculate peaks of the first channel
            for x in range(len(segment)):  # Create time stamps accroding to sample rate
                tlist.append(tref)
                tref = tref + dtsamples
            times = tlist[len(tlist)-size_an_window:len(tlist)]
            peak_found=R_peaks_detection_onelead(segment, times, thld) #Calculate peaks in given lead
            peaks.append(peak_found)
    peaks_final_sel=[x for x in peaks if x is not None] #Time, amplitude
    r_in_samples = [x[0]*sr for x in peaks_final_sel] #Peak location in samples

    if plot == True:
        plt.clf()
        xax = np.linspace(0, len(sig)/sr, len(sig))
        th_arr= len(xax) * [thld]
        #Plot hilbert transform:
        plt.plot(xax, sig)
        plt.plot(xax,th_arr, 'r')
        #Plot Original signal
        plt.plot(xax, originalsigtoplot, 'g', alpha= 0.25)
        for i in range(len(peaks_final_sel)):
            plt.plot(peaks_final_sel[i][0],peaks_final_sel[i][1], "*" )
        plt.xlabel('Time [s]')
        plt.ylabel ('Amplitude [mV]')

    return peaks_final_sel

def find_R_peaks_multi_channel(signal, sr=1000.0, window=200):
    '''This function calculates peaks of a multichannel ECG signal'''
    channels = np.shape(signal)[1]
#     print (channels)
    all_peaks_matrix=[]
    for ch in range(channels):
        rpks = find_R_peaks_onechannel(signal[signal.columns[ch]],sr, window)
        all_peaks_matrix.append(rpks)
    return all_peaks_matrix


def hr(peaks, sr, N):
    '''Calculates the HR based every N peaks.
    At least N peaks are needed for the calculation.
    The first N values are equal to zero while there are enough peaks to calculate the heart rate'''
    HR = []
    for x in range(N):
        HR.append(0)
    for x in range(N, len(peaks)):
        pks_loc = [x[0] for x in peaks[x-N:x]]
        RRdeltas = [a - b for a, b in zip(pks_loc[1:], pks_loc[:-1])] 
        hrmean = np.mean(RRdeltas)
        HR.append(int(60 / hrmean))
    return HR


def update_thr(nmax, nmin, omax, omin):

    maxbuf=[]
    minbuf=[]
    buf_ct=[]

    if nmax>omax:
        omax=nmax
    if nmin<omin:
        omin=nmin
    maxbuf.append(omax)
    minbuf.append(omin)
    buf_ct.append([omax,omin])
    if len(maxbuf)>5: #200 ms
        maxbuf.pop(0)
        minbuf.pop(0)
    mean_max_buf=np.mean(maxbuf)
    mean_min_buf=np.mean(minbuf)
    nnmax=omax
    nnmin=omin
    if len(buf_ct) > 5:
        up_flag=1 #update threshold every 5 ctcles of nSamples
    else:
        up_flag=0
    return nnmax, nnmin, mean_max_buf, mean_min_buf, up_flag


def seg_dur_for_analysis(dt, sr=1000.0):

    '''This function calculates the duration of half QRS complex, the total QRS complex, isoelectricity  
    and the PR segment based on the heart rate or time difference between consecutive peaks'''

    dtinsamples = dt*sr
    total_qrs_width = 1/10.0*(dtinsamples) #samples #QRS is aprox 1/10 of the RR
    QRSdur = (total_qrs_width/2) #samples
    qt_interval= (dtinsamples)/2.5 #samples #QT is aprox 2.5 of the RR according to pdf
    #http://lifeinthefastlane.com/ecg-library/basics/qt_interval/   -- Bazetts formula QTC = QT / sqrt RR
    # qtc=((qt_interval/sr)/np.sqrt(dt)*sr)  #The formula is in seconds, then to samples 
    #Fredericia’s formula: QTC = QT / RR 1/3
    qtc=((qt_interval/sr)/np.cbrt(dt)*sr)  #The formula is in seconds, then to samples 

    pr_int_dur=(dtinsamples)/6.25 
    iso_dur= (dtinsamples - qtc - pr_int_dur)  #iso_dur in samples

    # print (("qrs: %s, qtc: %s, isodur: %s, pr: %s")%(QRSdur, qtc, iso_dur, pr_int_dur))

    return int(QRSdur), int(qtc),int(iso_dur), int(pr_int_dur)

#ST elevation functions

def st_amplitude_one_lead(sig, sr=1000.0, window=200, thpercentage = 0.65, fc1=0.5, fc2= 50, accuracy= 0.0001, st_length = 50, longer_qrs = 0, plot = False):
    '''This function estimates the ST elevation in each heart beat based on the amplitude difference between
    J point and TP segment'''

    first_deriv= np.gradient(sig) 
    jpoints = []
    jpoint_search_flag = 0 
    tppoints = []
    tpseg_searchflag = 0

    cur_hbeat = 0
    rrdeltas = []
    hr = []

    #TODO: Check effect of setting this variable before its calculation
    qrs_width = 50
    qtc_width = 400
    iso_dur = 50

    #Filter signal
    sigfil = bpIIR_filter(sig, fc1, fc2, sr)
    peaks = find_R_peaks_onechannel(sig, sr, window, thpercentage, fc1, fc2) #To avoid dobule filtering in the peaks detection
    sig = sigfil #Filetered signal for the plotting, not for the peaks

    #Determine dt in each heart beat
    for pks in range(1, len(peaks)):
        rri = peaks[pks][0] - peaks[pks-1][0]
        rrdeltas.append(rri)
        hr.append(60/rri)

    #Find J point and TP segment
    for sig_idx, val in enumerate (range(len(sig))):
        #If the maximum number of peaks has been reached
        if cur_hbeat < len(peaks):
            #Find sample where the peaks are
            if sig_idx == int(peaks[cur_hbeat][0]*sr):
                #Calculate segments durations
                if cur_hbeat < len(peaks)-1: #update segment values, there is one dt value less than peaks
                    qrs_width, qtc_width, iso_dur, _ = seg_dur_for_analysis(rrdeltas[cur_hbeat], sr)
                #Calculate J point
                jpoint_search_flag = 1
                search_start = int(sig_idx+qrs_width+longer_qrs) #Start is a peak
                search_end = int(sig_idx+(qtc_width/2.0)) #End is half of qtc
                for der_idx, valdev in enumerate(first_deriv[search_start:search_end]):
                    if valdev < accuracy and valdev > -(accuracy) and jpoint_search_flag==1:
                        jpoints.append(search_start+der_idx)
                        jpoint_search_flag = 0 #Only stores the first value found
                #Calculate T-P segment
                tpseg_searchflag = 1
    #             tpsearch_start = int(search_start+(qtc_width-(qrs_width*2)))
                tpsearch_start = int(search_start+der_idx+(qtc_width-(qrs_width*2)))
                tpsearch_end = int(tpsearch_start+iso_dur)
                for der_idx, valdev in enumerate(first_deriv[tpsearch_start:tpsearch_end]):
                    if valdev < accuracy and valdev > -(accuracy) and tpseg_searchflag==1:
                        tppoints.append(tpsearch_start+der_idx)
                        tpseg_searchflag = 0 #Only stores the first value found
                cur_hbeat = cur_hbeat+1


    if (len(jpoints)) != (len(tppoints)):
    	'''When the heartbeat is not complete in the signal segmentation, J and TP segments have different length values'''
        # print ('Mismatch between number of detected J points and TP segments')
        pass

    if plot == True:
        plt.clf()
        plt.plot(sig)
        plt.plot(first_deriv, alpha = 0.5)
        for x in range(len(jpoints)):
            plt.plot(jpoints[x], sig[jpoints[x]], 'r*') #Plot J points
            plt.plot(tppoints[x], sig[tppoints[x]], 'b<') #Plot TP segment


    #Calculate amplitude in the ST segment
    st_amp_inmv = []
    n_full_heartbeats = np.min([len(jpoints), len(tppoints)])
    for val in range(n_full_heartbeats):
        st_amp_inmv.append(np.mean(sig[jpoints[val]:jpoints[val]+st_length]) - np.mean(sig[tppoints[val]:tppoints[val]+st_length]))
    
    return st_amp_inmv, hr

def st_amplitude_multichannel(sig, sr = 1000.0, window = 200, thpercentage = 0.65, fc1=0.5, fc2= 50, st_length = 50, accuracy = 0.0001,plot = False):
    #Determine peaks
    pkmch= find_R_peaks_multi_channel(sig, sr = 1000.0)
    #Number of channles or leads in dataset
    num_ch= np.shape(pkmch)
    #Create empty matrix [ch,peaks]
    # st_matrix = np.zeros((num_ch[0], len(pkmch[0])))

    # st_matrix = np.zeros((num_ch[0], len(pkmch[0])))

    st_matrix = []
    hr_matrix = []
    #List pf leads
    leads_list = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13","L14", "L15"]
    l = itertools.cycle(leads_list)
    
    #Goes through all channels L1, L2 and fill ST matrix
    for ch in range(np.shape(sig)[1]): #These are the channels (#)
        
        #Choose a lead
        lead = next(l)
        st, hr = st_amplitude_one_lead(sig[lead], sr, window, thpercentage, fc1, fc2, st_length, accuracy, plot = False)

        print (("Len ST vector is %s, ch is: %s") %(len(st), ch))

        # st_matrix[ch] = st

        st_matrix.append(st)
        hr_matrix.append(hr)
    
    return st_matrix, hr_matrix

#Arrhythmia Functions

def predict_next_R_peak(sig, peaks, sr, avgvalue, plot = False):

	'''This function predicts the next R peak occurrence according to the previous n peaks given.
	avgvalue = n peaks, used to predict next onset'''

	rpf_ma = np.zeros(np.shape(peaks)[0])
	rpf_lr = np.zeros(np.shape(peaks)[0])
	rpf_ma_cumul = np.zeros(np.shape(peaks)[0])
	rpf_lr_cumul = np.zeros(np.shape(peaks)[0])
	mav = np.zeros(np.shape(sig)[0])

	# Linear regression model
	model = linear_model.LinearRegression()

	#Numer of peaks found
	Np = np.shape(peaks)[0]
	for i in range(Np-1):
		imi = i - avgvalue
		if imi < 0: 
			imi = 0
		start_sample = int(peaks[i][0] * sr) #In samplesfor moving average and plotting
		end_sample = int (peaks[i+1][0] * sr) 
		m_average = np.mean(sig[start_sample:end_sample])
		mav[start_sample:end_sample] = m_average #It is stationary
		#Moving average forecast
		avghrate = ((peaks[i][0]) - (peaks[imi][0]))/(i-imi+0.0001) #In seconds
		rpf_ma[i+1] = avghrate #This is the forecast of the moving average in seconds
		rpf_ma_cumul[i+1] = peaks[i][0]+avghrate #Cumulative forecast, includes value of peak, not just RR value

		#Linear regression forecast
		selectedPeaks = [x[0] for x in peaks[imi:i]] #Take last nth peaks in seconds
		sel_data = selectedPeaks

		if i>avgvalue:
			xdata=np.linspace(i,i+len(sel_data)-1,len(sel_data)-1) #xdata 
			rr_time= [p - q for p, q in zip(sel_data[1:], sel_data[:-1])] #y data = rr_time

			x = np.reshape(xdata, (-1, 1))  # Reshape to use scikit learn with 1d dataset
			y = np.reshape(rr_time, (-1, 1))  # Reshape to use scikit learn with 1d dataset
			#fit model
			model.fit(x, y)

			#predict next rr delta:
			rpf_lr[i+1] = model.intercept_ + ((i+1) * model.coef_)
			rpf_lr_cumul[i+1] = (peaks[i][0]) + (model.intercept_ + ((i+1) * model.coef_))


    #Plot moving average
	if plot == True:
	    
		plt.clf()
		plt.plot(np.arange(0,(len(sig[avgvalue:])/sr), 1/sr), sig[avgvalue:]) #Starts plot after nth sample, before there is no average
		plt.plot(np.arange(0,(len(sig[avgvalue:])/sr), 1/sr),  mav[avgvalue:], 'r-') #moving average
		for x in range(Np):
			plt.plot(peaks[x][0], peaks[x][1], "r>", alpha = 0.5)
			plt.plot(rpf_ma_cumul[x], peaks[x][1], "b>", alpha = 0.5)
			plt.plot(rpf_lr_cumul[x], peaks[x][1], "g>", alpha = 0.5)

	return peaks, rpf_ma, rpf_lr, rpf_ma_cumul, rpf_lr_cumul

def actual_vs_predicted_onset_one_lead(sig, sr=1000.0, window = 200, thpercentage = 0.65, avgvalue=5, fc1 = 0.5, fc2 = 40, plot = False):
       
    '''This function returns the time difference between actual and expected R peaks according to
    the moving average and linear regression forecasts.
    Returns time difference for moving average and returns time difference for linear regression.
    It also returns the R peaks'''
    
    #Filter signal
    #Filter for peaks in noisy signals
    peaks = find_R_peaks_onechannel(sig, sr, window, thpercentage, fc1, fc2, plot = False)
    #
    sigfil = bpIIR_filter(sig, 0.5, 50, sr) #Filter signal according to the medical standard
    sig = sigfil
      

    #Forecast
    rp, rpf_ma, rpf_lr, rpf_ma_cumu, rpf_lr_cumu = predict_next_R_peak(sig, peaks, sr, avgvalue)

    #dt between real peaks
    for x in range(len(peaks)):
        RR = [np.abs((a[0] - b[0])) for a, b in zip(peaks[1:], peaks[:-1])] 
        RR_ma = [np.abs((a - b)) for a, b in zip(rpf_ma_cumu[1:], rpf_ma_cumu[:-1])] 
        RR_lr = [np.abs((a - b)) for a, b in zip(rpf_lr_cumu[1:], rpf_lr_cumu[:-1])] 
    
    #Time difference between actual vs expected onset
    time_dif_ma = [a[0]-b for a, b in zip(rp, rpf_ma_cumu)] #Real peak minus predicted peak 
    time_dif_lr = [a[0]-b for a, b in zip(rp, rpf_lr_cumu)]

    if plot == True:
        plt.clf()
        for x in np.arange(avgvalue, len(RR)):
            plt.plot(x-avgvalue,RR[x], 'b.',  alpha = 0.7, markersize = 5)
        plt.plot(RR[avgvalue:], 'b', alpha = 0.4, markersize = 7, label = 'Real R peaks')    
        plt.plot(RR_ma[avgvalue:], 'g', alpha = 0.4, markersize = 7, label = 'Moving Average prediction')
        plt.plot(RR_lr[avgvalue:], 'm', alpha = 0.4, markersize = 7, label = 'Linear Regression prediction')
        plt.ylabel('Time [s]')
        plt.xlabel('Number of beats')
        plt.title('Time difference between R peaks')
        plt.ylim((np.min(RR)-0.5, np.max(RR)+0.5))
        plt.legend()

        
    return time_dif_ma, time_dif_lr, peaks, RR, rpf_ma_cumu, rpf_lr_cumu, RR_ma, RR_lr


def actual_vs_predicted_onset_multichannel(sig, sr=1000.0, window = 200, thpercentage = 0.65, avgvalue=5, plot = False):
    '''This function returns the time difference between actual and expected R peaks according to
    the moving average and linear regression forecasts.
    Returns time difference matrix for moving average and returns time difference matrix for linear regression.
    It also returns the R peaks'''

    time_dif_ma_matrix = []
    time_dif_lr_matrix = []
    peaks_matrix = []
    num_ch = np.shape(sig)[1]
    for ch in range(num_ch):
        td_ma, td_lr, peaks = actual_vs_predicted_onset_one_lead(sig[sig.columns[ch]], sr, window, thpercentage, avgvalue, plot)
        time_dif_ma_matrix.append(td_ma)
        time_dif_lr_matrix.append(td_lr)
        peaks_matrix.append(peaks)
    
    return time_dif_ma_matrix, time_dif_lr_matrix, peaks_matrix


#Create surrogate ECG signals
def create_ecg_signal_st(jpoint_amp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], cycles= 2,  qrsl= 100, jpointl= 80, stl = 200, tpl = 300, pwavel = 200, pql = 160, plot = False, savefile = False, filenumber = '01'):
    surro_signal = []
    for amps in range(len(jpoint_amp)):
        for cycle in range (cycles):
            qrs = signal.gaussian(qrsl, std=5)
            qrs = list(qrs)
            jpoint = np.zeros(jpointl)
            jpoint = [jpoint_amp[amps] for x in jpoint]  
            mu = 0; variance = 4
            sigma = math.sqrt(variance)
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, stl)
            st = stats.norm.pdf(x, mu, sigma)
            st = list(st)
            tp = np.zeros(tpl)
            tp = [0.0 for x in tp]
            mu = 0; variance = 18
            sigma = math.sqrt(variance)
            x2 = np.linspace(mu - 3*sigma, mu + 3*sigma, pwavel)
            pwave = stats.norm.pdf(x2, mu, sigma)
            pwave = list(pwave)
            pq = np.zeros(pql)
            pq = [0.0 for x in pq]
            ecgsignal = qrs[:-25] + jpoint + st[80:] + tp + pwave + pq 
            surro_signal = surro_signal + ecgsignal
            ##Add noise
            for i in range(len(surro_signal)):
                surro_signal[i] = surro_signal[i]+np.random.uniform(0.001, 0.003)
                
    if plot == True:
        plt.clf()
        plt.plot(surro_signal)
    
    N = len(surro_signal)
    if savefile == True:
        with open(dir_path + '/st_surrogate_data_python/novdic2018/'+'Py_ST_Surrogate'+filenumber+'.csv', 'w') as csvfile:
            xmldata = csv.writer(csvfile, delimiter=' ')
            for i in range(N): 
                xmldata.writerow([surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i]])
        
    return surro_signal

def create_ecg_signal_arrhythmia(stdeviations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], heartbeats= 3, plot = True, savefile = False, filenumber = '02'):
    surro_signal = []
    ecgsignal = []

    for std in range(len(stdeviations)):
        
        rr=np.abs(np.random.normal(1.0, stdeviations[std], heartbeats)) #Mean, std, N
    
        for val in range(len(rr)):
            
            RR = rr[val]
            hr_factor = 60/RR
            qrsl= int(100*RR); jpointl= int(80*RR)
            stl = int(150*RR); tpl = int(320*RR)
            pwavel = int(200*RR); pql = int(150*RR)
            # print (("qrsl %s, jpointl %s, stl %s, tpl %s, pwavel %s, pql %s")%(qrsl, jpointl, stl, tpl, pwavel, pql))
            # print (RR)
            
            qrs = signal.gaussian(qrsl, std=10)
            qrs = list(qrs)
            jpoint = np.zeros(jpointl)
            jpoint = [0.001 for x in jpoint]  
            
            mu = 0; variance = 10
            
            sigma = math.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, stl)
            st = stats.norm.pdf(x, mu, sigma)
            st = list(st)
            tp = np.zeros(tpl)
            tp = [0.0 for x in tp]
            mu = 0; variance = 30
            sigma = math.sqrt(variance)
            x2 = np.linspace(mu - 3*sigma, mu + 3*sigma, pwavel)
            pwave = stats.norm.pdf(x2, mu, sigma)
            pwave = list(pwave)
            pq = np.zeros(pql)
            pq = [0.0 for x in pq]
            ecgsignal = qrs + jpoint + st + tp + pwave + pq 
            surro_signal = surro_signal + ecgsignal
            ecgsignal = []

    ##Add noise
    for i in range(len(surro_signal)):
        surro_signal[i] = surro_signal[i]+np.random.uniform(0.01, 0.02)

    if plot == True:
        plt.clf()
        plt.plot(surro_signal)

    N = len(surro_signal)
    if savefile == True:
        with open(dir_path + '/study_final_data/Final_Arrhythmia_Det_Files/'+'Py_Arr_Surrogate'+filenumber+'.csv', 'w') as csvfile:
            xmldata = csv.writer(csvfile, delimiter=' ')
            for i in range(N): 
                xmldata.writerow([surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i],surro_signal[i]])

    return surro_signal
    

def HRV_dt_act_exp_ons(prediction):
    '''Calculates HRV based on the time difference between
    actual vs expected onset.
    The argument is the prediction made using the moving average or the linear regression methods.
    The HRV is given in ms''' 
    sdrr = 0
    for x in range(len(prediction)):
        RRdeltas = [np.abs((a - b)*1000) for a, b in zip(prediction[1:], prediction[:-1])] 
        sdrr = np.std(RRdeltas)
    return sdrr

def hrv_nn50(peaks):
    '''Calculates HRV based on the number of consecutive peaks that have a time difference larger
    than 50 ms.
    The HRV is a scalar indicating the number of heart beats where the threshold is exceded'''
    nn50 = 0
    pks_loc = [x[0] for x in peaks]
    #Calculate time difference between consecutive peaks
    RRdeltas = [np.abs((a - b)) for a, b in zip(pks_loc[1:], pks_loc[:-1])] 
    #Calculate nn50, if time dif between consecutive peaks is larger than 50 ms
    nndt = [np.abs((a - b)*1000) for a, b in zip(RRdeltas[1:], RRdeltas[:-1])]
    nn50 = [nn50+1 for x in nndt if x>50]
    nn50 = np.sum(nn50)
    return nn50

def P_wave_features(signal, peaks,  rr, sr, plot = False):
    pwaves = []
    pw_mean = []
    pw_std = []
    pw_count = []
    iso_ref = 0.0

    if plot == True:
        plt.clf()
        plt.plot(signal, 'r', alpha = 0.5)

    for rs in range(len(rr)):
        pq_estimate = (rr[rs]/3)*sr
        rpeak_dur_estimate = (rr[rs]/10)*sr
        qt_dur_estimate = (rr[rs]/2.5)*sr
        segment = signal[int(peaks[rs][0]*sr - pq_estimate):int(peaks[rs][0]*sr - rpeak_dur_estimate)]

        
        if plot == True:
            plt.plot(segment, 'b--')

        # seg_normalization = segment/np.abs(np.max(segment)) #Absolute in case max is a negative number  
        meanp = segment.mean()
        stdp = segment.std()
        if rs == 0: #First peak
            iso_ref = meanp #iso ref is the mean amplitude of the Pwave segment
        elif (rs > 0 and rs < len(rr)): #iso ref is the mean amplitude of the signal between consecutive peaks
            seg_between_peaks = signal[int(peaks[rs][0]*sr+qt_dur_estimate):int(peaks[rs+1][0]*sr - rpeak_dur_estimate)]
            # seg_between_peaks_norm = seg_between_peaks/ np.abs(np.max(seg_between_peaks))
            iso_ref = seg_between_peaks.mean()
        elif rs == len(rr): #Last peak
            iso_ref = meanp  #iso ref is the mean amplitude of the Pwave segment

        if plot == True:
            plt.plot(np.linspace(int(peaks[rs][0]*sr)-pq_estimate, int(peaks[rs][0]*sr), num=len(segment)), (len(segment) * [iso_ref]), 'g--')


        pwave_count = (segment> iso_ref + 0.01).sum() # medical ref = 15 mV 
        pwaves.append(segment)
        pw_mean.append(meanp)
        pw_std.append(stdp)
        pw_count.append(pwave_count/(rr[rs]*sr)) #This results in a factor, a healthy signal should be around 0.1, 10 percent. 

                
    return pwaves, pw_mean, pw_std, pw_count

