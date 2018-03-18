function [frequency_sliding,bands,bandpow,bandphases] = MODAL(signal,params)
%  Multiple Oscillation Detection Algorithm (MOD-AL)

%  Provides the instantaneous power, phase, and frequency of a (neural) signal 
%  in adaptively identified bands in which power exceeds a global and 
%  (optionally) a local 1/f fit  of the signal "background"
%  Andrew J Watrous October 2017

% Instantaneous frequency estimates obtained using "Frequency Sliding" methods 
% http://mikexcohen.com/data/Cohen2014_freqslide.pdf

%Inputs:
% signal - signal to analyze.  Could be any neural timeseries data.
% params must include include:
%   srate - sampling rate of the signal in Hz
%   wavefreqs - frequencies to sample for background fitting. recommend max
%   frequency to be at or above 30Hz for a good 1/f fit estimate

%optional params are
%bad_data: boolean vector of bad data. 1 == bad data to exclude from
%calculations.  must be length of signal.

%local_winsize_sec(default =10sec): vector of different winsizes for local fitting
%(eg. [1 5 10]) in seconds. if empty, does not compute local fitting/thresholding
%and will return  power, phase, and frequency estimates for all timepoints

%params.crop_fs = boolean.  Crop frequency estimates outside of the band

%wavecycles - wavelet cycles (default is 6)

% Outputs %%%
% frequency sliding- instantaneous frequency of signal in each band.(Bands X samples)
% bands - edges of each detected band. Bands X 2 (lower,upper edge)
% bandpow - average power of signal in each detected band. (Bands X samples)
% bandphase - instantaneous phase of signal in each detected band (Bands X samples)


%Key Steps.  Implementations of these steps are commented throughout code.
%1.  Adaptively identifies narrowband oscillations exceeding background 1/f
%    as in Lega, Jacobs, & Kahana (2012) Hippocampus

%2.  Calculates "frequency sliding" (MX Cohen 2014; JNeuroscience) in each
%    band.  Note that these frequency estimates are continuous values within a
%    band and are NOT rounded to wavefreqs.

%3.  By default, removes power,phase, and frequency
%    estimates where power is below local 1/f fit line

%4.  Removes frequency estimates outside of detected band occuring due to
%    phase slips, see Cohen 2014 JNeurosci. Figure 1B and caption.

%Other notes: 
% Requires Kahana lab eegtoolbox function "multiphasevec2" or substitute
% your own function which calculates power in a single Frequencies X Times matrix



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%begin code
%establish inputs
wavefreqs = params.wavefreqs;
if isfield(params,'local_winsize_sec') 
wins = params.local_winsize_sec*params.srate; %if empty, will not run local fitting
else
wins = params.srate.*10; %default, single window length of 10 seconds
end

if isfield(params,'wavecycles')
    wavecycles = params.wavecycles;
else
    wavecycles  = 6; %default to 6
end

if isfield(params,'crop_fs') %default is to crop 
    crop_fs = params.crop_fs;
else
    crop_fs = 0;
end

if size(signal,2)>size(signal,1) %make sure samples is first dimension
    signal = signal'; 
end

%ensure the signal is mean centered so that the 
%hilbert transform and pow/phase estimates are valid
signal= signal-nanmean(signal);

%Adaptive section
%from Kahana eegtoolbox.
% http://memory.psych.upenn.edu/files/software/eeg_toolbox/eeg_toolbox.zip
%extract frequencies X times matrix of power estimates using wavelets
[~,pow]= multiphasevec2(params.wavefreqs,signal',params.srate,wavecycles);


%deal with bad data by replacing power values during
%bad times with NaN. 
%These will be excluded when doing band identification 
if isfield(params,'bad_data')
    bad_idx = find(params.bad_data==1);
    pow(:,bad_idx) = NaN;
end

%Key step #1.  Do 1/f fit to adaptively identify bands
[bands,bandidx,bandpow] = GetBands(wavefreqs,pow);

%Key Step #2.  Frequency sliding code from MX Cohen.
%% filter data
% apply a band-pass filter with 15% transition zones.
FS = zeros(size(bands,1),length(signal)).*NaN; %Initialize FS (i.e.all frequency sliding estimates within a band)
bandphases = zeros(size(bands,1),length(signal)).*NaN; %Initialize phases
for iBand = 1:size(bands,1)
    freq_bands = bands(iBand,:);
    trans_width    = .15;
    idealresponse  = [ 0 0 1 1 0 0 ];
    filtfreqbounds = [ 0 (1-trans_width)*freq_bands(1) freq_bands(1) freq_bands(2) freq_bands(2)*(1+trans_width) params.srate/2 ]/(params.srate/2);
    filt_order     = round(2*(params.srate/freq_bands(1))); 
    filterweights  = firls(filt_order,filtfreqbounds,idealresponse);
    filtered_signal = filtfilt(filterweights,1,signal);
    
    %hilbert the filtered signal
    temphilbert = hilbert(filtered_signal);
    anglehilbert = angle(temphilbert);
    bandphases(iBand,:) = anglehilbert;
    
    %code from MX Cohen
    frompaper = params.srate*diff(unwrap(anglehilbert))/(2*pi); %code from fs paper cohen 2014
    frompaper(end+1) = NaN; %deal with fact that diff loses a sample
    time_wins = [.05 .2 .4]; %time windows in fractions of a second from MX Cohen
    orders = time_wins*params.srate;

    %window signal into 10 epochs to make it more 
    %tractable. surprisingly parfor doesn't appreciably speed this up...
    numchunks = 10;
    chunks = floor(linspace(1,length(frompaper),numchunks)); %make epochs
    
    meds = zeros(length(orders),length(frompaper));
    for iWin = 1:length(orders)%median filter using different window sizes. 
        for iChunk = 2:numchunks
            chunkidx = chunks(iChunk-1):chunks(iChunk)-1; %don't double count edges, last sample will be excluded.
            meds(iWin,chunkidx) = medfilt1(frompaper(chunkidx),round(orders(iWin)));
        end
    end
    
    %take the median value across different medians
    median_of_meds = median(meds);
    
    % Key Step #4. NaN out frequency estimates outside of the filter band
    clear below* above* outside*
    if crop_fs
    below_idx = (median_of_meds<bands(iBand,1));
    above_idx = (median_of_meds>bands(iBand,2));
    outside_idx = find([below_idx+above_idx]==1);
    median_of_meds(outside_idx)=NaN;
    end
    FS(iBand,:) = median_of_meds; %all frequency sliding estimates within band.
end


% Optional Key Step #3.  
%1/f fit on smaller timewindows and replace pow, phase, & frequency estimates below 1/f fit
%with NaNs
if size(bands,1)>0
    frequency_sliding = FS;zeros([size(FS) length(wins)]);
    for iW = 1:length(wins) %different local fitting window sizes
        winsize = wins(iW);
        for iWin = 1:winsize:length(signal)
        windex = iWin:iWin+winsize; %index of samples to use 
        if windex(end)>length(signal)
            windex = iWin:length(signal);
        end

        %added if statement nov 1 to deal with windows where all values are
        %NaN based on bad_data
        if sum(sum(isnan(pow(:,windex))))< (length(windex).*length(wavefreqs));% don't run if its all nans
        %inputs all frequency sliding estimates, outputs trimmed version
            [frequency_sliding(:,windex,iW)] = ...
            fit_one_over_f_windows(FS(:,windex),wavefreqs,pow(:,windex),bandidx);

        else
         frequency_sliding(:,windex,iW)= NaN;
        end
    end
    end
    
    frequency_sliding = nanmean(frequency_sliding,3); %take the nanmean over windwows, so any point where it exceeds 
    %the 1/f fit across window size(s)
else %no bands detected
    frequency_sliding = NaN;
end

%assign phase and bandpow to have NaNs where frequency estimates are nan
%ensuring we have estimates at consistent points across measures
bandpow(isnan(frequency_sliding))=NaN;
bandphases(isnan(frequency_sliding))=NaN;

%convert everything to single to save space
bandpow = single(bandpow);
bandphases = single(bandphases);
frequency_sliding = single(frequency_sliding);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%subfunctions below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%identify bands from 1/f fit
function  [freq_bands,bandidx,bandpow] = GetBands(wavefreqs,pow)
%inputs, wavefreqs and pow are not log-transformed
%outputs, freq_bands: edges of bands in hz
%         bandidx : indices of wavefreqs for each band
%         bandpow:  mean log power in each band

fz = log(wavefreqs); %log transform frequencies
mean_pow = log(nanmean(pow,2)); %nanmean to deal with pow values that are nan based on bad data
[b,~] = robustfit(fz,mean_pow); %key fitting step
fit_line = b(1)+b(2).*fz;
above1f = (mean_pow-fit_line')>0;
bw = bwlabel(above1f);
ctr=1;%band counter
for iBand = 1:max(unique(bw))
    idx = find(bw==iBand);
    if length(idx)>1  %make sure its actually a band and not a point frequency
        freq_bands(ctr,1) = wavefreqs(min(idx));
        freq_bands(ctr,2) = wavefreqs(max(idx));
        bandidx{ctr} = idx;
        bandpow(ctr,:) = log(mean(pow(idx,:)));
        crit_pow = mean(fit_line(idx));
        ctr=ctr+1;
    end
end
%end band identification subfunction

%Key step #3. remove frequency sliding estimates below local 1/f fit
function [frequency_sliding] = fit_one_over_f_windows(frequency_sliding,wavefreqs,pow,bandidx)
    %input is all frequency sliding estimates within a band
    %output is frequency sliding estimates above local 1/f fit

    %same as other subfunction
fz = log(wavefreqs);
local_mean_pow = log(nanmean(pow,2)); %nanmean nov 1 to deal with pow values that are nan based on bad data
    [b,~] = robustfit(fz,local_mean_pow);
    local_fit_line = b(1)+b(2).*fz;

%look to see if the frequency estimate at each
%moment in time is above the 1/f fit
logpow = log(pow);
 fitpow = repmat(local_fit_line,size(logpow,2),1)';
 powdiff = logpow-fitpow;
 threshpow = (powdiff>0);
 
 tmpfs= frequency_sliding;
 for iB = 1:length(bandidx)
     %get times where fs estimates are not nan
     idx1=[];
 idx1 = find(~isnan(frequency_sliding(iB,:))==1); %fs estimates not nan
 if ~isempty(idx1) %only do it if there are some non NaN FS estimates
 fswf=[];
    for iT = 1:length(idx1)
        fswf(iT) = dsearchn(wavefreqs',frequency_sliding(iB,idx1(iT))) ;
    end
    subz=sub2ind(size(threshpow),fswf,idx1);
    threshvalz=[];
    threshvalz = threshpow(subz);
    tmpfs(iB,idx1(find(threshvalz==0)))=NaN; %replace subthreshold FS estimates with NaN
 else
     tmpfs(iB,:)=NaN;
 end
 end

frequency_sliding = tmpfs;

 
     


