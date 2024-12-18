function [d, d_raw] = MIKNN(x_cs, y_ps, fs_signal, varargin)
%==========================================================================
%==========================================================================
% MIKNN is Copyright (C) 2014 by Jalal Taghia and Rainer Martin
% at Institute of Communication Acoustics (IKA),
% Ruhr-Universität Bochum.

% This program is free software for academic and non-commercial
% use, to be used at user's own risk, no guarantee of performance.
% The user is permitted to copy, distribute and transmit only unaltered
% copies of this program.
%==========================================================================
%==========================================================================
% MIKNN instrumental intelligibility measure for MATLAB
% Version 1.0, October 14, 2014
% Implemented by JALAL TAGHIA
% Email: jalal.taghia@rub.de

% This program implements the instrumental intelligibility measure MIKNN.
% First, input signals are transformed into 15 subbands by using a 1/3 octave
% band filter bank and then mutual information between the amplitude envelopes
% of the reference and the test signals is estimated per subband using the
% k-nearest neighbor approach. For more details on the method, see [1].

% The intelligibility score (in unit of nats) is derived by taking the average
% of all values over subbands.

% In the normalization stage, the intelligibility score between the test signal
% and the reference signal is normalized with the intelligibility score
% between the reference signal and itself which has the maximum information.
% The normalized intelligibility score is presented in percent (%).

% REFERENCE:
% [1] Jalal Taghia and Rainer Martin, "Objective intelligibility measures based
% on mutual information for speech subjected to speech enhancement processing",
% IEEE Transactions on Audio, Speech, and Language Processing, vol. 22,
% no. 1, pp. 6-16, January 2014.
%==========================================================================
% MAIN INPUTS:
%             x_cs          a vector of clean speech (reference) signal or
%                           a matrix of clean speech (reference) signals.
%                           x_cs is always considered as the reference for
%                           measuring the intelligibility of the test signals.
%                           If x_cs is chosen a matrix of reference
%                           signals, then the size the matrix x_cs must be the
%                           same as the size of the matrix of test signals
%                           y_ps.
%                           If x_cs is a vector containing one reference
%                           signal then all the test signals in the matrix
%                           y_ps are evaluated with respect to the single
%                           reference signal in x_cs.

%             y_ps          a vector of test signal or a matrix of test signals
%                           (e.g., noisy and processed speech signals).

%             fs_signal     sampling frequency of input signals in Hz.


% OPTIONAL INPUTS:

%            'sr'           silence removal, can be set 'on' or 'off'. By
%                           default, it is 'on' i.e., silence removal is
%                           performed. You can set it to 'off' if there is almost no
%                           speech pause in the reference signal(s).

%            'kneig'        the k-nearest neighbor parameter (a number).
%                           By default, it is determined based on the input
%                           signal's length.

%==========================================================================
% MAIN OUTPUT
%            d              normalized value of instrumental intelligibility
%                           score in percent (%) for test signals.
%                           This is the main output of the function. It takes
%                           values between 0 and 100.

%                           The normalized instrumental score d is computed
%                           by dividing the intelligibility score between
%                           the test signal and the reference signal with
%                           the intelligibility score obtained between the
%                           reference signal and itself.

%                           Hence, it shows the percentage of information
%                           which the test signal contains about the reference
%                           signal.
%                           It has been shown in [1] that this measure has
%                           a high performance in predicting the intelligibility
%                           of speech subjected to speech enhancement processing.
%                           Its value should not be misinterpreted with the percent
%                           value of the word correct score (WCS) which is defined
%                           in a subjective listening test.
%                           For the latter, a map function is required to find the
%                           equivalent values for word correct scores.
%                           This program does not provide the map function.
%                           The map function should be derived based on the
%                           existing corpus and the noise conditions
%                           in the experiments.

% ADDITIONAL OUTPUT

%            d_raw          the value of instrumental intelligibility score
%                           for test signal in nats before the normalization
%                           takes place. The unit of d_raw is in nats.
%==========================================================================
% Usage Examples

%           [d] = MIKNN(x_cs, y_ps, fs_signal);
%                           The simplest use of function (the default setting).
%                           In this case, the silence removal is 'on'.
%                           The k-nearest neighbor (kneig) parameter is set
%                           automatically based on the length of input
%                           signal.

%
%           [d] = MIKNN(x_cs, y_ps, fs_signal,'sr','off');
%                           In this way, the silence removal is turned 'off'.

%           [d] = MIKNN(x_cs, y_ps, fs_signal,'kneig',100);
%                           In this way, you can set an arbitrary value for
%                           the k-nearest neighbor parameter.


%           [d,d_raw] = MIKNN(x_cs, y_ps, fs_signal,'sr','on', 'kneig',100);
%                           The usage of function in a complete form.
%==========================================================================
%==========================================================================
% check the length of input signals
if length(x_cs) ~= length(y_ps)
    error('Inpunt signals should have the same length!');
end

if size(y_ps,1) <= size(y_ps,2)
    y_ps = y_ps';                        % test signals are placed in columns
end

if size(x_cs,1) <= size(x_cs,2)
    x_cs = x_cs';                        % reference signals are placed in columns
end

if size(x_cs,2)==1
    v_flag = 'on';                 % there is a single reference signal in the input
    x_cs = repmat(x_cs(:,1),1,size(y_ps,2));
elseif (size(x_cs,2) < size(y_ps,2)) || (size(x_cs,2) > size(y_ps,2))
    v_flag = 'off';
    error(['The reference input x_cs should be either a vector or a',...
        ' matrix with the same size as the matrix of test signals y_ps.']);
else
    v_flag = 'off';
end
%--------------------------------------------------------------------------
% Default setting
if nargin<3
    error('Provide at least three inputs!');
end

sr = 'on'; % silence removal flag. By default, silence removal is
%           performed, you can set it to 'off' if there is almost no speech pause in
%           reference signal.
kneig_flag = 'off'; % By default, kneig is estimated automatically based on
%           the signal's length
%--------------------------------------------------------------------------
% check input arguments
if nargin == 3
    sr = 'on';
    display(['>> By default, the silence removal is performed. You can set it to "off"' ...
        ' if there is almost no speech pause in referece signal!']);
    string = sprintf(['>> By default, the k-nearest neighbor parameter is set proportional to the length of input signals.\n' ...
        'You can manually set it by assigning a positive value to the parameter "kneig".']);
    disp (string);
else
    if (rem(length(varargin),2)==1)
        error('>> Optional parameters should always go by pairs!');
    end
    
    for i=1:2:(length(varargin)-1)
        if ~ischar (varargin{i}),
            error (['>> Unknown type of optional parameter name (parameter' ...
                ' names must be strings).']);
        end
        switch lower (varargin{i})
            case 'sr'
                sr = lower (varargin{i+1});
                if strcmp(sr, 'on')
                    display('>> Silence removal is performed ...');
                elseif strcmp(sr, 'off')
                    display('>> Silence removal is not performed ...');
                else
                    error('>> Decide on silence removal by choosing "on" or "off"!');
                end
            case 'kneig'
                kneig = round(varargin{i+1});
                kneig_flag = 'on';
            otherwise
                error ('>> Provide a valid optional input argument!');
        end
    end
end
%--------------------------------------------------------------------------
% Check the length of input signal
if strcmp(sr,'off')
    if (length(x_cs)/fs_signal)<1
        error('Input signal(s) is (are) too short (i.e., less than 1s)!');
    elseif ( (length(x_cs)/fs_signal)>=1 ) && ( (length(x_cs)/fs_signal)<2 )
        warning('>> Input signal(s) is (are) too short (less than 2 s), causing inaccurate result!')
    else
    end
    sig_leng = length(x_cs);
    
elseif strcmp(sr,'on')
    x_np_len = zeros(1,size(x_cs,2));
    for kk = 1:size(x_cs,2)
        x_np = sr_func(x_cs(:,kk),x_cs(:,kk) , 40, (0.050*fs_signal));
        %              for silence removal: dynamic range 40 dB and frame length 50 ms
        x_np_len(kk)=length(x_np);
        if (x_np_len(kk)/fs_signal)<1
            error(['>> The speech-active part of the ',num2str(kk),...
                ' th reference signal is too short (less than 1 s)!']);
        elseif ( (x_np_len(kk)/fs_signal)<2 ) && ( (x_np_len(kk)/fs_signal)>=1 )
            warning(['>> The speech-active part of the ',num2str(kk),...
                ' th reference signal is too short (less than 2 s), causing inaccurate result!']);
        else
        end
    end
    sig_leng = mean(x_np_len);
else
end
%--------------------------------------------------------------------------
% check kneig parameter
if strcmp(kneig_flag, 'off')
    if strcmp(sr,'off')
        kneig = round( (sig_leng/fs_signal)*5 );        % The default value for the
        %                                               k-nearest neighbor parameter
        %                                               if the silence removal is
        %                                               not performed. This
        %                                               is a heuristically
        %                                               chosen value.
    elseif strcmp(sr,'on')
        kneig = round( (sig_leng/fs_signal)*5 );        % The default value for the
        %                                               k-nearest neighbor parameter.
        %                                               if the silence removal is
        %                                               performed. This
        %                                               is a heuristically
        %                                               chosen value.
    else
        error('');
    end
elseif strcmp(kneig_flag, 'on') % kneig parameter is given at the input argument!
    % do nothing
else
    error('');
end

if kneig<10
    kneig = 10;
    if strcmp(kneig_flag, 'off')
        warning(['>> The value of k-nearest parameter (kneig) should not be smaller than 10. ',...
            'The speech active part of your reference input signal is too short (less than 2s).'...
            ' Now, kneig is reset to 10. '...
            'To have more accurate performance, make sure that the speech active part of',...
            ' all reference signals is longer than 2s.']);
    elseif strcmp(kneig_flag, 'on')
        warning(['>> The value of k-nearest parameter (kneig) should not be smaller than 10 to have more accurate performance. ',...
            'Now, kneig is reset to 10.']);
    else
    end
end

display(['>> The chosen k-nearest neighbor parameter kneig =',num2str(kneig),'.']);
display('>> Program is running (the running time depending on the length of input signals may take a few seconds)...');
%--------------------------------------------------------------------------
% computing instrumental scores
N_test = size(y_ps,2);
d_vec = zeros(1,N_test);
for test =1:N_test
    x = x_cs(:,test);
    y = y_ps(:,test);
    d_vec(1,test) = intellscore(x, y, sr, kneig, fs_signal);
end
% normalization of intelligibility scores
if strcmp(v_flag,'on')
    Nfactor = intellscore(x_cs(:,1),x_cs(:,1), sr, kneig,fs_signal);
elseif strcmp(v_flag,'off')
    [AA,uniq,ccc]=unique(x_cs','rows'); % finds signals which are unique!
    N_uniq = length(uniq); % number of unique signals
    Nfactor_vec = zeros(1,N_uniq);
    for ii=1:N_uniq
        Nfactor_vec(1,ii) = intellscore(  x_cs(:,uniq(ii)),  x_cs(:,uniq(ii)), sr, kneig,  fs_signal  );
    end
    Nfactor = max(Nfactor_vec);
else
end
d = 100 * (d_vec / Nfactor); % normalized instrumental intelligibility score(s)
%                             in percentage for test signal(s).
d_raw = d_vec;              % instrumental intelligibility score(s) in nats
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%    subfunctions    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% intellscore
function d = intellscore(x, y, sr, kneig,fs_signal)
% initialization for DFT analysis and the one-third octave filter bank
fs          = 10000;                            % sample rate of proposed intelligibility measure
frame_len   	= 256;                              % window support or window length
% here frame update length is 50%
K           = 512;                              % FFT size
J           = 15;                               % Number of 1/3 octave bands
mn          = 150;                              % Center frequency of first 1/3 octave band in Hz.
H           = onethird_filterbank(fs, K, J, mn);           % Get 1/3 octave band matrix

%--------------------------------------------------------------------------
% resample signals if other samplerate is used than fs
if fs_signal ~= fs
    x	= resample(x, fs, fs_signal);
    y 	= resample(y, fs, fs_signal);
end
%----------------------------------------------------------------------
% remove silent frames
if strcmp(sr, 'on')
    dyn_range=40; % in dB ( human speech is normally perceived over a range of about 40 dB)
    N_frame    	= 500; % 50 ms frame length
    % %     figure
    % %     subplot(2,1,1),plot(y)
    [x, y] = sr_func(x, y , dyn_range, N_frame);
    % %     subplot(2,1,2),plot(y,'r')
    % %     soundsc(x,fs)
    % %     pause
end
%----------------------------------------------------------------------
% apply 1/3 octave band TF-decomposition
x_hat     	= Analysis_DFT(x, frame_len, frame_len/2, K); 	% apply short-time DFT to clean speech
y_hat     	= Analysis_DFT(y, frame_len, frame_len/2, K); 	% apply short-time DFT to processed speech

x_hat       = x_hat(:, 1:(K/2+1)).';         	% take clean single-sided spectrum
y_hat       = y_hat(:, 1:(K/2+1)).';        	% take processed single-sided spectrum
%--------------------------------------------------------------------------
X           = zeros(J, size(x_hat, 2));
Y           = zeros(J, size(y_hat, 2));

for i = 1:size(x_hat, 2)
    X(:, i)	= sqrt(H*abs(x_hat(:, i)).^2);      % apply 1/3 octave bands
    Y(:, i)	= sqrt(H*abs(y_hat(:, i)).^2);
end

d_interm  	= zeros(J, 1);

for j = 1:J
    d_interm(j, 1) = knnjt(X(j, :), Y(j, :), kneig);
end
d = mean( max(d_interm(:), 0) );    % final objective score
end
%==========================================================================
%==========================================================================
%% onethird_filterbank
function  FBmat = onethird_filterbank(fs, DFTlen, NB, fcf)
%   outputs:
%       FBmat:          matrix of 1/3 octave bands

%   inputs:
%       fs:         sampling frequency (Hz)
%       DFTlen:      DFT length
%       NB:         number of subbands
%       fcf:         center frequency of first 1/3 octave band in Hz
freq_vec = linspace(0, fs, DFTlen+1);
freq_vec = freq_vec(1:(DFTlen/2+1));
NumFbin = length(freq_vec);
k = 0:(NB-1);
cf_k  = 2.^(k/3)*fcf;
cf_ku  = 2.^((k+1)/3)*fcf;
cf_kd  = 2.^((k-1)/3)*fcf;
fd = sqrt(cf_k.*cf_kd);
fu = sqrt(cf_k.*cf_ku);
FBmat = zeros(NB, NumFbin);

for i = 1:NB
    [aaa, nf_ind]  = min((freq_vec-fd(i)).^2);
    fd(i)  = freq_vec(nf_ind);
    fd_ii  = nf_ind;
    %--------------------------%
    [aaa, nf_ind] = min((freq_vec-fu(i)).^2);
    fu(i) = freq_vec(nf_ind);
    fu_ii = nf_ind;
    FBmat(i,fd_ii:(fu_ii-1))= 1;
end

NF_active  = sum(FBmat, 2);
% finding actual number of subbands
NB_ac = find( ( NF_active(2:end) >= NF_active(1:(end-1)) ) & ( NF_active(2:end)~=0 ) ~=0, 1, 'last' )+1;
FBmat   = FBmat(1:NB_ac, :);
end
%==========================================================================
%==========================================================================
%% Analysis_DFT
function X_DFT = Analysis_DFT(x, N, K, DFTlen)
% Inputs:
%           x:      input signal (a vector)
%           N:      frame length
%           K:      overlap between frame_ind
% Outputs:
%           X_DFT:  DFT coefficients of the imput signal where its rows and
%                   columns indicate the DFT bin and time-frame indices
window  = hanning(N);
x = x(:);
frame_ind = 1:K:(length(x)-N);
Num_frame = length(frame_ind);
X_DFT     = zeros(Num_frame, DFTlen);
for i = 1:Num_frame
    sample_ind = frame_ind(i):(frame_ind(i)+N-1);
    win_x = x(sample_ind).*window;
    X_DFT(i, :) = fft(win_x, DFTlen);
end
end
%==========================================================================
%==========================================================================
%% sr_func
function [x_a ,y_a] = sr_func(x, y, dyn_range, N)
%   x:      reference input signal (a vector)
%   y:      the second input signal which in the reconstruction follows the
%           reference signal x.
%   N:      frame-length (number of samples samples)
%   dyn_range: dynamic range in dB

N= floor(N);
x = x(:); y  = y(:);
lenx = floor(length(x)/N)*N;
xmat = reshape(x(1:lenx),N,lenx/N);
xx = x(lenx+1:end);
Exx = xx.^2;
Exxmax = mean(Exx);
Exmat = xmat.^2;
Exmax = max([mean(Exmat),Exxmax]);
Ex = .5*db(x.^2);
Exmax = .5 * db(Exmax);
frame_get = (Ex>(Exmax - dyn_range));
x_a = x(frame_get);
y_a = y(frame_get);
end
%==========================================================================
%==========================================================================
%% knnjt
% The k-nearest neighbor (KNN) method proposed by
% Kraskov et al. 2004 for estimation of MI.

% -- Note: This implementation of KNN approach is only for one-dimensional input vectors 
% by JALAL TAGHIA 
function I_XY = knnjt(X, Y, k)
% >>>>> X and Y are two vectors of input data for which MI is estimated. Rows should show
% the dimension of each data 
% while number of columns shows the number of data points or realizations for
% each of random vectors X and Y
%>>>>>> k: the k-nearest neighbor parameter which has to be selected
% appropriately large enough to minimize the sampling (statistical) error while
% not too large resulting in the increase of systematic error.
X = X./max(X(:));
Y = Y./max(Y(:));
[dim, N] = size(X);
nx_ik = zeros(1, N);
ny_ik = zeros(1, N);
for i = 1: N
    Dx = X - repmat (X(:,i), 1,N);
    Dy = Y - repmat(Y(:,i), 1,N);
    dx = norm_mat (Dx, 'inf'); % a row-vector consisting of the norm of each column
    % any type of norm could be considered instead of 'inf' which means the
    % infinite norm. In the case of one-dimensional data (i.e., in the case
    % of our MI-KNN measure), it does not matter what we choose as the
    % norm.
    dy = norm_mat (Dy, 'inf');  % a row-vector consisting of the norm of each column
    Dz = [dx;dy];
    [dz, indz] = max(abs(Dz)); % infinite-norm
    [sdz_val, sdz_ind] = sort(dz);
    eps_z_ik = sdz_val(k); % it is (epsilon_i/2), the distance between the ith point zi and its k-nearest neighbour
    ind_z_ik = sdz_ind(1:k);
    % --------------------------
    nx_ik (1,i) = sum (dx < eps_z_ik); % number of points in X data whose distance
    % from xi is strictly less than eps_z_ik (epsilon_i/2)
    ny_ik (1,i) = sum (dy < eps_z_ik); % number of points in Y data whose distance
    % from yi is strictly less than eps_z_ik (epsilon_i/2)
end
% Note: the following equations are for the first method introduced by
% Kraskov et al 2004 which works better than the second method for
% low-dimensional data.
psi_x = psi(nx_ik+1); % psi is digamma function
psi_y = psi(ny_ik+1);
psi_z = psi_x + psi_y;
I_XY = psi(N) + psi(k) - mean(psi_z);
end
%==========================================================================
%==========================================================================
%% norm_mat
function [norme]= norm_mat (x_mat, type)
[dim, N] = size(x_mat);
if dim~=1
    if strcmp (type, 'inf')
        % find the maximum element columnwise
        [norme, ind] = max(abs(x_mat)); % infinite-norm
    elseif strcmp (type, 'norm2')
        norme = zeros (1,N);
        for n = 1: N
            norme(1,n) = norm (x_mat(:,n), 2);
        end
    elseif strcmp (type, 'norm1')
        norme = zeros (1,N);
        for n = 1: N
            norme(1,n) = norm (x_mat(:,n), 1);
        end
    end
    norme = norme(:)'; % output a row vector
else
    norme = abs(x_mat);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% THE END   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
