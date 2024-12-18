clc; clear; close all;

demo=0;
% Add the MIKNN directory to path if needed
% addpath('');

% Directories (adjust as needed)
cleanDir = 'Clean audio directory';
degsDir = 'Degraded audio directories'; 

% Subdirectories of degsDir that contain degraded files
% Adjust these to match the actual directory names you have.
degSubDirs = {'0','2','4','6','8','n2','n4'};

% Target frequency (if applicable)
targetFreq = 22050;

% Get list of clean files
cleanFiles = dir(fullfile(cleanDir, '*.wav'));

% For storing results
results = {};
totalFiles = length(cleanFiles);
h = waitbar(0, 'Processing files...');

for i = 1:totalFiles
    cleanFileName = cleanFiles(i).name;
    cleanFilePath = fullfile(cleanDir, cleanFileName);
    
    % Try reading the clean audio
    [cleanAudio, cleanFs] = audioread(cleanFilePath);
    
    % Check sampling rate
    if cleanFs ~= targetFreq
        warning('Clean file %s does not match target frequency (%d). Resampling or skipping.', cleanFileName, targetFreq);
        % Depending on your needs, you can resample here:
        % cleanAudio = resample(cleanAudio, targetFreq, cleanFs);
        % cleanFs = targetFreq;
    end
    
    % Create the extended version of the clean audio
    silence = zeros(round(0.5*cleanFs),1); 
    clean_2x = [cleanAudio; silence; cleanAudio];
    
    % For each degraded subdirectory, find matching file and compute MIKNN
    for s = 1:length(degSubDirs)
        degSubDir = degSubDirs{s};
        degFilePath = fullfile(degsDir, degSubDir, cleanFileName);
        
        if ~exist(degFilePath, 'file')
            fprintf('No matching degraded file found for %s in subdir %s\n', cleanFileName, degSubDir);
            continue;
        end
        
        [degAudio, degFs] = audioread(degFilePath);
        
        % Check sampling rate match
        if degFs ~= cleanFs || degFs ~= targetFreq
            warning('Degraded file %s does not match target frequency (%d). Adjusting.', degFilePath, targetFreq);
            % Resample if needed
            % degAudio = resample(degAudio, targetFreq, degFs);
            % degFs = targetFreq;
        end
        
        % Create the extended version of the degraded audio
        deg_2x = [degAudio; silence; degAudio];
        
        % Run MIKNN
        % Make sure 'sr','off' and the targetFreq is correct.
        try
            [d_2x, d_2x_raw] = MIKNN(clean_2x, deg_2x, targetFreq, 'sr', 'off');
            fprintf('MIKNN Score for %s with degradation %s: %.2f %%\n', cleanFileName, degSubDir, d_2x);
            
            % Store result
            results = [results; {cleanFileName, degSubDir, d_2x, d_2x_raw}];
            if (demo == 1); break; end
        catch ME
            fprintf('Error computing MIKNN for %s with %s: %s\n', cleanFileName, degSubDir, ME.message);
            close(h)
        end
        if (demo == 1); break; end
    end
    if (demo == 1); break; end
    waitbar(i/totalFiles, h, sprintf('Processing file %d of %d...', i, totalFiles));
end
resultsTable = cell2table(results, ...
    'VariableNames', {'CleanFile','Degradation','MIKNNScore2x','MIKNNScore2xRaw'});

writetable(resultsTable, 'myresults.csv');
close(h)