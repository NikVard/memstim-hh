function [PSD, fv] = calc_PSD(x, fs, varargin)
    % Step0: Argument parsing
    defaultNFFT = 0;
    checkSign = @(x) x>0;
    
    defaultSides = 'single';
    validSides = {'single', 'double'};
    checkSides = @(x) any(validatestring(x, validSides));

    p = inputParser;
    p.KeepUnmatched = false;

    addRequired(p, 'x', @isnumeric)
    addRequired(p, 'fs', @isnumeric)
    addOptional(p, 'NFFT', defaultNFFT, checkSign)
    addOptional(p, 'sides', defaultSides, checkSides)

    parse(p, x, fs, varargin{:})
        
%     if ~isempty(p.UsingDefaults)
%        disp('Using defaults: ')
%        disp(p.UsingDefaults)
%     end

    NFFT = p.Results.NFFT;
    sides = p.Results.sides;

    % Step1: FFT
    if (NFFT == 0)
        NFFT = length(x);
    end
    L = length(x);
    X = fft(x, NFFT);
    
    % Step2: freq. vector
    fv = -fs/2 : fs/NFFT : fs/2;

    % Step3: PSD
    if strcmp(sides, 'single')
        X = X(1:NFFT/2+1);
        fv = fv(NFFT/2+1:end);
    end
    PSD = (1/(fs*L)) * abs(X).^2;
    
end