function [filtered_data] = filter_pass(data, fs, fl, fh, order)
% operates on columns
    
    fv = [];
    
    if ~isempty(fh)
        fv = [fh];
        filt_str = 'low';
        
        if ~isempty(fl)
            fv = [fl, fh];
            filt_str = 'bandpass';
        end
    else
        if ~isempty(fl)
            fv = [fl];
            filt_str = 'high';
        end
    end
    
    if ~isempty(fv)
        [z,p,k] = butter(order,[fv]*2/fs,filt_str);
        [sos,g] = zp2sos(z,p,k);
        filtered_data = filtfilt(sos,g,data);
    end

end
