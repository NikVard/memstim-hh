function data = file_read(fname, format)
    fid = fopen(fname);
    data_ = textscan(fid, format);
    data = data_{:};
end