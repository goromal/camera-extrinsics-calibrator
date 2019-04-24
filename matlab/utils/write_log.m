function [] = write_log(data, logname)
% takes a matrix of doubles and strings the columns together, writing one long 
% string of doubles to a binary file
sz = size(data);
Log = fopen(logname,'w');
fwrite(Log, reshape(data, [1, sz(1)*sz(2)]), 'double');
fclose(Log);

end