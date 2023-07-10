%%% This will show you how to use parquet read to get concatenated data
%%% More can be found here: https://www.mathworks.com/help/matlab/ref/parquetread.html


% Read the Parquet
T = parquetread('example.parquet');

% Drop more functions in here as you add them