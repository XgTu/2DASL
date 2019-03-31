function log_info( input_str )
time_str = datestr(now, 31);
info_str = strcat('[', time_str, '] ', input_str);
disp(info_str);
end