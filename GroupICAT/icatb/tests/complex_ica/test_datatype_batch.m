function test_datatype_batch()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..')));
% Minimal input file exercising the complex branch of the batch reader.
tmp = tempname; fid = fopen([tmp '.m'], 'w');
fprintf(fid, "dataType = 'complex';\n");
fprintf(fid, "read_complex_images = 'real&imaginary';\n");
fprintf(fid, "write_complex_images = 'real&imaginary';\n");
fclose(fid);
% Read just the dataType-relevant fields the way the batch reader does.
inp = icatb_eval_script([tmp '.m']);
assert(strcmpi(inp.dataType, 'complex'), 'batch input must carry dataType=complex');
disp('PASS test_datatype_batch');
end
