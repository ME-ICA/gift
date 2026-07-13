function export_oracle()
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
fixDir = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures');
d = load(fullfile(fixDir, 'complex_sources.mat'));

% --- oracle decompositions ---
W = icatb_complex_ica_ebm(d.cX); Ahat = pinv(W); Shat = W*d.cX;   %#ok<NASGU>
save(fullfile(fixDir, 'oracle_cebm.mat'), 'W', 'Ahat', 'Shat', '-v7');
W = icatb_complex_nc_fastica(d.cX, 'log'); Ahat = pinv(W); Shat = W*d.cX;
save(fullfile(fixDir, 'oracle_ncfastica.mat'), 'W', 'Ahat', 'Shat', '-v7');

% --- portable lookup tables ---
export_nf('nf_table.mat',         fullfile(fixDir, 'nf_table.csv'));
export_nf('complex_nf_table.mat', fullfile(fixDir, 'complex_nf_table.csv'));
disp('export_oracle done');
end

function export_nf(matName, csvPath)
S = load(matName);                 % nf_table.mat defines nf1..nf8 (and grid vars)
vars = fieldnames(S);
% write each variable as its own CSV column-block: name on header row, values below
fid = fopen(csvPath, 'w');
for k = 1:numel(vars)
    v = S.(vars{k});
    fprintf(fid, '%s\n', vars{k});
    fprintf(fid, '%.17g\n', v(:));
    fprintf(fid, '\n');
end
fclose(fid);
end
