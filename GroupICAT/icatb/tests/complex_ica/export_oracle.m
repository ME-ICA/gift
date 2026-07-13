function export_oracle()
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
fixDir = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures');
if (~exist(fixDir, 'dir')); mkdir(fixDir); end
d = load(fullfile(fixDir, 'complex_sources.mat'));

% --- oracle decompositions ---
W = icatb_complex_ica_ebm(d.cX); Ahat = pinv(W); Shat = W*d.cX;   %#ok<NASGU>
save(fullfile(fixDir, 'oracle_cebm.mat'), 'W', 'Ahat', 'Shat', '-v7');
W = icatb_complex_nc_fastica(d.cX, 'log'); Ahat = pinv(W); Shat = W*d.cX;
save(fullfile(fixDir, 'oracle_ncfastica.mat'), 'W', 'Ahat', 'Shat', '-v7');

% --- ship the nonlinearity lookup tables verbatim ---
% These are structs (piecewise-polynomial spline forms); the Python track
% flattens them to a portable canonical format.
copyfile(which('nf_table.mat'),         fullfile(fixDir, 'nf_table.mat'));
copyfile(which('complex_nf_table.mat'), fullfile(fixDir, 'complex_nf_table.mat'));
disp('export_oracle done');
end
