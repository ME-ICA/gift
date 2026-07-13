function test_dispatch_complex()
% all of icatb: icatb_icaAlgorithm depends on icatb_get_modality, which lives at the
% icatb/ root rather than under icatb_analysis_functions/
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..')));
% names appear in the algorithm list
algos = icatb_icaAlgorithm;
names = lower(cellstr(algos));
assert(any(strcmp(names, 'complex ica-ebm')), 'complex ica-ebm not registered');
assert(any(strcmp(names, 'complex nc-fastica')), 'complex nc-fastica not registered');
% dispatch runs and returns W, A, icasig with correct shapes
rng(1); N = 3; T = 1500;
S = randn(N,T) + 1i*randn(N,T); A0 = randn(N,N)+1i*randn(N,N); X = A0*S;
[~, W, A, icasig] = icatb_icaAlgorithm('complex ica-ebm', X);
assert(isequal(size(W), [N N]) && isequal(size(icasig), [N T]), 'ebm dispatch shapes');
[~, W2, A2, icasig2] = icatb_icaAlgorithm('complex nc-fastica', X);
assert(isequal(size(W2), [N N]) && isequal(size(icasig2), [N T]), 'ncfastica dispatch shapes');
disp('PASS test_dispatch_complex');
end
