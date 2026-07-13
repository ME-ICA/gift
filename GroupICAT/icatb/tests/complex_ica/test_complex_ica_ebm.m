% test_complex_ica_ebm — smoke + recovery test for the CEBM wrapper
function test_complex_ica_ebm()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(42);
N = 4; T = 2000;
% Sources MUST be non-Gaussian: ICA cannot separate Gaussian sources (identifiability
% limit -- after whitening, any rotation of Gaussian data is equally valid). ICA-EBM
% maximizes non-Gaussianity, so it has nothing to exploit on Gaussian input. The vendored
% demo1.m likewise drives CEBM with (strongly non-Gaussian) QAM sources.
S = (randn(N,T) .* abs(randn(N,T)).^1.5) + 1i*(randn(N,T) .* abs(randn(N,T)).^1.5);
% mirror demo1.m preprocessing: zero-mean + power normalization
S = S - mean(S, 2);
S = sqrt(T) * S ./ sqrt(sum(abs(S).^2, 2));
A = randn(N,N) + 1i*randn(N,N);              % complex mixing
X = A*S;
W = icatb_complex_ica_ebm(X);
assert(isequal(size(W), [N N]), 'W must be N x N');
assert(~isreal(W), 'W must be complex');
% global matrix G = W*A should be a scaled permutation (one dominant entry per row)
G = abs(W*A);
G = G ./ max(G, [], 2);
dominant = sum(G > 0.5, 2);
assert(all(dominant == 1), 'W*A must be a scaled permutation (ISI check)');
disp('PASS test_complex_ica_ebm');
end
