function test_complex_nc_fastica()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(7);
N = 4; T = 3000;
% noncircular complex sources: unequal real/imag variance
S = (randn(N,T) + 1i*0.3*randn(N,T));
A = randn(N,N) + 1i*randn(N,N);
X = A*S;
W = icatb_complex_nc_fastica(X, 'log');
assert(isequal(size(W), [N N]), 'W must be N x N');
G = abs(W*A); G = G ./ max(G, [], 2);
assert(all(sum(G > 0.5, 2) == 1), 'W*A must be a scaled permutation');
% default typeStr
W2 = icatb_complex_nc_fastica(X);
assert(isequal(size(W2), [N N]), 'default typeStr must work');
disp('PASS test_complex_nc_fastica');
end
