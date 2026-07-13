function make_complex_fixtures()
%% Deterministic complex-source fixtures for cross-language validation.
% Writes complex_ica_fixtures/complex_sources.mat. Seeded — reproducible.
seed = 20260713; rng(seed);
N = 6; V = 4000;
% super-Gaussian complex sources with mildly noncircular structure
re = randn(N,V) .* (abs(randn(N,V)) .^ 1.5);
im = randn(N,V) .* (abs(randn(N,V)) .^ 1.5) .* 0.6;   % unequal variance -> noncircular
cS = re + 1i*im;
cS = cS - mean(cS, 2);                                 % zero-mean per source
A  = randn(N,N) + 1i*randn(N,N);
cX = A*cS;
outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', 'complex_ica_fixtures');
if (~exist(outDir, 'dir')); mkdir(outDir); end
save(fullfile(outDir, 'complex_sources.mat'), 'cS', 'A', 'cX', 'N', 'V', 'seed', '-v7');
fprintf('Wrote %s\n', fullfile(outDir, 'complex_sources.mat'));
end
