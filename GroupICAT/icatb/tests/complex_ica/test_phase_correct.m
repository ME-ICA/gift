function test_phase_correct()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(5);
N = 4; V = 3000; M = 20;
% real-axis-aligned sources (elongated along real axis), then inject a known rotation
Sbase = (randn(N,V)) + 1i*(0.05*randn(N,V));      % energy mostly real
A0 = randn(M,N) + 1i*randn(M,N);
theta_true = [0.7; -1.1; 0.3; 2.0];
Srot = Sbase .* exp(1i*repmat(theta_true, 1, V));
Arot = A0 .* exp(-1i*repmat(theta_true.', M, 1)); % keep A*S invariant
mask = true(V,1);

% PROPERTY 1: X = A*S preserved by the whole correction
X_before = Arot*Srot;
[Sc, Ac, theta] = icatb_complex_phase_correct(Srot, Arot, mask);
assert(norm(X_before - Ac*Sc, 'fro') / norm(X_before, 'fro') < 1e-10, 'A*S must be preserved');

% PROPERTY 2: corrected sources are real-axis aligned (imag energy minimized)
imag_frac = sum(imag(Sc).^2, 2) ./ sum(abs(Sc).^2, 2);
assert(all(imag_frac < 0.05), 'corrected sources must concentrate energy on the real axis');

% PROPERTY 3: recovered rotation cancels the injected one (up to pi sign)
resid = mod(angle(sum(Sc.^2, 2)), pi);
assert(all(min(resid, pi - resid) < 1e-3), 'residual major-axis angle ~ 0 mod pi');
disp('PASS test_phase_correct');
end
