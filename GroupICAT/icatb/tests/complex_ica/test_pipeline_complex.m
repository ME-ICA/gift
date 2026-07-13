function test_pipeline_complex()
%% In-memory integration test of the wired complex pipeline (Tasks 3, 7, 8):
%%   phase-quality mask -> Hermitian PCA reduction -> complex ICA dispatch
%%   ('complex ica-ebm') -> phase-ambiguity correction. No file/batch I/O.
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..')));   % all of icatb (dispatch + complex_ica + deps)
rng(21);

N = 3;              % sources
Vsig = 1000;        % signal voxels (stable phase)
Vnoise = 300;       % noise voxels (random phase)
V = Vsig + Vnoise;
T = 60;             % timepoints

% Physical model of complex fMRI:  z(v,t) = rho(v,t) * exp(1i*phi(v))
%   - rho(v,t): POSITIVE magnitude series (static baseline + BOLD-like modulation)
%   - phi(v):   STATIC per-voxel background phase (B0/receiver), small spread
% This is what makes signal voxels phase-stable over time (which is what the
% phase-quality mask keys on) while keeping each component's voxels clustered
% along a line (which is what the phase-ambiguity correction needs). The Adali
% complex-fMRI simulation uses the same structure (per-voxel phase jitter ~ +/-10 deg).

% --- real, super-Gaussian spatial maps (non-Gaussian => ICA-separable) ---
Smaps = randn(N, Vsig) .* (abs(randn(N, Vsig)).^1.5);
Smaps = Smaps ./ max(abs(Smaps(:)));                      % scale to [-1, 1]

% --- real time courses (T x N), full column rank ---
tc = randn(T, N);

% --- static per-voxel background phase: small spread (+/-10 deg), as in the Adali sim ---
phi = (rand(1, Vsig)*20 - 10) / 180 * pi;

% --- positive magnitude: large static baseline + modulation ---
base = 100 + 10*rand(1, Vsig);                            % static baseline image
M = repmat(base, T, 1) + 5*(tc * Smaps);                  % (T x Vsig)
assert(all(M(:) > 0), 'magnitude must stay positive (baseline dominates modulation)');

% --- full data (V x T): signal voxels phase-stable; noise voxels random phase ---
Zsig = (M .* repmat(exp(1i*phi), T, 1)).';                % (Vsig x T)
Znoise = (0.5 + rand(Vnoise, T)) .* exp(1i*2*pi*rand(Vnoise, T));
Z = [Zsig; Znoise];                                       % (V x T)

% === Step 1: phase-quality mask over voxels (Task 7) ===
[mask, ~] = icatb_complex_phase_mask(Z, true(V, 1));
assert(mean(mask(1:Vsig)) > 0.9, 'mask must keep signal voxels (stable phase)');
assert(mean(mask(Vsig+1:end)) < 0.05, 'mask must drop noise voxels (random phase)');

% === Step 2: strip the static baseline, then Hermitian PCA reduce (T x Vm) -> (N x Vm) ===
Zm = Z(mask, :).';                    % (T x Vm)
% Per-VOXEL temporal de-mean: removes the static complex baseline image. (This is the
% deliberate axis choice for complex data -- see reference doc §5 -- and it leaves a clean
% mixture Zm = tc_dm * Strue, with Strue(k,v) = Smaps(k,v)*exp(1i*phi(v)).)
Zm = Zm - mean(Zm, 1);
C  = (Zm * Zm') / size(Zm, 2);        % (T x T) Hermitian covariance
[U, d] = eig(C, 'vector');
[d, idx] = sort(real(d), 'descend'); U = U(:, idx);
Xr = (U(:, 1:N)' ./ sqrt(d(1:N))) * Zm;    % (N x Vm) whitened mixture

% === Step 3: complex ICA through the real GIFT dispatch (Task 3) ===
[~, W, A, icasig] = icatb_icaAlgorithm('complex ica-ebm', Xr);   %#ok<ASGLU>
assert(isequal(size(icasig), [N, size(Xr, 2)]), 'icasig must be N x Vm');

% recovery: each estimated source matches one true source (over masked voxels)
% up to permutation + complex scale/phase. The sources ICA actually sees are the real
% maps carrying the voxel's static phase; noise voxels are not part of the signal model.
StrueFull = [Smaps .* repmat(exp(1i*phi), N, 1), zeros(N, Vnoise)];   % (N x V)
Strue = StrueFull(:, mask);                                  % (N x Vm)
corrM = abs(icasig * Strue') ./ ...
        (sqrt(sum(abs(icasig).^2, 2)) * sqrt(sum(abs(Strue).^2, 2))');
corrM = corrM ./ max(corrM, [], 2);
assert(all(sum(corrM > 0.5, 2) == 1), 'each estimated source matches one true source');

% === Step 4: phase-ambiguity correction (Task 8) ===
recon = A * icasig;
[icasig2, A2, ~] = icatb_complex_phase_correct(icasig, A, true(size(icasig, 2), 1));
assert(norm(recon - A2*icasig2, 'fro') / norm(recon, 'fro') < 1e-10, 'A*S preserved');
imagFrac = sum(imag(icasig2).^2, 2) ./ sum(abs(icasig2).^2, 2);
assert(all(imagFrac < 0.1), 'corrected maps concentrate energy on the real axis');

disp('PASS test_pipeline_complex');
end
