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

% --- complex spatial sources over signal voxels; ~0 over noise voxels ---
Ssig = randn(N, Vsig) .* (abs(randn(N, Vsig)).^1.5);          % super-Gaussian maps
Ssig = Ssig .* repmat(exp(1i*0.05*randn(N,1)), 1, Vsig);      % small consistent source phase
S = [Ssig, 0.001*(randn(N, Vnoise) + 1i*randn(N, Vnoise))];   % (N x V)

% --- complex time mixing (T x N), full column rank ---
Mtc = randn(T, N) + 1i*randn(T, N);

% --- full data (V x T); overwrite noise voxels with random-phase series ---
Z = (Mtc * S).';                                             % (V x T)
Z(Vsig+1:end, :) = (0.5 + rand(Vnoise, T)) .* exp(1i*2*pi*rand(Vnoise, T));

% === Step 1: phase-quality mask over voxels (Task 7) ===
[mask, ~] = icatb_complex_phase_mask(Z, true(V, 1));
assert(mean(mask(1:Vsig)) > 0.8, 'mask must keep most signal voxels');
assert(mean(mask(Vsig+1:end)) < 0.05, 'mask must drop noise voxels');

% === Step 2: Hermitian PCA reduction of masked data (T x Vm) -> (N x Vm) ===
Zm = Z(mask, :).';                    % (T x Vm)
Zm = Zm - mean(Zm, 2);                % de-mean per timepoint
C  = (Zm * Zm') / size(Zm, 2);        % (T x T) Hermitian covariance
[U, d] = eig(C, 'vector');
[d, idx] = sort(real(d), 'descend'); U = U(:, idx);
Xr = (U(:, 1:N)' ./ sqrt(d(1:N))) * Zm;    % (N x Vm) whitened mixture

% === Step 3: complex ICA through the real GIFT dispatch (Task 3) ===
[~, W, A, icasig] = icatb_icaAlgorithm('complex ica-ebm', Xr);   %#ok<ASGLU>
assert(isequal(size(icasig), [N, size(Xr, 2)]), 'icasig must be N x Vm');

% recovery: each estimated source matches one true source (over masked voxels)
% up to permutation + complex scale/phase
Strue = S(:, mask);                                          % (N x Vm)
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
