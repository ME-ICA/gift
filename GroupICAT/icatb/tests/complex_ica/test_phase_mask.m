function test_phase_mask()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(3);
V = 500; T = 80;
% signal voxels: stable phase; noise voxels: random-walk phase
Z = zeros(V, T);
mag = 1 + 0.1*randn(V, T);
for v = 1:V
    if v <= 250
        phi = 0.1*randn(1, T);           % stable
    else
        phi = 2*pi*rand(1, T);           % random
    end
    Z(v, :) = mag(v, :) .* exp(1i*phi);
end
magMask = true(V, 1);

[mask, Q, tau] = icatb_complex_phase_mask(Z, magMask);
assert(mean(Q(1:250)) > mean(Q(251:end)) + 0.3, 'stable voxels must score higher Q');
assert(mean(mask(1:250)) > 0.8 && mean(mask(251:end)) < 0.2, 'mask must select stable voxels');

% PROPERTY: invariance to a constant per-voxel phase offset
offset = exp(1i * (2*pi*rand(V,1)));        % one constant phase per voxel
Z2 = Z .* repmat(offset, 1, T);
[mask2, Q2] = icatb_complex_phase_mask(Z2, magMask);
assert(max(abs(Q - Q2)) < 1e-10, 'Q must be invariant to constant per-voxel phase');
assert(isequal(mask, mask2), 'mask must be invariant to constant per-voxel phase');
disp('PASS test_phase_mask');
end
