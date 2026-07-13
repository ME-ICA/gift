function [mask, Q, tau] = icatb_complex_phase_mask(Z, magMask)
%% Phase-quality mask (Rodriguez 2011). Q(v) = |sum_t Z(v,t)| / sum_t |Z(v,t)|.
% Invariant to a constant per-voxel phase offset (measures temporal variation).
% Input:  Z       - (V x T) complex data
%         magMask - (V x 1 logical) magnitude/brain mask (optional; default all true)
% Output: mask (V x 1 logical), Q (V x 1), tau (scalar Otsu threshold)
[V, ~] = size(Z);
if (~exist('magMask', 'var') || isempty(magMask)); magMask = true(V, 1); end
magMask = logical(magMask(:));
num = abs(sum(Z, 2));
den = sum(abs(Z), 2) + eps;
Q = num ./ den;
tau = icatb_otsu_threshold(Q(magMask));
mask = magMask & (Q > tau);
end
