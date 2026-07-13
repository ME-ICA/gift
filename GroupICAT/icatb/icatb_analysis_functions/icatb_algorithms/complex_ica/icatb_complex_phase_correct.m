function [S, A, theta] = icatb_complex_phase_correct(S, A, mask)
%% Phase-ambiguity correction (Rodriguez 2012). Rotate each component so energy
% is maximally concentrated on the real axis; apply inverse to A so A*S is preserved.
% Input:  S    - (N x V) complex sources
%         A    - (M x N) complex mixing
%         mask - (V x 1 logical) high-quality voxels for the estimate (optional)
% Output: S, A (rotated), theta (N x 1 applied rotations, incl. pi sign fix)
[N, V] = size(S);
if (~exist('mask', 'var') || isempty(mask)); mask = true(V, 1); end
mask = logical(mask(:));
theta = zeros(N, 1);
for k = 1:N
    sk = S(k, :);
    smk = sk(mask);
    th = -0.5 * angle(sum(smk.^2));      % orient major axis to real axis
    sk = sk * exp(1i*th);
    % residual pi (sign) ambiguity: make real-part skewness positive
    re = real(sk(mask));
    sg = mean((re - mean(re)).^3);
    if (sg < 0); th = th + pi; sk = sk * exp(1i*pi); end
    S(k, :) = sk;
    A(:, k) = A(:, k) * exp(-1i*th);
    theta(k) = th;
end
end
