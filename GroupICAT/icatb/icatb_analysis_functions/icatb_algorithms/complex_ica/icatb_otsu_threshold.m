function tau = icatb_otsu_threshold(x)
%% Otsu threshold on a vector x (256-bin histogram). Returns scalar tau.
x = x(:); x = x(isfinite(x));
lo = min(x); hi = max(x);
if (hi <= lo); tau = lo; return; end
nbins = 256;
edges = linspace(lo, hi, nbins+1);
counts = histc(x, edges); counts = counts(1:nbins);
p = counts / sum(counts);
omega = cumsum(p);
mu = cumsum(p .* (1:nbins)');
muT = mu(end);
sigma_b = (muT*omega - mu).^2 ./ (omega .* (1 - omega) + eps);
[~, k] = max(sigma_b);
centers = (edges(1:nbins) + edges(2:nbins+1))' / 2;
tau = centers(k);
end
