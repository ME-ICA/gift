function W = icatb_complex_nc_fastica(data, typeStr)
%% Noncircular complex FastICA (wrapper around Adalí-lab nonCircComplexFastICAsym).
% Input:  data    - (components x volume) complex matrix (N x T)
%         typeStr - nonlinearity 'log' | 'kurt' | 'sqrt' (default 'log')
% Output: W       - (N x N) complex demixing matrix; icasig = W*data
%
% Wraps nonCircComplexFastICAsym (Novey & Adalí 2008). GPL v3 — see that
% file's header; cite Novey & Adalí 2008 (TSP).
if isreal(data)
    error('icatb_complex_nc_fastica:realInput', 'Noncircular complex FastICA requires complex-valued data.');
end
if (~exist('typeStr', 'var') || isempty(typeStr))
    typeStr = 'log';
end
[Ahat, ~] = nonCircComplexFastICAsym(data, typeStr);
W = pinv(Ahat);
end
