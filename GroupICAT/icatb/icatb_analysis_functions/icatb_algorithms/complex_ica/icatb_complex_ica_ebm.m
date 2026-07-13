function W = icatb_complex_ica_ebm(data)
%% Complex ICA by entropy bound minimization (wrapper around Adalí-lab CEBM).
% Input:  data - (components x volume) complex matrix (N x T)
% Output: W    - (N x N) complex demixing matrix; icasig = W*data
%
% Wraps complex_ICA_EBM (Li & Adalí 2010). Requires nf_table.mat on the path (shipped
% alongside this file). GPL v3 — see CEBM.m header; cite Li & Adalí 2010.
if isreal(data)
    error('icatb_complex_ica_ebm:realInput', 'Complex ICA-EBM requires complex-valued data.');
end
W = complex_ICA_EBM(data);
end
