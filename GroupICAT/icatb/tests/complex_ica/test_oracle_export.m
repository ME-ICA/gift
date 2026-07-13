function test_oracle_export()
here = fileparts(mfilename('fullpath'));
fixDir = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures');
for f = {'oracle_cebm.mat','oracle_ncfastica.mat','nf_table.mat','complex_nf_table.mat'}
    assert(exist(fullfile(fixDir, f{1}), 'file') == 2, ['missing ' f{1}]);
end
d = load(fullfile(fixDir, 'complex_sources.mat'));
o = load(fullfile(fixDir, 'oracle_cebm.mat'));
assert(isequal(size(o.W), [d.N d.N]), 'oracle W shape');
% W recovers sources: W*A is a scaled permutation
G = abs(o.W*d.A); G = G ./ max(G, [], 2);
assert(all(sum(G > 0.5, 2) == 1), 'oracle CEBM must recover sources');
disp('PASS test_oracle_export');
end
