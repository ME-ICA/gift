function test_fixtures()
here = fileparts(mfilename('fullpath'));
f = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures', 'complex_sources.mat');
assert(exist(f, 'file') == 2, 'fixtures not generated — run make_complex_fixtures first');
d = load(f);
assert(~isreal(d.cS) && ~isreal(d.cX) && ~isreal(d.A), 'all arrays must be complex');
assert(isequal(size(d.cX), [d.N d.V]), 'cX shape');
assert(norm(d.cX - d.A*d.cS, 'fro') < 1e-9, 'cX must equal A*cS');
disp('PASS test_fixtures');
end
