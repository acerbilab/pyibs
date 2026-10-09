% Checks of ibslike.m's behaviour under GNU Octave, for the port review.
% Run from the root of the PyIBS repository, with ../ibs checked out:
%   octave-cli --no-gui --norc -q dev/experiments/port-review_20261009/octave_checks.m
1;
warning('off', 'Octave:shadowed-function');
addpath('../ibs', 'dev/scripts/octave/compat', 'dev/scripts/octave/headless', ...
        'dev/experiments/port-review_20261009');
fprintf('Octave %s\n', version());
fun = @(x, dmat) rand(size(dmat, 1), 1) < x;

% One trial: the shapes of the per-trial arrays (F-3).
rand('state', 1);
for vec = [true, false]
  for p = [0.3, 1]
    o = struct('Vectorized', vec, 'Nreps', 5);
    [nl, v, ef, out] = ibslike(fun, p, true, [], o);
    fprintf(['N = 1, Vectorized = %d, p = %.1f: nlogL %.4f, nlogLvar %.4f; ' ...
             'size(nlogL_trials) %s, size(nlogLvar_trials) %s, ' ...
             'sum(nlogL_trials) %.4f, sum(nlogLvar_trials) %.4f\n'], ...
            vec, p, nl, v, mat2str(size(out.nlogL_trials)), ...
            mat2str(size(out.nlogLvar_trials)), sum(out.nlogL_trials), ...
            sum(out.nlogLvar_trials));
  end
end

% MaxIter = Inf disables the cap (F-4); MaxIter = 2 is the control.
rand('state', 2);
for vec = [true, false]
  o = struct('Vectorized', vec, 'Nreps', 3, 'MaxIter', Inf);
  [nl, v, ef] = ibslike(fun, 0.001, true(3, 1), [], o);
  fprintf('MaxIter = Inf, Vectorized = %d, p = 0.001: nlogL %.3f, exit flag %d\n', ...
          vec, nl, ef);
  o.MaxIter = 2;
  try
    ibslike(fun, 0.001, true(3, 1), [], o);
    fprintf('MaxIter = 2, Vectorized = %d: no error\n', vec);
  catch err
    fprintf('MaxIter = 2, Vectorized = %d: %s\n', vec, err.identifier);
  end
end

% A per-trial Nreps vector (F-7). The first three cases draw from a stream of
% alternating misses and hits, in which both trials complete together; in
% the fourth, trial 1 completes in the first round and trial 2 is then
% sampled alone.
global STREAM POS
streams = {repmat([0 1], 1, 1000), repmat([0 1], 1, 1000), ...
           repmat([0 1], 1, 1000), [1 0 1 0 0 0 0 0, ones(1, 100)]};
cases = {struct('Vectorized', true, 'Nreps', [2; 3], 'NsamplesPerCall', 4, 'Acceleration', 1), ...
         struct('Vectorized', true, 'Nreps', [2; 3]), ...
         struct('Vectorized', false, 'Nreps', [2; 3]), ...
         struct('Vectorized', true, 'Nreps', [2; 3], 'NsamplesPerCall', 4, 'Acceleration', 1)};
names = {'vectorized, NsamplesPerCall = 4', 'vectorized, NsamplesPerCall = 0', 'loop', ...
         'vectorized, NsamplesPerCall = 4, trial 1 done first'};
for k = 1:numel(cases)
  STREAM = streams{k};
  POS = 0;
  try
    [nl, v, ef, out] = ibslike(@scripted_fun, 0, true(2, 1), [], cases{k});
    fprintf('Nreps = [2; 3], %s: nlogL_trials %s, nlogLvar_trials %s, exit flag %d\n', ...
            names{k}, mat2str(out.nlogL_trials', 5), mat2str(out.nlogLvar_trials', 4), ef);
  catch err
    fprintf('Nreps = [2; 3], %s: error at line %d, "%s"\n', names{k}, ...
            err.stack(1).line, err.message);
  end
end

% A time limit passed before the first round (F-6).
rand('state', 3);
for vec = [true, false]
  o = struct('Vectorized', vec, 'Nreps', 3, 'MaxTime', 1e-9);
  [nl, v, ef, out] = ibslike(fun, 0.5, true(3, 1), [], o);
  fprintf(['MaxTime = 1e-9, Vectorized = %d: nlogL %g, nlogLvar %g, exit flag %d, ' ...
           'funcCount %d, nlogL_trials %s\n'], vec, nl, v, ef, out.funcCount, ...
          mat2str(out.nlogL_trials'));
end

% The self-tests.
rand('state', 4);
fprintf('ibslike(''test''): %d\n', ibslike('test'));
