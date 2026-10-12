# The film's recorded run

One PyIBS estimate of the log-likelihood of the four-in-a-row model on
real positions, recorded on 2026-10-10 and kept as data. Scenes 2 to 5 of
the film draw from it. The trace is not regenerated: the model runs in a
build of its own on one machine, and a rebuild can fetch a different Eigen.

## What was run

- **Trials.** 100 positions drawn without replacement (seed 1) from the
  5,482 positions of human-versus-human play in `data_hvh.txt` of
  [basvanopheusden/fourinarow](https://github.com/basvanopheusden/fourinarow)
  (SHA-256 `f3e6d78e…c53251`). The response of each trial is the move the
  person played. These are the positions that [1] Section 5.4 sampled
  from, there credited to van Opheusden et al. (2016); the Methods of van
  Opheusden et al. (2023, Nature) describe the same human-versus-human
  experiment and count the same 5,482 positions.
- **Model.** The heuristic search of
  [WeiJiMaLab/ninarow](https://github.com/WeiJiMaLab/ninarow) at
  `75fca680`, `model_fitting/tree_search.py`, built from `cpp/` with
  SWIG 4.3.0 and Boost headers from conda-forge (Python 3.12.15,
  NumPy 2.5.3), with Eigen fetched by its CMake at `af6cb145`. It is the
  model of the Nature study, in the variant with an opponent-scaling
  constant; [1] Section 5.4 fitted a reduced version of it (fixed weights
  and tree size, free value noise, pruning threshold and drop rate).
- **Parameters.** `params_hvh_median.json`: the medians of the 200 fits
  of that variant to the human-versus-human experiment (40 participants,
  five cross-validation groups each), from the study's
  [data on OSF](https://osf.io/n2xjm/), `model_fits_opponent_scaling.csv`.
  The stopping probability is 0.005, so a search runs about 200
  iterations; the lapse rate is 0.02, so every legal move has a nonzero
  probability and every trial matches.
- **PyIBS.** At `5979f05` of this repository (`dev-next`), run from the
  source tree; its version string, `0.1.dev149+gd7f6b7227.d20261009`, is
  stale. One repeat (`num_reps=1`, where the default is 10),
  `vectorized=False`, so that each call of the simulator is one round,
  one simulation for every trial not yet matched; no likelihood threshold
  or time limit; `random_seed=1`; `additional_output="full"`. Each call
  seeds the model's generator from PyIBS's.

## Files

- `run_ibs.py` runs the estimate and writes the trace,
  `trial_h_s1_s1.json`: the trials, every simulator call (trials asked,
  moves returned), each trial's count, the rounds and the estimate. It
  recomputes the counts and the estimate from the log and checks them
  against PyIBS's result.
- `candidates.py` lists the trials that could carry scenes 2 to 4, with
  the probability of the person's move from 2,000 fresh simulations each.
- `film_data.py` writes `run_data.js`, which `film.html` loads: the
  counts, the rounds, the estimate, and the two positions of scenes 2 to
  4 with their recorded rows and some fresh simulations (seed 11).

Each script printed its results to a log under `logs/`, which git does
not track. What the film uses is in `trial_h_s1_s1.json` and
`run_data.js`.

They ran in WSL (Ubuntu), from this folder:

```console
PYTHONPATH=<ninarow>/model_fitting:<pyibs> OMP_NUM_THREADS=1 python -u run_ibs.py data_hvh.txt trial_h_s1_s1.json --params params_hvh_median.json --select-seed 1 --seed 1
PYTHONPATH=<ninarow>/model_fitting python -u candidates.py trial_h_s1_s1.json --m 2000
PYTHONPATH=<ninarow>/model_fitting python -u film_data.py trial_h_s1_s1.json run_data.js --likely 49 --surprising 86 --column 0,1,2
```

## The facts the film relies on

| Fact | Value | Lines |
|---|---|---|
| The estimate and its SD | log-likelihood −206.31, SD 10.28; PyIBS's own result agrees with the one recomputed from the log | 5.4 |
| Simulations and rounds | 1,750 simulations in 293 rounds, 3.8 s | 5.3, 5.4 |
| Trials still open after 40 rounds | 12 | 5.3 |
| Counts | median 5; 27 trials match at the first simulation; 20 need more than 20; the two longest need 253 and 293 | 5.3, 5.4 |
| The likely position | trial 49 (line 2704 of the data file), White to move; count 3; probability of the person's move 0.586 from 2,000 simulations, the model's most frequent move | 2.2 to 4.2 |
| Its fresh simulations | 13 of 20 match (line 3.1); 18 of 40 match (line 2.4) | 2.4, 3.1 |
| The surprising position | trial 86 (line 3689), White to move; count 32; probability 0.030 and 0.036 in two runs of 2,000 simulations | 3.2 to 5.1 |
| Its first twenty simulations | all miss; they are the first twenty of its recorded row | 3.2, 3.3 |

The run was the first made with these parameters and seeds; no seed was
chosen for the story. The scenes needed a trial with a count of a few and
one with a count in the thirties among the hundred, and both were there.
An earlier run with the build's default parameters, which mirror the
regime of a study of monkeys (`model_fitting/config.yaml`), was set
aside for these.

[1] van Opheusden, Acerbi & Ma (2020). Unbiased and efficient
log-likelihood estimation with inverse binomial sampling. *PLOS
Computational Biology* 16(12): e1008483.
