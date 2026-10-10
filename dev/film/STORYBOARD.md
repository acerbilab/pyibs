# The PyIBS film: storyboard

Draft 8, 2026-10-10: the storyboard of the masters, with the recorded
run, the narration as revised after the independent review and the
director's notes, the motion of every shot, timed on the words of the
voice, and the score. The voice is Kokoro's `am_michael`, which the
director keeps for the final.

The film line by line: the narration, what each shot shows, and where
each claim comes from. The sources are the IBS paper [1] by its section
numbers, the package's code, and measurements that the production makes
for the film and records under `dev/film/` when it makes them.

[1] van Opheusden, Acerbi & Ma (2020). Unbiased and efficient
log-likelihood estimation with inverse binomial sampling. *PLOS
Computational Biology* 16(12): e1008483.
https://doi.org/10.1371/journal.pcbi.1008483

## Audience and angle

**Audience.** Scientists and engineers who fit models to data. They know
what a model and its parameters are, and have met a likelihood. Many of
them have a model they can simulate but cannot write a likelihood for,
and have estimated it by simulation, counting matches, without knowing
what that does to the estimate.

**Angle.** The film follows the paper's own order. It says what the
log-likelihood is and what it is used for, why for many models it has
to be estimated by simulation, why the obvious estimate is biased in a
way that more simulations do not fix, and how IBS removes the bias by
simulating each trial until the first match and counting the
simulations. It then hands the estimate to PyBADS and PyVBMC, and ends
on why the bias matters: a biased estimate can lead a study to a false
conclusion held with confidence, while the noise of an IBS estimate is
measured by its error bar.

**The running example.** IBS needs only a model that can be simulated
and discrete observations. The film, like the paper, writes the data as
trials of an experiment, each a stimulus and the response a person
gave, and line 1.1 says that this is an example.

**What the film is not about.** The board-game model is a prop. It
appears where the chain needs a model that can be run but not written
down, and as the trial on screen while IBS is explained. The film says
what the model is used for, from the Nature paper, and makes no other
claim about the game or the model beyond what [1] states.

**Length.** 3,665 characters of narration in this draft. On the draft
voice, Kokoro's `am_michael` at speed 1.12, the film runs 4:07 with its
end card (the masters, `media/masters`). The soft target is 2:20, and the two previous
films ran 2:48 and 3:00. The film stays at full length for now; the cut
candidates are listed at the end.

## Format

- The narration is plain explanation: complete sentences in normal word
  order, each saying one thing. It has no fragments, rhetorical
  questions, colon reveals, inverted word order or slogans. It puts no
  comma before "and" or "or" inside a sentence unless the sentence is
  unclear without it, and two full clauses joined by "and" are two
  sentences. A line lands a point only when the film has set it up, as
  lines 4.4 and 6.4 answer the problem of scene 3.
- The tool's name is on screen from the first frame: "PYIBS Inverse
  Binomial Sampling", small, top left, until the end card.
- A flat 2D page, not a 3D landscape. The marks: a trial is a row; a
  simulation is a small cell, a grey cross for a miss and a green check
  for a match, and on the board a ghost piece where the simulated move
  falls; a count stands above its column; the log-likelihood is a
  number, and where a line speaks of changing the parameters, a grey
  curve over one parameter. Fixed-sampling estimates are amber, IBS
  estimates and error bars green, true values grey, the posterior
  purple. One meaning per mark for the whole film, and one meaning per
  word: "noise" is only the noise in an estimate, and the model's own
  randomness is called randomness.
- Each shot builds its picture a mark at a time, each mark on the words
  that introduce it, placed by the take's word times. A shot's last mark
  is in place at least 1.5 s before its picture leaves the screen, so
  that the viewer can take it in before the cut. A mark whose words come
  late in the line goes on an earlier phrase of the same sentence, and a
  picture can hold while the next line begins (line 3.3, by 1.0 s).
  Lines 3.2 and 5.3 hold less, since the next shot continues their
  picture unchanged.
- Technical names appear as small italic labels next to what they name,
  for at least 2.5 s: *likelihood*, *maximum likelihood*, *heuristic
  search*, *simulator*, *fixed sampling*, *geometric distribution*,
  *digamma*, *trigamma*, *Bayesian posterior*.
- The captions are the narration word for word. A caption that ends in
  an asterisk calls a footnote at the bottom right.
- Numbers are spoken in words and shown in digits, and appear on screen
  only after the narration has introduced them.
- The spoken text respells the names for the voice ("Pie-I-B-S",
  "Pie-Bads", "Pie-V-B-M-C"); the captions write them normally.

## The score

The score is synthesized from the film's own timeline, in A minor, in
two halves that take the ways of the lab's two earlier films. While the
film explains the problem (scenes 1 to 3), it is scored as the PyVBMC
film is, with pads of detuned saws whose harmony changes with the
lines, a glassy ping for each mark and clicks for the simulations that
miss. From inverse binomial sampling to the end card it is the PyBADS
film's groove, with drums, a pulsing bass, a pad, an arpeggio and a
sparse lead, from that film's synthesizer.

- The first half follows the lines. On "there is no formula" the pads
  darken to one low chord. The chords of line 2.4 change faster as its
  simulations speed up. At "minus infinity" a swell rises into a low
  boom, and the music stops.
- The bias is a rising inner voice. From line 3.3 the fifth of the chord
  rises a semitone at a time (Am, F/A, Am6, Am7, then the same on D).
  Scene 3 ends on an augmented chord of E, which holds into line 4.1.
  It resolves into A minor as the groove comes in on "turns the obvious
  approach around".
- The second half follows the argument. The terms of line 4.3 play over
  a drone on A as the harmonic series of A2. Term k is the k-th
  harmonic at an amplitude of 1/k, so that their sum is heard as a
  sawtooth wave. "This estimate is unbiased" arrives on F major 7, where
  the lead enters. Line 5.4 holds a dark chord of E under the voice. A
  build leads to the name of PyIBS in line 6.1. The bias returns in
  line 6.3 with the flat second, B flat over A. The false conclusion
  falls into the drone. "With IBS" lifts into the end card, which ends
  on A major.
- Every mark has a sound at the moment the picture shows it. In the
  first half a simulation that misses is a click and one that matches a
  ping. In the second half they are a tap and a bell. A cloud's true value sounds
  E5. Its mean under fixed sampling sounds F5, a semitone too high. Its
  mean under IBS sounds E5. Line 1.3's knob is a soft tone whose
  pitch follows it. In the second half, accents move to the nearest
  thirty-second note of the beat, at most 34 ms from the picture.
- The second half runs on a tempo map. Each section starts on a
  downbeat where its picture starts, or at "turns around", at "PyIBS
  works", at "a false conclusion" and at "With IBS". It holds a fixed
  number of bars, which on the draft voice gives tempi from 92 to 103
  BPM.
- Under the voice the music dips by 45 % and the events by 25 %. Over the
  film the score sits 17 LU below the voice, the PyBADS film's setting.
  The first half's music sits 1.5 LU below the second half's. Line by
  line the voice stands 14 to 29 dB above the score. The film is
  mastered to -16 LUFS and -1.5 dBTP.

## Scenes

### 1. The log-likelihood

Generic trials, no game yet: a column of rows, each with a stimulus tile
and a response token.

- **1.1** "You have a model and some data. The model assigns a
  probability to each observation. For example, in a behavioral
  experiment, each observation is the response a person gave on one
  trial." On "some data", response tokens come in down a column, eight
  observations, under the label DATA. On "The model assigns a
  probability", a bar grows beside each, one after another, to its
  probability under the model, and the first is labelled *likelihood*.
  On "in a behavioral experiment", a stimulus tile comes in at the head
  of each row, and on "each observation is the response" the label DATA
  becomes RESPONSE. On "a person gave", the first row is outlined and
  labelled one trial. *Source:* [1] 2.1, the likelihood as the
  probability of the observed data given the parameters, written for a
  data set of trials, each a stimulus and a response.
- **1.2** "The log-likelihood is the sum of the logs of these
  probabilities. It measures how well the model explains the data." The
  outline of the trial goes and the column dims. On "the sum", the log
  of each probability appears at the end of its row, one after another.
  On "of these probabilities", a rule is drawn under the column and the
  log-likelihood counts up to its value, labelled *log-likelihood*.
  *Source:* [1] 2.1, Equations 1 and 2: the likelihood as a product over
  trials and the log-likelihood as its log, a sum.
- **1.3** "The log-likelihood changes with the model's parameters. A
  common way to fit a model is to choose the parameters where the
  log-likelihood is highest." The logs go. An empty panel appears beside
  the column, with a knob for one parameter under it, set where the
  number at the foot was computed. On "changes with the model's
  parameters", the knob sweeps to one end and across to the other; the
  bars and the number move with it, and a grey curve traces the number
  against the parameter. On "choose the parameters", the knob turns back
  to the top of the curve, where a ring lands, labelled *maximum
  likelihood*. *Source:* [1] 1 and 2.1, maximum-likelihood estimation as
  finding the parameters that maximize the log-likelihood. [1] 1 calls
  it a principled method and names other common ones, which also rest on
  the likelihood.
- **1.4** "The log-likelihood is also used to compare models and to
  compute the Bayesian posterior over the parameters." The curve holds
  with its ring, and the knob goes. On "compare models", a dashed curve
  draws in, another model's log-likelihood, keyed "model comparison" at
  the top of the panel. On "the Bayesian posterior", a purple posterior
  grows over the parameter axis, labelled *Bayesian posterior*. Scene 6
  uses the ring and the posterior again. *Source:* [1] 1, the
  log-likelihood as the basis of model comparison and of Bayesian
  inference of posterior distributions; 6.2 for Bayesian inference.
### 2. Models you can only simulate

- **2.1** "For many models, however, there is no formula for these
  probabilities." The column of scene 1 returns with its bars full. On
  "there is no formula", the bars empty one after another to hollow
  outlines, each with a question mark, and the number at the foot
  becomes a question mark. *Source:* [1] 1, simulator-based models whose
  likelihood is intractable.
- **2.2** "Take a model of how people play four-in-a-row, which is used
  to study how people plan ahead and how this changes as they learn.
  Each trial is a position from a real game. The response is the move
  the person played in it." The hollow column fades, and a large empty
  board takes its place. On "four-in-a-row", its pieces come in: the
  4-by-9 board, a real position from a human game. On "Each trial is a
  position", a column of small boards comes in at the left, one per
  trial, the first of them the large one, and the position is named the
  stimulus. On "The response is the move", a ring marks the move the
  person played on every board, and the move is named the response. The
  provenance, small, bottom left: "Positions from human-versus-human
  games of van Opheusden et al. (2023, Nature); model code: Bas van
  Opheusden and the Wei Ji Ma lab." *Source:* the Nature paper (van
  Opheusden et al. 2023, "Expertise increases planning depth in human
  gameplay"), which uses the model to study planning, with a learning
  experiment over five sessions, and whose Methods say that the model
  was fitted by estimating the log-likelihood with inverse binomial
  sampling. The line says what the model is used for, not what the study
  found. [1] 5.4: the board position is the stimulus of a trial.
- **2.3** "Given a position, the model searches a few moves ahead. Its
  evaluation of the moves is partly random. The probability of each move
  depends on every search tree the model could build, so it cannot be
  computed." On "searches a few moves ahead", a small tree of candidate
  moves unfolds beside the board, level by level, labelled *heuristic
  search*. On "Its evaluation of the moves", a value appears beside each
  candidate move, the best one brightest; on "partly random" the values
  flicker and the best move changes with them, until the next sentence.
  On "every search tree", trees multiply behind the board, faint, more
  than can be counted, and the probability of a move becomes a question
  mark. *Source:* [1] 5.4: a value function over board features with
  additive noise, pruning and feature dropping, and the distribution
  over the model's moves "would require integrating over all possible
  trees it could build, features which may be dropped, and realizations
  of the value noise". The film calls the model's noise randomness,
  since "noise" is reserved for the noise in an estimate.
- **2.4** "But you can give the model the same position as many times as
  you like. Each time it simulates a move. Over many simulations, the
  fraction that match the person's move approaches its probability." The
  board stays as it is, labelled *simulator*. On "Each time it simulates
  a move", a ghost piece appears where the model's move falls and drops
  into a row under the board as a cross for a miss or a check for a
  match; a small dot stays where each simulated move fell, green on the
  person's square. The first three go slowly, a miss, a miss and a
  match, and on "Over many simulations" the rest follow faster and
  faster. The position never changes within a trial, here or in scenes 3
  and 4. The forty simulated moves are the model's own, on the likely
  position, from the run's data; eighteen of them match. *Source:* [1]
  2.2, conditional simulation: the simulator returns a response for a
  given stimulus; 2.3, the reduction to Bernoulli sampling: the
  probability of the observed response equals the probability that a
  simulated response matches it.
### 3. The obvious estimate

The positions of scene 2 are the trials on screen. A simulation is a
cell: a grey cross when the model's move misses the ringed square, a
green check when it lands on it.

- **3.1** "The obvious approach is to simulate each trial a fixed number
  of times and take the log of the fraction that match." On "to simulate
  each trial", a row of twenty cells fills one cell at a time for the
  likely position: twenty simulations of the model from the run's data,
  thirteen of which match. As the row completes, the count, "13 of 20",
  comes in, in amber, and its log on "take the log". Label *fixed
  sampling*. *Source:* [1] 2.4, the fixed-sampling policy and the
  fraction of matches as its simplest estimator.
- **3.2** "If the person made an unlikely move, none of the simulations
  may match. Then the estimate is the log of zero, which is minus
  infinity." The surprising position, its ring flaring on "an unlikely
  move". On "none of the simulations", twenty crosses fill the row one
  at a time: the first twenty simulations of its row in the recorded
  run. The amber readout reads "0 of 20" as the row completes, then "log
  0" on "the log of zero", and "= −∞" when the voice says it. Line 3.3
  continues the picture, with the readout, until "adds one". *Source:*
  [1] 2.4, Fixed sampling: as long as the probability is below one,
  there is a nonzero chance of no match, and the log of the fraction is
  then minus infinity.
- **3.3** "A simple fix adds one to both counts, but with twenty
  simulations a probability of one in a hundred gets almost the same
  estimate as one in a million." The twenty crosses stay. On "adds one",
  (0 + 1) / (20 + 1) = 1/21 and its log, −3.04, replace the readout of
  line 3.2, in amber. On "one in a hundred", a table begins with that
  probability, its true log in grey (−4.6) and the average
  fixed-sampling estimate with twenty simulations in amber (−2.9); on
  "one in a million", its second row (−13.8 and −3.0). The table shows
  only the probabilities the voice names. The picture holds for 1.0 s
  into line 3.4, so that its last row has time to be read. *Source:* [1]
  2.4, Equation 11: "This divergence can be fixed in multiple ways; in
  the main text we use" log((m + 1)/(M + 1)); [1] 3, the gambling
  analogy, which makes the same point: after a run of misses, one
  percent and a hundredth of a percent cannot be told apart. *Picture:*
  the averages are computed exactly from the binomial distribution.
- **3.4** "With a fixed number of simulations, no fix makes the estimate
  unbiased. This one is too high on average, most of all for the
  responses the model explains worst." The picture comes on 1.0 s into
  the line, after line 3.3's holds. Three columns, one per
  probability, the probability shrinking from left to right, each with
  the grey line of the true log probability. During the first sentence,
  repeated fixed-count estimates of one trial drop in as amber dots. On
  "too high on average", the mean of each column settles above its grey
  line. On "most of all", a bracket marks the gap in each column in
  turn, the widest at the smallest probability, labelled *bias*.
  *Source:* [1] 2.4 and 3, Figures 1B and 2: the bias of the paper's
  fixed-sampling estimator is positive (the log-likelihood is
  overestimated), a function of the probability times the number of
  samples, and it diverges as the probability goes to zero. The first
  sentence rests on [1] 2.4: any estimator under the fixed-sampling
  policy is biased, for any regularization (Appendix A.2). *Picture:* an
  illustration computed from the paper's formulas, twenty samples per
  trial, at probabilities 0.2, 0.05 and 0.01.
- **3.5** "This bias does not average out over the trials. It adds up.
  It is larger at some parameter values than at others, so it moves the
  best fit." The column of trials returns beside the grey curve of scene
  1. On "does not average out", each row gets a small amber push, one
  after another, the largest on the unlikely response. On "adds up", the
  pushes add into one long arrow at the foot of the column. On "It is
  larger", an amber curve rises off the grey one, and on "at some
  parameter values" the gap between the curves is marked at three
  values, widest on the left. On "than at others", dashed lines drop
  from the two tops, and an arrow on the axis runs from the true best
  fit to the best fit under fixed sampling, so that the move is on
  screen while the voice says it. *Source:* [1] 4.5, point 1: the bias
  of the sum grows with the number of trials where the noise grows with
  its square root; 5.6: fixed sampling with too few samples biases the
  parameter estimates. The bias moves the top because it differs between
  values of the parameter; a constant offset would leave the top in
  place. *Picture:* the toy model of scene 1 (300 trials, a lapse rate
  of 0.03): its true log-likelihood and the expected fixed-sampling
  estimate with twenty samples per trial, in the window around their
  tops. The top moves from 0.95 to 0.82 of the parameter, the direction
  that [1] finds for the noise parameters of its models (5.3, 5.4).
- **3.6** "More simulations reduce the bias, but how many are enough
  depends on the probabilities you want to estimate. With fixed
  sampling, a hundred simulations per trial still left one parameter of
  the four-in-a-row model biased." At the rare response, the cloud for
  twenty simulations, with its bracket. On "reduce the bias", the clouds
  for fifty and then a hundred simulations drop in beside it, each with
  its mean and bracket: the gap narrows only slowly. On "With fixed
  sampling" the board comes back small on the right, "100 simulations
  per trial" comes in on "a hundred simulations per trial", and "fitted
  parameter still biased" on "still left one parameter". The caption's
  asterisk calls the footnote "\*van Opheusden, Acerbi & Ma (2020),
  Figure 9A.", which comes in with the board. *Source:* [1] 3.1: enough
  samples to be unbiased means many more than one over the smallest
  probability, and choosing that number requires the probabilities,
  which is the problem being solved. [1] 5.4 and Figure 9A: with M =
  100, fixed sampling still underestimates the model's value noise,
  which IBS estimates accurately with a quarter of the samples, on data
  simulated with known parameters. The film does not name the parameter,
  so that "noise" keeps its one meaning.
### 4. Inverse binomial sampling

- **4.1** "Inverse binomial sampling turns the obvious approach around.
  It simulates each trial until the first match and counts how many
  simulations it took." The likely position, and beside it, faint, the
  row of twenty from line 3.1 under the label *fixed sampling*. On
  "turns the obvious approach around", that row empties from the right
  and the label becomes *inverse binomial sampling*. On "It simulates
  each trial", the model's moves on the position come one at a time, the
  recorded run's own: a miss, a miss, then the person's move. Each shows
  as a ghost piece on the board and drops into the row as a cross or a
  check, with the count above it, in green. The row stops at the match:
  three simulations to the first match. *Source:* [1] 2.4, the IBS
  policy. Binomial sampling fixes the number of samples and counts the
  matches; inverse binomial sampling fixes the number of matches, here
  one, and counts the samples, which is where its name comes from.
- **4.2** "Likely responses need few simulations. Unlikely ones need
  many more. The average count is one over the probability." Two trials,
  one above the other. The likely move's cells come in, a check after
  three. On "Unlikely ones", the surprising move's long row of crosses
  comes in, ending in a check at the thirty-second. The counts are the
  recorded run's. On "The average count", one over each move's
  probability appears beside its count, about 1.7 and about 33, with the
  label *geometric distribution*. *Source:* [1] 2.4: the count is
  geometrically distributed with the trial's probability; 4.2: its
  expected value is one over the probability, so IBS spends its samples
  on the trials that need them, and a very unlikely response is costly
  (Appendix C.1; the run's longest count is 293).
- **4.3** "The estimate starts at zero. Each miss before the match
  subtracts one more term, first one, then a half, then a third." The
  surprising move's row of thirty-two. On "starts at zero", a zero under
  it and the running estimate, in green. On "Each miss before the
  match", one term appears under each cross, left to right: −1, −1/2,
  −1/3, ..., −1/31, and the running estimate counts down with them to
  the sum, −4.03. Then ψ(1) − ψ(K), labelled *digamma* just below it.
  All of it stands
  before the voice names the first three terms. *Source:* [1] 2.4,
  Equation 14, minus the sum of 1/k for k from 1 to K − 1, which is zero
  for K = 1, and its compact form ψ(1) − ψ(K).
- **4.4** "This estimate is unbiased. On average it equals the true log
  probability, however small the probability is." The caption ends in an
  asterisk, and the footnote reads "\*de Groot (1959); van Opheusden,
  Acerbi & Ma (2020), Appendix A.1. Under this sampling, no other
  estimator that is unbiased for every probability has a smaller
  variance." The amber clouds of line 3.4 return with their grey lines
  and means. Green dots of repeated IBS estimates fall in beside them,
  column by column. On "On average", the green mean of each column lands
  on its grey line at every probability, the amber mean above it, and on
  "however small" the column of the smallest probability lights. Both
  means are the exact expectations. *Source:* [1] 2.4: the estimator is
  uniformly unbiased (de Groot 1959), and [1] shows it is the uniformly
  minimum-variance unbiased estimator under the IBS policy (Appendix
  A.1).
- **4.5** "Summing over the trials gives an unbiased estimate of the
  log-likelihood." The column of trials, each row's cells coming in down
  the column, each row stopped at its first match with its count. On "an
  unbiased estimate", a rule under the column and the green sum,
  counting to its value: one IBS estimate of the toy model's
  log-likelihood, beside the true value in grey. *Source:* [1] 2.4 and
  4.1: the estimate of the data set's log-likelihood is the sum of the
  per-trial estimates, and a sum of unbiased estimates is unbiased.

### 5. What comes with it

- **5.1** "The count also gives an error bar for the estimate. Summed
  over the trials, the error bars are calibrated. The true
  log-likelihood falls inside them as often as it should." At the top,
  the surprising position's stopped row and its estimate; on "an error
  bar", its error bar, with ψ₁(1) − ψ₁(K) labelled *trigamma* just
  below it. On "Summed over the trials",
  twenty summed estimates of a whole data set drop in one under another,
  each with its error bar of one SD. On "The true log-likelihood", a
  grey line marks the true value, the bars that miss it dim, and a count
  says how many bars contain it, against about two in three for bars of
  one SD. *Source:* [1] 4.3, Equation 16: the variance estimate from the
  count, which for one trial is a Bayesian posterior variance; 4.6 and
  Figure 3C: the calibration of the summed estimate. *Picture:* the toy
  model of scene 1, one IBS repeat per estimate, computed in the page;
  16 of the 20 bars contain the true value.
- **5.2** "To reduce the noise, you can repeat the estimate and
  average." The column of trials with its counts, and at the right one
  estimate with its error bar. On "you can repeat", the whole column
  runs again with new counts, and the error bar of the average narrows;
  then once more. The label beside the bar reads one repeat, then
  average of 2 repeats, then average of 3, and a faint outline keeps the
  width of the first bar. *Source:* [1] 4.4: averaging repeated
  estimates reduces the variance.
- **5.3** "PyIBS simulates all the trials together, one round at a time,
  until each has matched." The recorded run: a hundred rows, one per
  real position, one square per simulation. Round by round, a column of
  squares fills for every trial not yet matched; a match is green and
  its row dims. A counter gives the round and the trials still open: by
  the end of the line, forty rounds and twelve open trials. *Source:*
  the package's sampler, which with `vectorized=False` draws one
  simulation for every open trial per round (`pyibs/_sampler.py`); [1]
  Appendix C.1, the parallel implementation. With the default settings
  a round can draw several simulations per trial.
- **5.4** "Then it returns the estimate and its error bar." The last
  trials run on alone, the longest to round 293. As the last match
  lands, the readout: log-likelihood −206.3 ± 10.3, and 1,750
  simulations in 293 rounds. *Source:* the recorded run
  (`run/README.md`). PyIBS returns the negative log-likelihood unless
  asked otherwise; the readout shows its negation, and the error bar is
  its estimated SD.
### 6. The hand-off, and why bias matters

- **6.1** "The four-in-a-row model is only one example. PyIBS works with
  any model that can simulate each observation on its own, as long as
  the observations are discrete." The board of the four-in-a-row model,
  labelled four-in-a-row model. On "PyIBS works", the wordmark. On "any
  model", the board's label becomes a simulator. On "simulate each
  observation", ghost pieces fall where the model's simulated moves
  land, one after another. As the voice reaches "as long
  as the observations are discrete", three response tokens come in one
  by one, labelled discrete observations. *Source:* [1] 2.1 and 2.2: the
  responses are discrete, and the simulator takes the stimulus and the
  parameters. The film says nothing about continuous responses, which
  [1] 6.3 handles by binning or by an approximate variant with a
  tolerance, or about the cost in samples, which is line 4.2's. [1] 2.2
  also requires conditional simulation, which "on its own" states, and
  responses that are not too improbable for the budget, which a lapse
  rate ensures ([1] 6.4).
- **6.2** "You can then use PyBADS to find the maximum-likelihood fit or
  PyVBMC to approximate the Bayesian posterior." Two panels with the
  curve of scene 1, faint. On "PyBADS", green IBS estimates with their
  error bars appear one by one where the optimizer tries the parameter
  and gather toward the top, where a ring lands, labelled *maximum
  likelihood*, as the voice says "fit". On "PyVBMC", the estimates
  spread over the range, and a purple posterior grows over the parameter
  axis, labelled *Bayesian posterior*; it is drawn by the time the voice
  names it. *Source:* [1] 6.4: perform inference with a sample-efficient
  algorithm based on Gaussian-process surrogates, with BADS named for
  maximum-likelihood estimation; 6.2: Bayesian inference, with VBMC.
  PyVBMC approximates the posterior by variational inference ([1] 6.2,
  Appendix C.3). The package's documentation on its use with PyBADS and
  PyVBMC. *Picture:* the toy model; each estimate averages ten IBS
  repeats; the parameter values tried are chosen for the picture, and
  the ring sits at the top of a quadratic fitted to the estimates.
- **6.3** "Both are designed to work with noisy estimates like those of
  IBS. No fitting method can correct a biased estimate like that of
  fixed sampling." One panel with the grey curve of line 3.5. During the
  first sentence, green IBS estimates appear one by one with their error
  bars, scattered around the grey curve, and on "noisy" the label
  *noise*. On "No fitting method", amber fixed-sampling estimates appear
  one by one, all above the grey curve, and the amber curve draws
  through them; on "biased", a bracket between the two curves is
  labelled *bias*. *Source:* [1] 4.5, point 3: surrogate-based methods
  "can operate successfully with noisy objectives", "no optimization
  algorithm can handle bias", and the same holds for the surrogate
  methods of Bayesian inference; point 5: bias "is much harder to
  recognize or correct no matter what statistical techniques one uses".
  *Picture:* the toy model; each estimate averages ten repeats of its
  estimator.
- **6.4** "With a biased estimate, a study can reach a false conclusion
  and be confident in it. With IBS, the estimate is only noisy. Its
  error bar shows how far to trust it." The same panel. The estimates
  fade and the bracket goes. On "a false conclusion", the amber top
  flares and its dashed line drops to the axis, away from the true top,
  labelled fit with fixed sampling. On "With IBS", the true top's dashed
  line drops and the fit read from the IBS estimates lands near it,
  labelled fit with IBS. On "the estimate is only noisy", the IBS
  estimate at that fit appears, and on "Its error bar" its error bar
  opens. *Source:* [1] 4.5, point 5: "Bias can cause researchers to
  confidently draw false conclusions", while variance, "when properly
  accounted for, causes decreased statistical power and lack of
  confidence"; 1 and 3.1: the bias of fixed sampling can reverse the
  outcome of a model comparison; 2.4: IBS is unbiased; 4.3: its variance
  estimate is calibrated.

Then the end card, over black for 8 s, as the PyBADS and PyVBMC films'
cards: PyIBS 1.5 and its name, Inverse Binomial Sampling; `pip install
--upgrade pyibs`; acerbilab.org/model-fitting; the IBS paper [1]; and
the credits, "Machine and Human Intelligence Group · University of
Helsinki", "Research Council of Finland · ELLIS Institute Finland",
"Directed by Luigi Acerbi · Made with Claude Code · Voice: Kokoro".

## The run

One real PyIBS estimate, recorded on 2026-10-10 and kept as data in
`run/`, whose README gives the method, the revisions and the facts the
lines rely on. In short: a hundred positions drawn from the 5,482 of
human-versus-human play that [1] 5.4 used, the move the person played as
each response; the four-in-a-row model of the Nature study, as the Wei
Ji Ma lab's build of it, at the medians of that study's fits to its
human-versus-human experiment; one repeat with `vectorized=False`, no
threshold or time limit, so that every trial ends in a match and each
round draws one simulation per open trial. The estimate is −206.3 with
an SD of 10.3, from 1,750 simulations in 293 rounds.

Scenes 2 to 4 follow two of its trials. The likely position (trial 49)
matched at the third simulation; its person's move has a probability of
about 0.59 and is the model's most frequent move. The surprising
position (trial 86) matched at the thirty-second; its probability is
about 0.03, so its first twenty simulations, all misses, are the twenty
crosses of line 3.2. The simulations shown in lines 2.4 and 3.1 are
fresh ones of the likely position, from the same build and parameters.

## Illustrations

- 3.3: the average fixed-sampling estimate with twenty simulations at
  probabilities of one in a hundred and one in a million, computed
  exactly from the binomial distribution.
- 3.4, 3.6 and 4.4: repeated estimates of one trial at a known
  probability, computed in the page from the estimators of [1] 2.4
  (fixed sampling with 20, 50 or 100 samples; IBS), with a seeded
  generator, at probabilities 0.2, 0.05 and 0.01.
- 1.3, 1.4, 3.5 and 6.2 to 6.4: a one-parameter toy model kept off
  screen (300 trials, a lapse rate of 0.03). The curves are its true
  log-likelihood and the expected fixed-sampling estimate with twenty
  samples per trial, computed exactly from the binomial sum, shown in
  the window around their tops. The estimates of scene 6 each average
  ten repeats of their estimator: one IBS repeat has a standard
  deviation of about 8.6 here, against a bias of fixed sampling of about
  7.6 at the top, and the picture needs the two to be told apart. The
  dashed curve of line 1.4 is the same model with a lapse rate of 0.12,
  as another model to compare.
- 1.2 to 1.4 and 4.5: the column shows eight of the toy model's 300
  trials, and its number is the toy's log-likelihood, or in line 4.5 one
  IBS estimate of it.
- 2.2 to 5.4: the positions, the simulated moves, the counts and the
  run of scene 5 are the recorded run's (`run/`). The search tree of
  line 2.3 is drawn for the picture.

## The page and its files

`run/` holds the recorded run and the scripts that made it;
`run/run_data.js`, which `film.html` loads, is its data for the film.
`film.html` draws the film: one entry of `SHOTS` per line, with the
line, the caption when it differs, a note on what moves, a draft length
and a `draw(t)`. `film.html?shot=K` draws shot K as its picture leaves
the screen in the film, and `&t=T` draws it T seconds into its line.
`scripts/stills.mjs` draws one still per shot so in headless Chrome
into `storyboard/frames/s<ID>.jpg` and writes the shot list to
`storyboard/shots.json`; `scripts/gen_storyboard.py` turns that into
`storyboard/storyboard.html`, the review page, which shows the previous
round's lines from `storyboard/lines-round<N>.json` struck through under
the lines that changed. The words live in `film.html` alone; this
document restates them with their sources.

The voice: `narration.json` gives the Kokoro voice and its speed, the
scenes in order with their lines, the pauses (a lead before each scene,
a gap between lines, a tail after each scene, and the end card's length)
and the respellings that the voice reads in place of the written names
("Pie-I-B-S" for PyIBS). The words themselves stay in `film.html`:
`scripts/voice.py` reads them from `storyboard/shots.json`, voices each
line into `media/<version>/voice/<id>.wav`, places the takes into
`narration.wav` and `narration.srt`, and writes `film_timeline.js`, the
timeline that `film.html?film=1&T=` plays. `scripts/words.py`
transcribes each take with faster-whisper and writes `film_words.js`,
the time of each of its words, from which a shot's `cue("phrase")` is
the moment its take reaches that phrase; it runs again whenever a line
is voiced again. A shot is drawn in its line's own seconds. Its picture
is on screen from the start of its line to the start of the next, or
longer by its `hold`, while the captions follow the voice. Scenes change
with a crossfade, and the shots within a scene cut. Without a voice, a
cue falls where the narration reaches its phrase at 13.3 characters a
second, after a lead of 0.4 s.

`scripts/record.mjs --film --audio media/<version>/narration.wav`
records the animatic, and `scripts/record.mjs OUT.json --film --holds`
reports, for each picture, its last change and how long it then holds
before the next replaces it. The same script records one shot as an MP4
through the hook `ibsFilm.render(k, t)`, with the captions changing by
sentence (`film.html?timed=1`); the review page plays a clip in
`storyboard/motion/s<ID>.mp4` in place of that line's still. Recordings
use the ffmpeg of the lab's media environment as `FFMPEG`.

Each shot lists its events, for the score, in `EVENTS`, timed by the
same cues as its picture. `scripts/record.mjs OUT.json --film --events`
writes them on the film's timeline. `scripts/score.py media/<version>`
reads them with the narration and writes the score, the mix with the
voice, the stems and `score_plan.txt`, the plan of every section and
bar. It imports `scripts/synth.py`, the PyBADS film's synthesizer, for
the second half. `scripts/mux.py` puts the mix under a recorded film at
-16 LUFS. The score is made again from a new export of the events
whenever a line is voiced again or a shot's timing changes. `score.py`
and `mux.py` run in the lab's media environment, about two and a half
minutes for the score.

The masters are recorded as the reviewed 1280 x 720 layout at a device
pixel ratio of 1.5, 1920 x 1080 at 25 fps:
`scripts/record.mjs OUT.mp4 --film --scale 1.5 --crf 18`, with
`--clean` for the master without captions, and `scripts/mux.py` puts the
mix under each. `scripts/record.mjs OUT.srt --film --captions` writes
the subtitles, as SubRip and WebVTT, from the captions that the page
shows: a line's first sentence with the line, each later one 0.15 s
before the voice reaches it.

## Claims to check

| Claim | Source | Scope |
|---|---|---|
| The log-likelihood is a sum over trials of the log probability of each response (1.2) | [1] 2.1 | Conditionally independent trials, as [1] assumes. |
| Maximum likelihood is a common way to fit a model (1.3) | [1] 1 | |
| The log-likelihood is used to compare models and to compute the Bayesian posterior (1.4) | [1] 1, 6.2 | |
| For many models there is no formula for the probabilities (2.1) | [1] 1 | "Many", not "most". |
| The four-in-a-row model is used to study how people plan ahead and how this changes as they learn (2.2) | the Nature paper's title and its learning experiment | What the model is for, not what the study found. |
| The game model's move probabilities are intractable (2.3) | [1] 5.4 | Said of this model only. |
| The probability of a response is the probability that a simulated response matches it (2.4) | [1] 2.2, 2.3 | |
| A fixed count with no match gives the log of zero (3.2) | [1] 2.4 | |
| Adding one to both counts removes the infinity and leaves small probabilities with almost the same estimate (3.3) | [1] 2.4, Equation 11; 3, the gambling analogy | The averages are computed exactly, for twenty simulations. |
| Fixed sampling overestimates the log-likelihood, most for rare responses, and the bias diverges (3.4) | [1] 2.4, 3, Figures 1B and 2 | Illustration at stated probabilities and counts. |
| No fix makes a fixed-sampling estimate unbiased (3.4) | [1] 2.4, Appendix A.2 | Any estimator under the fixed-sampling policy. |
| Bias grows with the number of trials, noise with its root (3.5) | [1] 4.5 point 1 | |
| Fixed sampling biases the fitted parameters, because its bias differs between parameter values (3.5) | [1] 5.6; the toy's curves | "Moves the best fit", no number. |
| Choosing the count needs the probabilities (3.6) | [1] 3.1 | |
| With a hundred simulations per trial, fixed sampling left one parameter of the four-in-a-row model biased (3.6) | [1] 5.4, Figure 9A | The value noise, underestimated with M = 100; IBS estimated it accurately with a quarter of the samples. Data simulated with known parameters, on [1]'s reduced model. |
| The IBS count is geometric, and its average is one over the probability (4.2) | [1] 2.4, 4.2, Appendix C.1 | |
| The estimator is Equation 14 and is zero for a count of one (4.3) | [1] 2.4 | |
| The estimator is uniformly unbiased (4.4) and, under the IBS policy, has the smallest variance among such estimators (footnote of 4.4) | [1] 2.4, Appendix A.1, de Groot 1959 | |
| The error bars, summed over the trials, are calibrated (5.1) | [1] 4.3, 4.6, Figure 3C | Shown in [1] for the summed estimate, in one model; per trial, the variance estimate is a posterior variance. |
| Averaging repeats reduces the variance (5.2) | [1] 4.4 | |
| PyIBS samples every open trial per round (5.3) | `pyibs/_sampler.py` | |
| The run's estimate and error bar (5.4) | the recorded run | |
| IBS needs a model that can simulate each observation on its own, and discrete observations (6.1) | [1] 2.1, 2.2, 6.3 | Continuous responses are not mentioned. Responses must not be too improbable for the budget ([1] 2.2), which a lapse rate ensures. |
| PyBADS finds the maximum-likelihood fit and PyVBMC approximates the posterior from IBS estimates (6.2) | [1] 6.2, 6.4, Appendix C.3 | |
| Surrogate-based methods work with noisy estimates; no fitting method can correct bias (6.3) | [1] 4.5 points 3 and 5 | Point 3 says "optimization" and extends it to the surrogate methods of Bayesian inference; point 5 says it of any statistical technique. |
| Bias can lead to false conclusions held with confidence, while noise lowers confidence (6.4) | [1] 4.5 point 5; 1 and 3.1 | The paper's argument in plain words. "Only noisy" holds when sampling runs to completion: a likelihood threshold or a time limit can bias the estimate ([1] Appendix C.1). |
| The Nature study fitted its model with IBS (2.2 provenance) | its Methods, in its own text | The film names the model's home and its use, not the study's findings. |

## Cut candidates

The film stays at full length for now. In the order to cut them if it
has to come down:

1. 5.2, the repeats.
2. Merging 5.3 into 5.4.
3. 5.3 itself, the package's rounds.
4. 6.1, whose words can go on the end card.
5. The spoken terms of 4.3, which can stay on screen.

## Open

- The director's review of the masters.
- The publication, after PyIBS 1.5 is released: the film's page on the
  lab's model-fitting site, YouTube, and the links from the README and
  the documentation.
