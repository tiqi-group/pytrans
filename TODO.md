# Pytrans

Split the remaining work in 4 areas

## Coding - me, Francesco

- [x] Classical simulator
- [x] Decent plotting API, independent from trap axes orientation
- [ ] Decide for the good on function signatures
- [ ] waveform generation: merge Yuto's contributions
- analysis:
  - [ ] Remove simulate1d, unify it with simulate 3d using the same mapper as in plotting
  - [ ] move diagonalize and other postprocesing in analysis, rather than in results
- mode_solver:
  - [ ] label modes of chain by projections on potential principal axes
  - [ ] pretty plots of normal modes
- Results:
  - [ ] make Result classes only container
  - [ ] save/load results
  - [ ] plot from results
  - [ ] SimulationResult container class?
  - [ ] Waveform container class?

## Examples

- [ ] Make sure there's at least one example for each use case discussed on the paper

## Paper - me, for the moment

- [ ] Intro
- [ ] Physics of RF trapping
- [ ] Math behind trap models
- [ ] Overview of pytrans
- [ ] Examples
- [ ] Conclusions

## Documentation - Francesco

- [ ] Function and class docstrings
- [ ] Cleanup comments and type hints
- [ ] Package documentation

## Leave for next release

- [ ] Multi charge ions
