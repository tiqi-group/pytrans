# Pytrans

Split the remaining work in 4 areas

## Coding - me, Francesco

- [x] Classical simulator
- [x] Decent plotting API, independent from trap axes orientation
- [x] waveform generation: merge Yuto's contributions
- [x] Move trap filter to "Waveform transform" API
- [x] Decide for the good on function signatures
- [ ] Waveform container class?
- analysis:
  - [ ] Remove simulate1d, unify it with simulate 3d using the same mapper as in plotting
  - [x] move diagonalize and other postprocesing in analysis, rather than in results
  - [ ] Test for nan values
- mode_solver:
  - [ ] label modes of chain by projections on potential principal axes
  - [ ] pretty plots of normal modes
- Results:
  - [x] make Result classes only container
  - [ ] save/load results
  - [ ] plot from results
  - [ ] SimulationResult container class?

## Examples

Make sure there's at least one example for each use case discussed on the paper

- [ ] 01 - static potential SegmentedTrap with Be-Ca-Ca-Be crystal
- [ ] 02 - transport SurfaceTrap
- [ ] 03 - 3d simulation of transport (same section)

not in the paper

- [ ] 04 - double well potential
- [ ] 05 - mode solver with HarmonicTrap and long ion strings (James / Monroe papers)
- [ ] 06 - filter-compensated sawtooth
- [ ] 07 - showcase plotting

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
