# Rethinking Rao-Blackwellized State Space Models

In `GeneralisedFilters` we dispatch on the `calc_` family of functions to compute the linear model given model parameters (stored in the process object), inner states, and outer states. For AD this imposes inefficiencies when we need to compute `calc_` only once in the case where the model treats that matrix statically.

The idea behind this testing suite is to run benchmarks on the Kalman filter process to ensure that the underlying process is efficient for experimental implementations.

Use `SSMProblems` as a backend to ensure sampling paths is trivial and consistent with other models.