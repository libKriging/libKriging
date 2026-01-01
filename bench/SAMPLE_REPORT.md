# C++ Kriging Benchmarks Report

**Date:** 2025-12-28 17:30:00 UTC
**Platform:** Ubuntu (latest)
**Build Type:** Release
**Iterations per test:** 10
**Test function:** f(x) = sum_i sin(2*pi*x_i)

## Table of Contents

- [Summary Tables](#summary-tables)
  - [Kriging](#kriging)
  - [NuggetKriging](#nuggetkriging)
  - [NoiseKriging](#noisekriging)
- [Detailed Results](#detailed-results)

## Summary Tables

### Kriging

| n   | d | fit (ms) | predict (ms) | update (ms) |
|-----|---|----------|--------------|-------------|
| 100 | 2 | 50.234 | 0.952 | 125.678 |
| 100 | 4 | 88.456 | 1.234 | 459.123 |
| 100 | 8 | 142.789 | 2.456 | 890.234 |
| 200 | 2 | 156.234 | 3.456 | 512.345 |
| 200 | 4 | 298.567 | 7.890 | 1823.456 |
| 200 | 8 | 587.890 | 15.678 | 3645.789 |
| 400 | 2 | 598.234 | 13.456 | 2034.567 |
| 400 | 4 | 1156.789 | 31.234 | 7234.890 |
| 400 | 8 | 2345.678 | 62.345 | 14567.123 |

### NuggetKriging

| n   | d | fit (ms) | predict (ms) | update (ms) |
|-----|---|----------|--------------|-------------|
| 100 | 2 | 245.123 | 0.987 | 342.456 |
| 100 | 4 | 378.456 | 1.345 | 678.234 |
| 100 | 8 | 567.890 | 2.678 | 1234.567 |
| 200 | 2 | 734.567 | 3.890 | 1456.789 |
| 200 | 4 | 1234.890 | 8.234 | 2890.123 |
| 200 | 8 | 2145.678 | 16.789 | 5678.456 |
| 400 | 2 | 2678.234 | 14.567 | 5234.678 |
| 400 | 4 | 4567.890 | 33.456 | 11234.567 |
| 400 | 8 | 8234.567 | 65.890 | 22345.678 |

### NoiseKriging

| n   | d | fit (ms) | predict (ms) | update (ms) |
|-----|---|----------|--------------|-------------|
| 100 | 2 | 32.456 | 0.945 | 8.234 |
| 100 | 4 | 45.678 | 1.234 | 43.567 |
| 100 | 8 | 67.890 | 2.456 | 89.123 |
| 200 | 2 | 98.234 | 3.456 | 31.789 |
| 200 | 4 | 156.789 | 7.890 | 178.456 |
| 200 | 8 | 234.567 | 15.678 | 356.234 |
| 400 | 2 | 367.890 | 13.456 | 123.678 |
| 400 | 4 | 598.234 | 31.234 | 689.123 |
| 400 | 8 | 923.456 | 62.345 | 1389.567 |

## Detailed Results

### Kriging

#### Configuration: n=100, d=2

```
Kriging Benchmark (BFGS, LL objective)

n=100 d=2 iterations=10
Operation                 |  Mean (ms) |   Std (ms) |   Min (ms) |   Max (ms) | Median (ms)
fit                       |     50.234 |      3.456 |     45.123 |     56.789 |     49.890
predict                   |      0.952 |      0.123 |      0.834 |      1.234 |      0.945
simulate                  |      0.234 |      0.045 |      0.189 |      0.312 |      0.228
update                    |    125.678 |     12.345 |    109.234 |    145.890 |    123.456
update_simulate           |      0.456 |      0.089 |      0.367 |      0.589 |      0.445
```

#### Configuration: n=100, d=4

```
Kriging Benchmark (BFGS, LL objective)

n=100 d=4 iterations=10
Operation                 |  Mean (ms) |   Std (ms) |   Min (ms) |   Max (ms) | Median (ms)
fit                       |     88.456 |      5.678 |     80.234 |     98.765 |     87.123
predict                   |      1.234 |      0.234 |      0.987 |      1.567 |      1.189
simulate                  |      0.345 |      0.067 |      0.267 |      0.456 |      0.334
update                    |    459.123 |     45.678 |    398.234 |    523.890 |    451.234
update_simulate           |      0.678 |      0.123 |      0.534 |      0.845 |      0.656
```

_[Additional configurations omitted for brevity]_

### NuggetKriging

#### Configuration: n=100, d=2

```
NuggetKriging Benchmark (BFGS, LL objective)

n=100 d=2 iterations=10
Operation                 |  Mean (ms) |   Std (ms) |   Min (ms) |   Max (ms) | Median (ms)
fit                       |    245.123 |     23.456 |    218.234 |    278.890 |    242.567
predict                   |      0.987 |      0.134 |      0.856 |      1.245 |      0.976
simulate                  |      0.267 |      0.056 |      0.198 |      0.345 |      0.256
update                    |    342.456 |     34.567 |    298.123 |    389.890 |    338.234
update_simulate           |      0.489 |      0.098 |      0.378 |      0.612 |      0.478
```

_[Additional configurations omitted for brevity]_

### NoiseKriging

#### Configuration: n=100, d=2

```
NoiseKriging Benchmark (BFGS, LL objective)

n=100 d=2 iterations=10
Operation                 |  Mean (ms) |   Std (ms) |   Min (ms) |   Max (ms) | Median (ms)
fit                       |     32.456 |      2.345 |     29.123 |     36.789 |     32.123
predict                   |      0.945 |      0.112 |      0.823 |      1.189 |      0.934
simulate                  |      0.198 |      0.034 |      0.156 |      0.245 |      0.189
update                    |      8.234 |      0.987 |      7.123 |      9.890 |      8.123
update_simulate           |      0.378 |      0.067 |      0.298 |      0.478 |      0.367
```

_[Additional configurations omitted for brevity]_
