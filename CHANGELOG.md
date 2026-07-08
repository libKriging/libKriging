# Changelog

All notable changes to libKriging are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

This file was introduced during the 1.x cycle. For the detailed notes of each
past release, see the corresponding entry on the
[GitHub releases page](https://github.com/libKriging/libKriging/releases).

## [Unreleased]

## [1.1.0] - 2026-07-08

### Added
- Vecchia approximated log-likelihood objective `VLL(m)`, with local prediction
  and a factorization-free "light" mode (#318).

### Fixed
- Fork-after-threads deadlock in forked child processes (#319).
- Windows CI on the `windows-2025-vs2026` runner image: CMake pinned to the
  version providing the "Visual Studio 18 2026" generator, and Octave/conda
  setup (#320).
- Thread Sanitizer job: removed false-positive data races caused by GCC's
  uninstrumented OpenMP runtime (libgomp) (#320).

### Documentation
- Documentation, licensing and metadata review: fixed stale dependency and
  architecture docs, added scientific and input-warping references, added
  `CITATION.cff`, `NOTICE` and this changelog, and README features/license/
  citation sections (#321).

## Released versions

| Version | Date | Notes |
|:--------|:-----|:------|
| [1.1.0](https://github.com/libKriging/libKriging/releases/tag/v1.1.0) | 2026-07-08 | Vecchia VLL objective; fork/threads, Windows CI and TSan fixes; docs & licensing review. |
| [1.0.0](https://github.com/libKriging/libKriging/releases/tag/v1.0.0) | 2026-05-13 | First stable 1.0 release. |
| [0.9.3](https://github.com/libKriging/libKriging/releases/tag/v0.9.3) | 2026-01-18 | |
| [0.9.2](https://github.com/libKriging/libKriging/releases/tag/v0.9.2) | 2025-12-17 | |
| [0.9.1](https://github.com/libKriging/libKriging/releases/tag/v0.9.1) | 2025-01-14 | |
| [0.9.0](https://github.com/libKriging/libKriging/releases/tag/v0.9.0) | 2024-09-04 | |
| [0.8.3](https://github.com/libKriging/libKriging/releases/tag/v0.8.3) | 2023-12-10 | |
| [0.8.2](https://github.com/libKriging/libKriging/releases/tag/v0.8.2) | 2023-12-10 | |
| [0.8.0](https://github.com/libKriging/libKriging/releases/tag/v0.8.0) | 2023-05-23 | |
| [0.7.4](https://github.com/libKriging/libKriging/releases/tag/v0.7.4) | 2023-01-13 | |
| [0.7.3](https://github.com/libKriging/libKriging/releases/tag/v0.7.3) | 2023-01-09 | |
| [0.7.2](https://github.com/libKriging/libKriging/releases/tag/v0.7.2) | 2022-12-23 | |
| [0.7.1](https://github.com/libKriging/libKriging/releases/tag/v0.7.1) | 2022-12-23 | |
| [0.7.0](https://github.com/libKriging/libKriging/releases/tag/v0.7.0) | 2022-10-06 | |
| [0.6.0](https://github.com/libKriging/libKriging/releases/tag/v0.6.0) | 2022-05-24 | |
| [0.5.1](https://github.com/libKriging/libKriging/releases/tag/v0.5.1) | 2022-04-07 | |
| [0.4.8](https://github.com/libKriging/libKriging/releases/tag/v0.4.8) | 2021-12-05 | |
| [0.4.7](https://github.com/libKriging/libKriging/releases/tag/v0.4.7) | 2021-09-05 | |
| [0.4.5](https://github.com/libKriging/libKriging/releases/tag/v0.4.5) | 2021-09-02 | |
| [0.4.4](https://github.com/libKriging/libKriging/releases/tag/v0.4.4) | 2021-09-02 | |
| [0.4.3](https://github.com/libKriging/libKriging/releases/tag/v0.4.3) | 2021-08-30 | |
| [0.4.2](https://github.com/libKriging/libKriging/releases/tag/v0.4.2) | 2021-06-01 | |
| [0.4.1](https://github.com/libKriging/libKriging/releases/tag/v0.4.1) | 2021-05-31 | First public pre-releases. |

[Unreleased]: https://github.com/libKriging/libKriging/compare/v1.1.0...master
[1.1.0]: https://github.com/libKriging/libKriging/compare/v1.0.0...v1.1.0
