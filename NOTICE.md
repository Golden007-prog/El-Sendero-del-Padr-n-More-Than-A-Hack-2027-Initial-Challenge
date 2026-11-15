# Attribution and rights

## TRACE

Copyright (c) 2026 Oikantik Basu. TRACE source and the bundled fitted artifacts are distributed under the repository’s [MIT licence](LICENSE). Team: El Sendero del Padrón — Oikantik Basu, Aniket Pal and Sayan Chakraborty.

## Organizer software

This repository includes source from Metrica Sports’ *More than a Hack 2027: Frame-level Ball Possession*, commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`.

Copyright (c) 2026–2027 Metrica Sports. The complete upstream notice is retained in [licenses/METRICA_LICENSE](licenses/METRICA_LICENSE). Attribution: Metrica Sports and Barça Innovation Hub, version 1.0.0 (2026), [official repository](https://github.com/metrica-sports/more-than-a-hack-2027).

The upstream licence covers its software/documentation. It expressly excludes the gated competition match assets. The upstream file refers to files and data in its original repository; no public ground truth or match assets are bundled here.

## External training data

All five fitted models use the Sportec Open DFL release:

Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*.

- Dataset: [version 1, DOI 10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1).
- Paper: [DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y).
- Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), including its [legal code and disclaimer of warranties](https://creativecommons.org/licenses/by/4.0/legalcode.en).

TRACE converts units and coordinates, derives geometric/motion/temporal features, maps provider status and possession fields to targets, and fits statistical estimators. Four matches supply training and three supply development evidence. The original dataset is not included. This attribution does not imply endorsement by the authors, DFL, Sportec or TRACAB.

The trained artifacts are locally fitted models, not redistributed pretrained vision checkpoints. Dataset licensing is separate from the rights over the competition’s supplied match.

## Excluded data and dependencies

Competition tracking, video, Smart Tagging, homography, possession references, local caches and generated predictions are excluded. Authorized participants obtain the original challenge input bundle from its organizers under the applicable enrollment terms.

Runtime dependencies retain their respective licences. This repository provides pinned installation requirements rather than redistributing the Python runtime or dependency binaries.
