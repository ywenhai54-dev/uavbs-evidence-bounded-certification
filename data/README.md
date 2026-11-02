# Data availability

This directory contains only data that can be distributed with the current
evidence of provenance and licensing.

## Included

- `formal_inputs/inputs/case_index_50m.csv`: the frozen 60-case schedule, including
  the 48 confirmation cases (seeds 1--4).
- `formal_inputs/protocol/`: the frozen experimental protocol and the SHA-256
  manifest for all 85 prepared formal-input CSVs. These files contain metadata
  and hashes, not the withheld records.
- `opencellid_gbs/S1`--`S4/gbs_0.csv`: the frozen GBS locations derived from
  OpenCelliD. They are distributed under CC BY-SA 4.0 with attribution.
- `formal_input_manifest.csv`: relative paths, sizes, SHA-256 hashes, release
  status, and the reason for each inclusion or exclusion.

## Withheld

The formal POI and demand records originate from Amap-derived tables. Amap's
platform terms do not provide permission to redistribute those records as a
downloadable research dataset. Candidate locations also depend on POI geometry,
so they are withheld rather than treated as license-free derivatives.

The four frozen DEM windows are also withheld. Their internal hashes and
geometric metadata are preserved, but the retained local provenance does not
conclusively identify the upstream DEM product and redistribution license.

Consequently, this public staging copy supports code inspection and aggregate
result verification, but not a clean-room rerun of all 48 formal cases. Authors
with authorized inputs can restore the missing files under
`data/formal_inputs/inputs/` and verify them against
`formal_inputs/protocol/preparation_manifest.json`. `PREPARED.flag` is
deliberately absent because the public input tree is incomplete.

Sources and terms:

- Amap platform terms: https://lbs.amap.com/pages/terms
- OpenCelliD attribution: https://docs.opencellid.org/docs/attribution

