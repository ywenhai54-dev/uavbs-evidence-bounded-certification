# Frozen formal-input contract

This directory preserves the redistributable part of the formal H6 input
contract:

- `protocol/frozen_protocol.json`: frozen experimental design and source hashes;
- `protocol/preparation_manifest.json`: SHA-256 hashes for all 85 prepared CSV
  inputs; and
- `inputs/case_index_50m.csv`: the redistributable 60-case schedule.

The public repository intentionally does **not** contain `PREPARED.flag`. That
sentinel means that the complete private input tree passed preparation checks,
whereas this release withholds restricted or license-unverified inputs.

A full formal H6 run additionally requires the exact files listed in
`protocol/preparation_manifest.json`: 60 demand-state CSVs, four frozen UAV
candidate tables, service-unit mapping/catalog tables, and 11 access, link, and
backhaul primitive files. The primitives are the direct inputs consumed by the
formal runner; the DEM and raw POI/GBS records are upstream inputs needed only
to rebuild them.

Do not create `PREPARED.flag` or run the formal search until every private input
has been restored at the recorded relative path and its SHA-256 matches the
preparation manifest.
