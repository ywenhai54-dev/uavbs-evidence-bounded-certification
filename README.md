# UAV-BS Evidence-Bounded Certification

Minimal research-code release for **Evidence-Bounded End-to-End Service
Certification and Deployment Refinement of UAV Base Stations over Real
Terrain** (ChinaNet 2026).

## Scope

The repository contains the code used for:

- four-stage nominal coverage (NC), access feasibility (AF), backhaul
  reachability (BR), and application admission (AA) certification;
- terrain/Fresnel/LoS-aware access and backhaul primitive construction;
- complete-domain LP/MILP lower and upper service bounds and witness checks;
- the corrected duplicate-state-cache CSD-MR search; and
- CT-KS and its Full-domain, Threshold-only, and relaxed-gap controls.

Only compact frozen result summaries are included. Search traces, solver logs,
debug output, historical experiments, manuscript files, and internal audits are
intentionally excluded.

The released formal runner differs from the frozen runner only in repository
module paths, the renamed packaging-only `openbase_adapters` module, and
relative input/output roots. The frozen and released SHA-256 values are
`9A93B631...02E0B6` and `81E94C6F...7FDFF`, respectively; no algorithm,
threshold, seed, or solver-setting change is present in the textual diff.

## Environment

The frozen CT-KS comparison used Python 3.12.7, SciPy 1.13.1, and
`scipy.optimize.milp` with its bundled HiGHS backend. Solver threads were fixed
to one and the solver seed to zero.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/check_frozen_summaries.py
```

## Data

The exact formal case schedule, frozen input protocol/hash manifest, and
OpenCelliD-derived GBS tables are included.
The following inputs are not redistributed:

- Amap-derived POI records and demand tables;
- candidate locations derived partly from those POI records;
- service primitives derived from restricted POI records and the DEMs; and
- DEM arrays, because the retained provenance reaches the frozen windows but
  does not conclusively identify the original DEM product and redistribution
  license.

See [`data/README.md`](data/README.md) and
[`data/formal_input_manifest.csv`](data/formal_input_manifest.csv). The manifest
provides hashes for matching an authorized private copy without publishing the
withheld records.

## Reproduction status and commands

The following lightweight verification command is available directly from the
public repository and was executed during release checks:

```bash
python scripts/check_frozen_summaries.py
```

A formal H6 rerun is **not available out of the box** because the public tree
intentionally omits restricted or license-unverified inputs. The formal runner
requires the complete 85-file input tree recorded in
`data/formal_inputs/protocol/preparation_manifest.json`: 60 demand-state CSVs,
four frozen UAV candidate tables, service-unit mappings/catalogs, and 11
access, link, and backhaul primitive files. The exact private inputs must be
restored under `data/formal_inputs/inputs/`, all hashes must match, and the
original preparation process must supply `data/formal_inputs/PREPARED.flag`.
Only then is the formal command applicable:

```bash
python scripts/run_main_50m_formal_search.py --run --horizon 6 --workers 1
```

The explicit `--horizon 6` argument is required. The remaining frozen settings
are recorded in `configs/formal_h6_protocol.json`. The formal runner consumes
precompiled primitives and does not read the DEM directly.

Rebuilding terrain, LoS, Fresnel, rate, and path primitives additionally
requires the withheld DEM and authorized POI/scenario inputs. When those exact
inputs are available, the retained entry point is:

```bash
python src/run_protocol_stack_exp3_service_primitives_compile.py \
  --config configs/protocol_stack_csd_mr_go_v1.yaml
```

The four-stage fixed-deployment certifier similarly requires the authorized
scenario and primitive inputs:

```bash
python src/run_protocol_stack_exp5_service_fragmentation_certifier.py \
  --config configs/protocol_stack_csd_mr_go_v1.yaml
```

`scripts/generate_main_result_tables.py` additionally requires the detailed H6
outputs, search trace, and frozen Exact Coverage result file; it cannot operate
on the compact summaries alone.

The frozen four-arm CT-KS comparison requires the authorized formal inputs and
the detailed formal H6 outputs under `outputs/formal_search/`:

```bash
python scripts/run_ctks_fair_mechanism_full.py --phase manifest
python scripts/run_ctks_fair_mechanism_full.py --phase outer --resume
python scripts/run_ctks_fair_mechanism_full.py --phase replay --resume
python scripts/summarize_ctks_fair_mechanism_full.py
```

These are expensive commands and are not executed as part of release checks.

## Included result summaries

`results/` contains the formal H6 confirmatory method summary, paired CSD-MR
versus BestStrong results, failure-layer means, final LB/UB certification-gap
summary, and compact CT-KS outer and fixed-candidate summaries. These files
support verification of the main reported aggregates without publishing
traces.

## Known limitation

The historical Full-domain fixed-candidate replay has three unresolved replay
records for one candidate triple. The preserved release audit therefore did not
claim complete replay verification. This does not alter the independently
checked CT-KS versus Threshold-only decision agreement of 1678/1692 (99.17%),
but the repository must not be described as having passed every historical
replay check.

## Licensing status

No software license has been selected in this staging copy. OpenCelliD-derived
tables are separately governed by CC BY-SA 4.0; see
[`data/opencellid_gbs/ATTRIBUTION.md`](data/opencellid_gbs/ATTRIBUTION.md).
Select a license for the original software before making the repository public.

