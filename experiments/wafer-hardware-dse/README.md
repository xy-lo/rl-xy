# Wafer hardware design exploration

The H0/H1/H2/H3 profiles, static mesh and capacity preflight, research scripts and analysis are provided as exploratory artifacts. Run the portable preflight tests from the repository root:

    PYTHONPATH=src python3 -m unittest discover -s experiments/wafer-hardware-dse -p test_architecture.py
    PYTHONPATH=src python3 experiments/wafer-hardware-dse/architecture.py --output /tmp/wafer-preflight.json

See RESULTS.md for the user-provided 4B-32B table and limitations. H0 is a historical result under a different contract. 32B uses the Math n32 trace, TP16 fixed-layout generation, approximate KV replication and group-level training help. No full shared-link contention, KV preemption, physical hardware calibration, or end-to-end native wide-owner implementation is claimed. 32B H3 is numerically faster than H0 in the supplied table, which does not prove a same-contract ranking.

The scripts in run/ preserve the research prototype: they refer to a separate experiment workspace and frozen cost/plan assets and are NOT a self-contained command-line workflow from this repository. The data/ folder includes the 32B length trace and sampled TP16 cost tables, not physical measurements. Additional runtime code and earlier 4B-14B cost catalogs and plans are required to replay those scripts. Do not interpret the tests above as validation of the complete simulation or its capacity assumptions.
