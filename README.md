# Adequacy-constraint conflicts: duration and self-discharge

Reproducibility code for the second manuscript by **Sergei Trashchenkov** (ORCID [0000-0001-8786-8336](https://orcid.org/0000-0001-8786-8336)). This is a local publication-preparation snapshot, not a published article or an uploaded archive. Article metadata in `CITATION.cff` remain pending. The first paper is not modified.

## Install (Python 3.12)

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh quick
```

All package versions, including transitive dependencies, are pinned. SciPy provides HiGHS; no separate HiGHS installation is needed. Pandas is retained because the supplied Rye preparation and annual runner already depend on it. The search accelerator requires a C++17 compiler (`c++`); it is built locally from `numerics/search/engine.cpp`, not shipped as a platform-specific binary. Quick mode does not require a compiler or network access.

Always run commands **from this repository root**. No external working folders or old virtual environments are required. Existing reference numbers are preserved: `full` copies code and inputs into a separate run tree and never overwrites them.

## Reproduction levels

```sh
./reproduce.sh quick                       # seconds to minutes; numerical assertions + saved Rye table
./reproduce.sh full                        # all blocks; each capped at 30 minutes
./reproduce.sh full theory figures         # selected independent blocks
./reproduce.sh full search refinement --work runs/search_trial
./reproduce.sh full rye --work runs/rye_trial
```

`quick` prints `ok`/`FAIL` with expected values, results and tolerances; it exits nonzero on a failed numerical check. It recomputes the small MILP examples, four shore points and one daily tank point. Rye is explicitly a **saved-results reconstruction**, not a fresh 8280-hour optimization. Its known manuscript discrepancy (series 3–9 h instead of 3–5 h) is printed separately as `DISCREPANCY`, not silently accepted as a matching claim.

`full` has ten independently timed blocks plus the multicharge block. Default output is `runs/<timestamp>/`. Use a new `--work` directory for reruns; overwrites are rejected. `refinement` requires the completed `search` block in the same work directory. `validation/reproduce.py` is the authoritative command list. A process exit status only establishes successful execution, **not that an uncertain MILP result was resolved**; inspect saved statuses and `REPORT_RU.md` / `validation/full_summary.json`.

| Block | Expected wall time (machine-dependent) | Scope |
|---|---:|---|
| theory | <1 min | exact examples, rational one-charge checks, shore and deep-shore / T≤3 checks |
| monotone | <2 min | 500 equal-efficiency, 200 unequal-efficiency cases; 30 capacity-bound tests; secondary LP/MILP checks |
| duration | 1–5 min | independent LP/support enumeration, half/quarter/triple bounds |
| abundant | 2–10 min | abundant-window cuts and abstract enumeration; near-quarter profile |
| multicharge | 10–20 min | targeted multi-charge and heterogeneous-fleet searches; wall-clock budgets retained |
| selfdischarge | up to 30 min; may exceed cap | full original G2 sample, refinement, daily profiles |
| figures | <2 min | recompute all four daily tank points and generate the three manuscript PDFs |
| rye | up to 25 min | original 1200 s annual grid budget plus independent physical witness replay |
| yakutia | up to 15 min | original 720 s practical grid budget, independent small checks and summary |
| search | <2 min typical, 1100 s cap in source | all sign structures to T=10; 650 evaluations/structure parameter |
| refinement | 3–20 min | original 1000 s refinement budget, negative-only supplement, exact examples, coverage audit |

Wall-clock-limited grids can produce different intervals, solver statuses and numbers of evaluations on another machine. A timeout or unknown solver status is never interpreted as a zero gap. The sign-structure enumeration is exhaustive in the stated discrete domain; continuous parameter optimization is **not a global proof**.

## Manuscript → executable → reference value

| Label / result | Command from root | Expected / scope |
|---|---|---|
| `ex:quarter`, Fig. 1 | `python checks/quick.py`; `python figures/make_figures.py` | Γ=1/4; H=3, D=2 |
| `thm:half`, sharpness | `python checks/quick.py`; `python checks/g4/g4_check.py` | Γ=1/2 for two pinned hours; not the differently scaled H/D example |
| `thm:quarter` (one charge) | `python checks/exact_theory.py`; `python checks/g4/g4_check.py` | rational budget inequality and exact equality examples |
| `thm:quarter` (abundant, fleets) | `python checks/g6/g6_abstract.py`; `python checks/g6/g6_verify.py`; `python checks/g6/g6_more.py` | abstract search best 0.24573, the manuscript's 0.246; MILP of that profile stays below P/4; fleet samples find no larger ratio |
| `conj:quarter`, search to T=10 | `python numerics/search/search.py --seconds 1100 --evals 650 --historical-initial`; then `refine.py`, `negative_supplement.py`, `rational_check.py`, `summarize.py` in the same directory | 2046 structures, 4,083,351 evaluations, best Γ=1/4; continuous global optimum unproven |
| targeted multi-charge search to T=16 | `python checks/g6/g6_search.py` | 127830 candidates, recorded maximum 1/4 on known examples, no point above 1/4 |
| `prop:compat` | `python checks/monotone/run_checks.py`; `python checks/monotone/secondary_checks.py` | 500 common-η tests; 200 unequal-η controls; failures of domination with unequal η are expected counterexamples |
| `prop:triple` | `python checks/g4/g4_check.py` | checked triple-bound examples and sweeps |
| `ex:sd`, `prop:three` | `python checks/quick.py` | 4/95; capacity-limited and surplus-limited formulas, η=1 and 1/2 |
| `prop:shore`, Fig. 2 | `python checks/quick.py`; `python checks/g3/g3_analytic.py` | four numerical shore-formula checks; constructive algebra |
| `thm:packets`, self-discharge comparison | `python checks/g3/g3_analytic.py`; `python checks/g2/g2_analysis.py` | formula/construction checks and original numerical sample; proofs remain in manuscript |
| `thm:T3`, deep shore | `python checks/g5/g5_t3.py`; `python checks/g5/g5_coast.py`; `python checks/g5/g5_coast_rand.py` | declared small examples and 40 randomized deep-shore checks |
| worst sample T≤6 | `python checks/g2/g2_analysis.py refine` | approximately 0.8 δ Pᵈ; small-model numerical result, not a global claim |
| daily tank, Fig. 3 | `python figures/make_figures.py` | Γ upper endpoint 0.0066871643 at δ=.01, E=3, Pᶜ=Pᵈ=1 |
| Yakutia | `python numerics/yakutia/run.py`; `python numerics/yakutia/summarize.py` | original bounded grid; unresolved mixed-storage cases remain unresolved |
| Rye | `python numerics/quick.py`; `python numerics/rye/run.py`; `python numerics/rye/replay.py` | 51 saved rows; H/D max upper bound .0868493454 kW; tank [.0139999609,.0475312024] kW |

Direct scripts may write results beside their code. Prefer `reproduce.sh full` for a fresh isolated run. All model parameters and budgets are inherited; this packaging does not introduce a new research grid.

## Layout and solver contract

- `src/ref_milp.py`: **one** reference MILP, the original `theory_checks/g6/ref_milp.py`. Includes ENS, LOLH, self-discharge and optional series constraint D. D is an integer number of intervals; manuscript calculations use τ=1 h. Without D it matched the base solver on 240 deterministic comparisons (two storage-boundary conditions, ENS/LOLH objectives, caps and τ values). See `validation/solver_equivalence.json`; quick replays these frozen baseline answers.
- `checks/`: retained scientific checks. `g2_solver`, `g5_core`, `monotone/independent_model`, the Rye witness model and the Yakutia sparse model are **independent formulations**, not copied reference solvers. They retain different verification and witness interfaces; replacing them with a reference wrapper would destroy independence and annual-run diagnostics. Every actual reference-solver import resolves to `src/`.
- `numerics/`: original finite-budget annual studies and sign-structure search; frozen JSON/CSV references are separate from reruns.
- `figures/make_figures.py`: generates `figures/output/fig_quarter.pdf`, `fig_shore.pdf`, `fig_gamma_delta.pdf`.
- `data/`: original/processed Rye and hash-verified derived Yakutia inputs; see [data attribution](data/README.md).
- `validation/`: baseline regression, quick results, packaging audit, original-source SHA256 list and adaptation notes.

## Archived evidence and limitations

Bulk search traces, original Rye dispatch witnesses and its >5 MB independent-audit JSON are kept **outside Git**, in the sibling `adequacy-conflicts-artifacts/zenodo_extra/` delivery folder. See its README, archive manifest and SHA256 sums. No repository payload file exceeds 5 MB; virtual environments and newly generated run trees are not distribution content.

The clean repository does not need these optional archives for quick or fresh full runs. To independently replay the original Rye witnesses, extract `rye_witnesses.tar.gz` outside the repo and run:

```sh
python numerics/rye/replay.py --witnesses ../adequacy-conflicts-artifacts/original_rye/witnesses --output validation/original_rye_replay.json
```

See [REPORT_RU.md](REPORT_RU.md) for verified results and manuscript discrepancies. Working logs, task files, G-series reports, duplicate reference solvers, platform binaries, unrelated exploration and Idjwi are excluded. The original manuscript and source-code tree remain read-only and unchanged.

## License and citation

Software: MIT (`LICENSE`). Rye: CC BY 4.0, Aaslid and Klæboe, DOI 10.5281/zenodo.4448894. PVGIS/ERA5 terms and transformations: [data/README.md](data/README.md). Third-party datasets are not relicensed as MIT. Citation metadata: `CITATION.cff`. No remote repository is configured, and nothing has been pushed or uploaded.
