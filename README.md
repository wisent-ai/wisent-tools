# Wisent Tools

<!-- wisent-readme-signals:start -->
[![PyPI](https://img.shields.io/pypi/v/wisent-tools)](https://pypi.org/project/wisent-tools/)
[![PyPI downloads](https://img.shields.io/pypi/dm/wisent-tools)](https://pypi.org/project/wisent-tools/)
![License not declared](https://img.shields.io/badge/license-not%20declared-critical)
[![Discord](https://img.shields.io/badge/Discord-Join%20Wisent-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54)
<!-- wisent-readme-signals:end -->

**Wisent Tools is the operational companion package for the Wisent Python family:
it contains activation-extraction and benchmark-evaluation runners, quality-sweep
scripts, a provider-neutral Stado object client, immutable private-input helpers,
and shared failure reporting.**

It is not the core `wisent` model/steering library, not a hosted evaluation
service, and not a stable umbrella CLI. Most modules are specialized operator
workflows with heavyweight runtime and infrastructure dependencies.

[Install](#quick-start) · [Released surface](#released-versus-source-surface) ·
[Stado boundary](#stado-object-interface) ·
[Canonical repository](https://github.com/wisent-ai/wisent-tools)

Current boundary: `wisent-tools` 0.1.111 is present on PyPI and supports Python
3.9+. **The repository has no `LICENSE` file and declares no package license.**
Public source and registry availability do not grant a general right to copy,
modify, or redistribute it. Obtain an explicit license before redistribution or
derivative use.

## Problem and intended users

The core Wisent library should not absorb one-off benchmark runners, database
migration helpers, GPU extraction jobs, or infrastructure transfer code. Those
workflows still need versioned imports, one namespace, and explicit boundaries
for private model/dataset inputs and generated evidence.

Wisent Tools serves:

- **Wisent researchers** running AIME, APPS, CoNaLa, LiveMathBench, MATH,
  PolyMath, and related evaluation utilities;
- **activation-pipeline operators** extracting missing hidden-state records and
  uploading raw activations;
- **Stado workload authors** materializing immutable private model/dataset inputs
  and publishing create-only JSON results;
- **platform operators** moving provider-neutral object trees and receiving
  consistent, machine-meaningful failure codes.

## Product boundaries

### Included

- Python namespace modules under `wisent.scripts`;
- benchmark-evaluation runners and constant-analysis/reorganization utilities;
- raw and processed activation extraction/upload helpers;
- quality-metrics sweep orchestration around the core `wisent` CLI;
- `wisent.stado.StadoClient` for authenticated object get/put/list/stat/delete and
  tree/prefix transfer;
- `wisent.stado_inputs` for SHA-256-verified private model/dataset materialization
  and immutable evaluation results;
- `wisent.failure` for stable dependency failure codes, retry semantics, safe
  human messages, and redacted operator logs;
- `wisent.onboarding` for the durable `first-use` journey and its safe local
  `wisent.surface` result path;
- namespace coexistence with sibling `wisent-*` distributions.

### Explicit non-goals and limitations

- This package does not provide the main `wisent` command or core steering/model
  implementation; it depends on `wisent>=0.11.21`.
- Installing the package does not provision models, datasets, GPUs, Stado,
  Supabase/PostgreSQL, Hugging Face caches, or credentials.
- Many operator scripts import undeclared workflow dependencies such as PyTorch,
  Transformers, psycopg2, NumPy, datasets, or task-specific evaluators. The three
  declared dependencies are not a complete environment lock.
- There is no lockfile and no supported claim that every historical runner works
  with the newest transitive dependencies.
- Scripts can allocate large models, GPU memory, activation tensors, database
  rows, files, and object-storage traffic. Treat them as data/compute jobs, not
  lightweight library calls.
- Benchmark datasets and model artifacts have their own licenses and access
  terms. This repository does not grant rights to them.
- Some scripts are operational snapshots tied to Wisent database schemas and
  internal infrastructure; they are not stable public APIs merely because they
  are importable.
- The source tree can contain functionality not present in the latest published
  sdist. Do not infer availability from repository files alone.
- No package license is currently granted. This is a release blocker for safe
  external reuse and redistribution.

## Released versus source surface

`released-surface.json` records the public/importable surface extracted from the
published PyPI 0.1.111 source distribution. That release contains:

- activation Supabase helper exports;
- activation extraction, migration, and upload modules;
- benchmark runners for AIME, APPS, CoNaLa, LiveMathBench, MATH, and PolyMath;
- constant-analysis, dead-constant, reorganization, extraction, and fixer
  modules;
- the quality-metrics sweep shell script;
- an activation-extraction coverage-universe entry point supplied by the broader
  package family.

The current checkout also contains `wisent.stado`, `wisent.stado_inputs`, and
`wisent.failure`. They are described below because they are implemented source,
but they are **not listed in the recorded PyPI 0.1.111 released surface**. Pin and
inspect the exact distribution/revision required by an operational workflow.

## Core use cases

### Run a released benchmark module

- **Actor:** a researcher with an approved model/dataset environment.
- **Initial state:** the exact `wisent-tools` and compatible `wisent` versions plus
  task dependencies are installed.
- **Outcome:** a runner evaluates its named benchmark and writes that runner's
  result surface.
- **Boundary:** benchmark correctness, dataset licensing, prompts, scoring, and
  hardware behavior are runner-specific; there is no single universal output
  contract in this README.

### Extract missing activations

- **Actor:** an activation-pipeline operator.
- **Initial state:** immutable model input, database schema/credentials, compatible
  model libraries, and GPU/CPU device are available.
- **Outcome:** missing contrastive-pair activations are computed and persisted.
- **Boundary:** these scripts can modify production-like database state and can
  consume substantial compute. Review arguments and schema before execution.

### Move operator objects through Stado

- **Actor:** a platform/workload author using current source.
- **Initial state:** outside a machine job, `STADO_API_URL` and
  `STADO_API_TOKEN` identify an authorized Stado object API.
- **Outcome:** validated `stado://namespace/key` objects or bounded local trees
  move through the provider-neutral API.
- **Boundary:** machine jobs are prohibited from making remote Stado calls; inputs
  must be pre-staged and outputs written under the job output boundary.

### Materialize immutable private inputs

- **Actor:** a workload running current source.
- **Initial state:** model/dataset URI and expected SHA-256 environment variables
  point inside the `wisent-tools` namespace.
- **Outcome:** archives are downloaded outside a machine job or consumed from
  pre-staged inputs, verified, and safely extracted.
- **Boundary:** digest equality proves byte identity, not artifact safety,
  provenance, model behavior, or license.

## Architecture

```text
wisent-tools distribution (shared `wisent` namespace)
  │
  ├─ wisent.scripts
  │    ├─ benchmark_evaluation/*
  │    ├─ activations/*
  │    ├─ extract_* / fix_*
  │    └─ run_quality_metrics_sweep.sh
  │
  ├─ wisent.stado             current source object client / CLI
  ├─ wisent.stado_inputs      current source immutable input/result boundary
  └─ wisent.failure           current source failure taxonomy

external owners:
  wisent core/evaluators · Stado API · model/dataset stores · PostgreSQL/Supabase
  · PyTorch/Transformers/Hugging Face · GPU/worker runtime
```

The package uses `pkgutil.extend_path` because several distributions contribute
modules under the `wisent` namespace. Import behavior therefore depends on the
complete installed package set, not this wheel alone.

## Quick start

### Prerequisites

- Python 3.9 or newer;
- an isolated virtual environment;
- access to PyPI or an approved package mirror;
- explicit approval for this unlicensed package's intended use;
- workflow-specific dependencies, data, credentials, and hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "wisent-tools==0.1.111"
```

Confirm the installed distribution rather than assuming the checkout and PyPI
artifact are identical:

```bash
python -c 'from importlib.metadata import version; print(version("wisent-tools"))'
```

Expected result: `0.1.111` for the command above. Installation alone does not
make any benchmark or extraction workflow ready.

For repository development/source inspection:

```bash
git clone https://github.com/wisent-ai/wisent-tools.git
cd wisent-tools
python -m pip install -e .
```

An editable checkout exposes current source, including modules that may be absent
from the recorded published surface. Do not use editable installs for reproducible
production jobs.

The first-use journey below is part of the `0.1.112` source candidate; it is not
present in the recorded `0.1.111` PyPI artifact.

### First use: observe a real toolkit result

The first-use journey executes the documented, safe surface inspector. It reads
Python syntax and package metadata without importing operator modules, starting
their workloads, using the network, or writing product data:

```bash
wisent-tools-onboarding run
# Equivalent:
python -m wisent.onboarding run
```

The command returns one JSON document. `result.tool_call` contains `tool_id`
`wisent.surface`, empty `inputs`, and a structured `result.surface` array of
supported commands and exports; `result.onboarding` reports the pinned journey
identity and completion state. The journey records `tool_result_observed` and
completes only after that structured result is returned and validated. Installation,
authentication, `--help`, or an exit-zero process without this result cannot
complete first use.

For separate, resumable steps, `wisent-tools-onboarding run-tool` executes and
durably retains the structured result without completing; a later
`wisent-tools-onboarding inspect` validates that retained result and completes.
`status`, `reset`, and `abandon` expose the remaining lifecycle controls.

Progress and the canonical analytics outbox are saved before central delivery,
under `~/.local/state/wisent-tools/onboarding.json` by default. Override that
location with `WISENT_TOOLS_ONBOARDING_STATE`. When
`STADO_INTEGRATION_API_URL` and
`WISENT_TOOLS_STADO_INTEGRATION_TOKEN` are both set, the adapter uses only the
product-scoped `bundle.read`, `experiments.assign`, `events.collect`, and
`state.read` onboarding operations. Missing or unavailable central delivery
uses the bundled, content-hashed `first-use` journey and never blocks the local
result.

## Interfaces, contracts and operation

The released runnable modules, the Stado object interface, the immutable
private-input contract, what `wisent.failure` classifies, the security and
privacy rules and the operational model are in
[docs/interfaces.md](docs/interfaces.md).

## Project status and support

- **Maturity:** published operational package with a heterogeneous script surface;
  not a single stable product API.
- **Latest recorded distribution:** PyPI `wisent-tools` 0.1.111; consult PyPI for
  the current registry state before installing.
- **Compatibility:** Python 3.9+; each operator workflow has additional unpinned
  runtime constraints.
- **Issues:** [`wisent-ai/wisent-tools`](https://github.com/wisent-ai/wisent-tools/issues).
- **Security:** use private GitHub Security Advisories; never include credentials,
  private object URIs, model/dataset contents, activations, database excerpts, or
  unredacted logs in a public issue.
- **License:** **none declared in the repository or package metadata.** Do not
  infer redistribution or derivative-work permission from public visibility or
  PyPI publication. Obtain an explicit license from the rights holder.
