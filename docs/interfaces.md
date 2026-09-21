<!-- Moved out of README.md on 2026-09-21: that file stood at 403 lines,
     past the three-hundred-line limit every file in this workshop lives
     under. Nothing here was rewritten. -->

# Interfaces, contracts and operation

## Primary interfaces

### Released runnable modules

Representative module invocation:

```bash
python -m wisent.scripts.benchmark_evaluation.math_coding.run_aime_evaluation --help
```

Use `released-surface.json` for the exact 0.1.111 module list. Each runner owns
its arguments and output; inspect its `--help` and source before launching a
costly or stateful job.

### Activation extraction

The current raw extraction path requires an immutable model URI/digest:

```bash
export STADO_MODEL_URI='stado://wisent-tools/models/<archive>.tar.gz'
export STADO_MODEL_SHA256='<64-lowercase-hex-sha256>'
python -m wisent.scripts.extract_raw_activations \
  --device cuda \
  --max-retries 3 \
  --log-interval 25
```

This path loads a local-only Transformers model after Stado materialization and
writes to the configured activation database. The example is a contract shape,
not permission to run it against production data.

### Quality-metrics sweep

`wisent/scripts/run_quality_metrics_sweep.sh` requires immutable model and dataset
URIs/digests plus `SWEEP_ID`. It runs a fixed benchmark/synthetic suite around
`wisent optimize-steering`, persists intermediate output, resumes from marker
files, and intentionally continues after individual failures.

A completed shell process can therefore contain failed benchmark entries. Inspect
its combined result and failed/completed lists rather than treating process
completion alone as success.

## Stado object interface

Current source provides:

```python
from wisent.stado import StadoClient

client = StadoClient()  # reads STADO_API_URL and STADO_API_TOKEN
objects = client.list_uri("stado://wisent-tools/evaluations")
```

`StadoClient` supports:

- `get_bytes`, `get_text`, `get_json`, and atomic `get_file`;
- `put_bytes`, canonical `put_json`, streamed `put_file`, and create-only
  `if_absent` writes;
- `stat`, `list`, `list_uri`, and `delete`;
- symlink-rejecting `put_tree` and traversal-filtering `get_prefix`.

The URL must be absolute HTTP(S), contain no embedded credentials/query/fragment,
and use HTTPS except authenticated loopback. `stado://` URIs are validated before
requests. The bearer token is sent in the `Authorization` header.

CLI surface in current source:

```bash
python -m wisent.stado list stado://<namespace>/<prefix>
python -m wisent.stado has-prefix stado://<namespace>/<prefix>
python -m wisent.stado put-tree stado://<namespace>/<prefix> <directory>
python -m wisent.stado put-tree --sync stado://<namespace>/<prefix> <directory>
python -m wisent.stado get-prefix stado://<namespace>/<prefix> <directory>
```

`has-prefix` reserves exit `1` for a legitimate absent answer. Retryable dependency
failure exits `69`; invalid configuration/input and non-retryable errors use the
failure contract instead of masquerading as absence.

## Immutable private-input contract

Current `wisent.stado_inputs` recognizes product-owned paths:

- `stado://wisent-tools/models/...` with `STADO_MODEL_URI` and
  `STADO_MODEL_SHA256`;
- `stado://wisent-tools/datasets/...` with dataset URI/digest variables;
- `stado://wisent-tools/evaluations/<STADO_EVALUATION_ID>/<file>.json` for
  create-only result publication.

Archive extraction accepts regular files/directories only and rejects traversal,
links, devices, and unsafe members. A machine job (`WC_JOB_ID` present) must use
pre-staged inputs and writes immutable results to `STADO_JOB_OUTPUT_DIR` or
`./output`; an operator process can use the authenticated Stado API.

## Failure semantics

`wisent.failure` classifies configuration, authentication, not-found, rate-limit,
timeout, infrastructure-down, and unknown failures. A classification carries
service, impact, severity, retryability, outage status, and an exit-code decision.

Sensitive key/token/password fragments are redacted from the structured operator
line. User-facing messages omit raw upstream bodies. Debug traceback output must
still be treated as potentially sensitive.

## Security, privacy, and data handling

- Never commit or print `STADO_API_TOKEN`, database URLs, Supabase keys, model
  credentials, Hugging Face tokens, or customer dataset locations.
- Activation tensors and contrastive pairs can encode sensitive source prompts or
  model behavior. Apply access control, retention, deletion, and export policy.
- Verify exact SHA-256 values from an independent trusted manifest; do not accept
  a digest supplied beside an untrusted artifact as provenance.
- `put-tree --sync` deletes remote objects missing locally. Review the namespace
  and source tree before using it.
- Evaluation/result writes are intended to be immutable. A conflicting payload
  at the same URI is an error, not an overwrite path.
- Machine workloads must not bypass pre-staging by calling Stado directly.
- Database-writing extraction scripts require least-privilege credentials and a
  schema backup/recovery plan.
- Logs can expose model URIs, benchmark names, row counts, timings, paths, and
  diagnostics. Do not attach them unredacted to public issues.
- Model and dataset code/artifacts may execute loaders or contain unsafe formats;
  digest verification alone does not sandbox deserialization.

## Operational model

- **Configuration:** runner arguments plus workflow-specific environment for
  Stado, database, immutable input digests, output directories, and compute.
- **State:** database rows, local caches/temp directories, Stado objects, sweep
  progress, and generated result artifacts.
- **Credentials:** externally supplied; no broker or secret rotation is provided.
- **Observability:** runner stdout/stderr, classified failure lines, database
  progress, intermediate result files, and object metadata.
- **Recovery:** preserve immutable inputs, rerun idempotent/create-only stages,
  inspect partial benchmark failures, and restore database/object state using the
  owning service's procedures.
- **Cost:** model storage/transfer, GPU/CPU runtime, database writes, object
  retention, and operator review. The package has no checkout or metering.

