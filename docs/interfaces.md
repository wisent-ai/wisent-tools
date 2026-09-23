<!-- Moved out of README.md; the README links here. -->
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

