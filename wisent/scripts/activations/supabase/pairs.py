"""Bridge between extract_and_upload and the Supabase ContrastivePair catalog.

Supabase project rbqjqnouluslojmmnuqi (Wisent App, eu-west-2) holds the
authoritative pair catalog:
  - ContrastivePairSet: 205 benchmarks named "<category>/<benchmark>"
    e.g. "commonsense/hellaswag", "coding/humaneval".
  - ContrastivePair: 160,317 pairs across those sets. Each row carries
    a stable Postgres id, positiveExample text, negativeExample text.
  - RawActivation: schema for per-(model, pair, layer, side, format)
    activation references. Currently empty; this module writes to it
    so coverage queries become trivial SQL.

The HF safetensors path used positional pair_ids (0..N-1) into a
temporary lm-eval-generated pairs file. Those positional ids were not
stable across re-runs and not linked to Supabase. Going forward,
extractions should use ContrastivePair.id as the pair_id stored in
shard metadata, so a single SQL query against RawActivation can answer
"which pairs are already extracted for this model+strategy+layer".

Requires SUPABASE_ACCESS_TOKEN env var. On macOS dev hosts the
Supabase CLI also stores the token in the Keychain under service name
"Supabase CLI" (`security find-generic-password -s "Supabase CLI"`),
which is auto-read when the env var is unset.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


WISENT_APP_PROJECT = "rbqjqnouluslojmmnuqi"
MANAGEMENT_API = "https://api.supabase.com/v1/projects"


def _access_token() -> str | None:
    """Find the Supabase access token. Order: env var, then
    gs://wisent-compute/config/supabase_access_token (read with the
    agent ADC the queue uses), then macOS Keychain on dev hosts."""
    tok = os.environ.get("SUPABASE_ACCESS_TOKEN", "").strip()
    if tok:
        return tok
    try:
        from google.cloud import storage
        bn = os.environ.get("WC_BUCKET", "wisent-compute")
        blob = storage.Client().bucket(bn).blob("config/supabase_access_token")
        if blob.exists():
            content = blob.download_as_text().strip()
            if content:
                return content
    except Exception:
        pass
    if os.uname().sysname == "Darwin":
        import subprocess
        try:
            raw = subprocess.check_output(
                ["/usr/bin/security", "find-generic-password",
                 "-s", "Supabase CLI", "-w"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if raw.startswith("go-keyring-base64:"):
                import base64
                return base64.b64decode(raw.split(":", 1)[1]).decode()
            return raw
        except Exception:
            pass
    return None


def _run_query(query: str, project_ref: str = WISENT_APP_PROJECT) -> list:
    """POST a SQL query to Supabase Management API and return JSON rows."""
    token = _access_token()
    if not token:
        raise RuntimeError(
            "no Supabase access token (set SUPABASE_ACCESS_TOKEN env var)"
        )
    req = urllib.request.Request(
        f"{MANAGEMENT_API}/{project_ref}/database/query",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "wisent-tools-supabase-bridge/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _escape_sql_literal(s: str) -> str:
    """Escape a Python string for embedding in a SQL single-quoted literal."""
    return s.replace("'", "''")


def resolve_set_id(task_name: str) -> int | None:
    """Map an HF lm-eval task name to a Supabase ContrastivePairSet.id.

    Supabase names follow "<category>/<benchmark>"; HF uses the bare
    lm-eval task (e.g. "hellaswag"). This function matches by exact
    name OR by suffix on the slash boundary, preferring the shorter
    match when ambiguous."""
    escaped = _escape_sql_literal(task_name)
    query = (
        "SELECT id, name FROM \"ContrastivePairSet\" "
        f"WHERE name = '{escaped}' OR name LIKE '%/{escaped}' "
        "ORDER BY length(name) ASC LIMIT 1;"
    )
    rows = _run_query(query)
    if not rows:
        return None
    return int(rows[0]["id"])


def fetch_pairs(set_id: int, limit: int | None = None) -> list[dict]:
    """Fetch (id, positiveExample, negativeExample) rows for a setId.

    Returned in ascending id order. The Postgres id is the canonical
    pair identifier."""
    limit_clause = f"LIMIT {int(limit)}" if (limit and limit > 0) else ""
    query = (
        "SELECT id, \"positiveExample\" AS pos, \"negativeExample\" AS neg, "
        "category FROM \"ContrastivePair\" "
        f"WHERE \"setId\" = {int(set_id)} "
        f"ORDER BY id ASC {limit_clause};"
    )
    return _run_query(query)


def model_id_for_hf_id(hf_model_id: str) -> int | None:
    """Map an HF model id (e.g. 'meta-llama/Llama-3.2-1B-Instruct') to a
    Supabase Model.id. Returns None when the model is not registered."""
    escaped = _escape_sql_literal(hf_model_id)
    rows = _run_query(
        f"SELECT id FROM \"Model\" WHERE \"huggingFaceId\" = '{escaped}' "
        f"OR name = '{escaped}' LIMIT 1;"
    )
    if not rows:
        return None
    return int(rows[0]["id"])


def extracted_pair_ids_for(
    model_id: int,
    set_id: int,
    layer: int,
    prompt_format: str,
) -> set[int]:
    """Return the set of ContrastivePair.id values already covered in
    RawActivation for this (model, set, layer, format). Coverage means
    BOTH the positive and negative sides exist."""
    fmt = _escape_sql_literal(prompt_format)
    rows = _run_query(
        "SELECT \"contrastivePairId\" AS pid, "
        "BOOL_OR(\"isPositive\") AS has_pos, "
        "BOOL_OR(NOT \"isPositive\") AS has_neg "
        "FROM \"RawActivation\" "
        f"WHERE \"modelId\" = {int(model_id)} "
        f"AND layer = {int(layer)} "
        f"AND \"promptFormat\" = '{fmt}' "
        "AND \"contrastivePairId\" IN ("
        f"  SELECT id FROM \"ContrastivePair\" WHERE \"setId\" = {int(set_id)}"
        ") "
        "GROUP BY \"contrastivePairId\";"
    )
    return {int(r["pid"]) for r in rows if r.get("has_pos") and r.get("has_neg")}


def insert_raw_activations(rows: list[dict]) -> int:
    """Insert RawActivation rows in bulk. Each row dict must carry keys:
    modelId, contrastivePairId, contrastivePairSetId, layer, seqLen,
    hiddenDim, promptLen, hiddenStates (hex-encoded bytes), answerText,
    isPositive, promptFormat.

    Returns count of rows submitted. ON CONFLICT DO NOTHING against the
    @@unique([modelId, contrastivePairId, layer, isPositive,
    promptFormat]) constraint keeps re-runs idempotent."""
    if not rows:
        return 0
    values_sql_parts = []
    for r in rows:
        ans = _escape_sql_literal(r.get("answerText", "") or "")
        fmt = _escape_sql_literal(r["promptFormat"])
        values_sql_parts.append(
            f"({int(r['modelId'])}, {int(r['contrastivePairId'])}, "
            f"{int(r.get('contrastivePairSetId') or 0)}, "
            f"{int(r['layer'])}, {int(r['seqLen'])}, "
            f"{int(r['hiddenDim'])}, {int(r['promptLen'])}, "
            f"decode('{r['hiddenStates']}', 'hex'), "
            f"'{ans}', {bool(r['isPositive'])}, '{fmt}', NOW())"
        )
    values_sql = ",\n".join(values_sql_parts)
    query = (
        "INSERT INTO \"RawActivation\" ("
        '"modelId", "contrastivePairId", "contrastivePairSetId", '
        'layer, "seqLen", "hiddenDim", "promptLen", "hiddenStates", '
        '"answerText", "isPositive", "promptFormat", "createdAt"'
        ") VALUES " + values_sql +
        " ON CONFLICT (\"modelId\", \"contrastivePairId\", layer, "
        "\"isPositive\", \"promptFormat\") DO NOTHING;"
    )
    _run_query(query)
    return len(rows)


def pair_id_lookup_table(set_id: int) -> dict[str, int]:
    """Build a {sha256_key -> ContrastivePair.id} lookup table for every
    pair in the set. Each pair contributes its key by hashing the
    Supabase-stored positiveExample + \\x1f + negativeExample directly
    (the row text as-is, whatever shape the inserter chose). The
    extractor uses the same hash on its own pair data; for benchmarks
    where the inserter stored just answers (hellaswag/piqa style) the
    extractor hashes (model_response + \\x1f + neg_model_response);
    for benchmarks where the inserter stored prompt + newline-newline
    + answer the extractor builds that same shape. The extractor tries
    both keys against this table so either inserter convention
    matches."""
    import hashlib
    table: dict[str, int] = {}
    pairs = fetch_pairs(set_id, limit=None)
    for p in pairs:
        pos = p.get("pos", "") or ""
        neg = p.get("neg", "") or ""
        key = hashlib.sha256(
            (pos + "\x1f" + neg).encode("utf-8")
        ).hexdigest()[:16]
        table[key] = int(p["id"])
    return table


def insert_activations(rows: list[dict]) -> int:
    """Bulk-insert Activation rows. Each row dict needs:
      modelId, contrastivePairId, contrastivePairSetId, layer,
      neuronCount, extractionStrategy, activationData (hex bytes),
      isPositive.
    ON CONFLICT against the @@unique([modelId, contrastivePairId,
    layer, extractionStrategy, isPositive]) constraint keeps re-runs
    idempotent."""
    if not rows:
        return 0
    parts = []
    for r in rows:
        strat = _escape_sql_literal(r["extractionStrategy"])
        parts.append(
            f"({int(r['modelId'])}, {int(r['contrastivePairId'])}, "
            f"{int(r.get('contrastivePairSetId') or 0)}, "
            f"{int(r['layer'])}, {int(r['neuronCount'])}, "
            f"'{strat}', decode('', 'hex'), "
            f"{bool(r['isPositive'])}, NOW(), NOW())"
        )
    values_sql = ",\n".join(parts)
    query = (
        "INSERT INTO \"Activation\" ("
        '"modelId", "contrastivePairId", "contrastivePairSetId", '
        'layer, "neuronCount", "extractionStrategy", '
        '"activationData", "isPositive", "createdAt", "updatedAt"'
        ") VALUES " + values_sql +
        " ON CONFLICT (\"modelId\", \"contrastivePairId\", layer, "
        "\"extractionStrategy\", \"isPositive\") DO NOTHING;"
    )
    _run_query(query)
    return len(rows)


def extracted_pair_ids_via_activation(
    model_id: int,
    set_id: int,
    layer: int,
    extraction_strategy: str,
) -> set[int]:
    """Coverage query against the Activation table (not RawActivation).
    Returns pair ids for which BOTH positive and negative Activation
    rows exist at the given layer + strategy."""
    strat = _escape_sql_literal(extraction_strategy)
    rows = _run_query(
        "SELECT \"contrastivePairId\" AS pid, "
        "BOOL_OR(\"isPositive\") AS has_pos, "
        "BOOL_OR(NOT \"isPositive\") AS has_neg "
        "FROM \"Activation\" "
        f"WHERE \"modelId\" = {int(model_id)} "
        f"AND layer = {int(layer)} "
        f"AND \"extractionStrategy\" = '{strat}' "
        "AND \"contrastivePairId\" IN ("
        f"  SELECT id FROM \"ContrastivePair\" WHERE \"setId\" = {int(set_id)}"
        ") "
        "GROUP BY \"contrastivePairId\";"
    )
    return {int(r["pid"]) for r in rows if r.get("has_pos") and r.get("has_neg")}


__all__ = [
    "resolve_set_id",
    "fetch_pairs",
    "model_id_for_hf_id",
    "extracted_pair_ids_for",
    "extracted_pair_ids_via_activation",
    "insert_raw_activations",
    "insert_activations",
    "pair_id_lookup_table",
    "WISENT_APP_PROJECT",
]
