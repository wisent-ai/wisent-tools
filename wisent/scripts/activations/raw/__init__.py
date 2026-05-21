"""Raw-mode activation extraction subpackage.

See raw.extract_and_upload for the entry point. The legacy pre-aggregated
7-strategy layout lives in the parent module (extract_and_upload.py);
this subpackage targets raw_activations/<safe>/<task>/<prompt_format>/
layer_<L>_chunk_<C>.safetensors per migrate_raw.py convention.
"""
