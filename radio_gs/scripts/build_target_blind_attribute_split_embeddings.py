#!/usr/bin/env python3
"""Encode benchmark-disjoint compositional attribute text for one frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.scripts.build_target_blind_compositional_siglip2_embeddings import ATTRIBUTE_STRUCTURES, _GpuTextEncoder
from radio_gs.scripts.build_target_blind_siglip2_embedding_artifact import MODEL_ID, MODEL_REVISION
from radio_gs.utils.immutable_artifacts import file_record, sha256_file, write_frozen_json, write_torch_noclobber


def run(args: argparse.Namespace) -> dict:
    source=Path(args.source).resolve();manifest=Path(args.source_manifest).resolve();snapshot=Path(args.snapshot).resolve()
    if sha256_file(source)!=args.expected_source_sha256 or sha256_file(manifest)!=args.expected_manifest_sha256: raise ValueError("attribute source binding differs")
    payload=json.loads(source.read_text());records=payload.get("query_records",[])
    queries=sorted({str(row["query"]) for row in records if row.get("split")==args.split and row.get("structure") in ATTRIBUTE_STRUCTURES})
    if not queries: raise ValueError("attribute split is empty")
    encoder=_GpuTextEncoder(snapshot,torch.device(args.device));embeddings=F.normalize(encoder(queries,args.batch_size).float(),dim=-1).contiguous()
    output=Path(args.output).resolve();write_torch_noclobber(output,{"schema_version":1,"artifact_type":"target_blind_instance_attribute_text_embedding_cache","benchmark_vocabulary_opened":False,"uses_benchmark_vocabulary_for_construction":False,"split":args.split,"component_id":"counterfactual_attributes","prompt_templates":["{query}"],"queries":queries,"text_encoder":{"model_id":MODEL_ID,"revision":MODEL_REVISION},"embeddings":embeddings,"metadata":{"source":{"path":str(source),"sha256":args.expected_source_sha256},"source_manifest":{"path":str(manifest),"sha256":args.expected_manifest_sha256},"snapshot_local_files_only":True}})
    report={"status":"complete","split":args.split,"queries":len(queries),"output":file_record(output)};write_frozen_json(output.with_suffix(output.suffix+".json"),report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",default="paper/artifacts/target_blind_imagenet1k_compositional_text_bank_v2.json");p.add_argument("--source-manifest",default="paper/artifacts/target_blind_imagenet1k_compositional_text_bank_v2.manifest.json");p.add_argument("--expected-source-sha256",default="b53693a2821c29a5cc18b3ab69a9e7d9189b2c0746343b702747234ce5704b7a");p.add_argument("--expected-manifest-sha256",default="e031a6bc38242af990ecf488c96c59f667f167126d67fecae66a0b23aeb1cd96");p.add_argument("--snapshot",default="/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40");p.add_argument("--split",choices=("fit","dev","audit"),required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=32);print(json.dumps(run(p.parse_args()),indent=2,sort_keys=True))
if __name__=="__main__":main()
