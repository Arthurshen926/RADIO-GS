"""Resolve literal query meaning before seeing any candidate image.

An optional online language prior, not benchmark aliases or verified semantics.
Generic categories must not acquire invented colour/material/shape constraints.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

INSTRUCTION = ('Explain the literal visual referent of this retrieval query in at most two short sentences: {query}. '
    'Resolve a well-known proper name to its canonical visual identity when you know it. '
    'Preserve explicitly requested attributes and whole-object versus part meaning. '
    'For generic categories do not invent a color, material, shape, brand, or instance restriction. '
    'Do not add associated nearby objects or guess an intended scene. '
    'If the name or meaning is unfamiliar or ambiguous, explicitly say it is uncertain. '
    'You have no image and must not claim any object is present.')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2)
    manifest=json.loads(Path(a.manifest).read_text());queries=list(dict.fromkeys(r['query'] for r in manifest['jobs']))
    modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True)
    processor.tokenizer.padding_side='left'
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    rows={};start=time.monotonic()
    for offset in range(0,len(queries),8):
        batch=queries[offset:offset+8]
        texts=[processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=INSTRUCTION.format(query=json.dumps(q)))])],tokenize=False,add_generation_prompt=True) for q in batch]
        inputs=processor(text=texts,padding=True,return_tensors='pt').to('cuda')
        generated=model.generate(**inputs,max_new_tokens=96,do_sample=False)
        descriptions=processor.tokenizer.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        rows.update(zip(batch,descriptions));print(offset+len(batch),'/',len(queries),flush=True)
    Path(a.output).write_text(json.dumps(dict(model=modelid,revision=model.config._commit_hash,instruction=INSTRUCTION,
        descriptions=rows,source_or_target_images_read=False,query_specific_handwritten_aliases=False,
        independently_verified_meaning=False,seconds=time.monotonic()-start),indent=2)+'\n')


if __name__=='__main__':main()
