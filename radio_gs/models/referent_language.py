"""Query-only role decomposition; no scene, photograph or annotation input."""
import json
from pathlib import Path
import subprocess
import tempfile
import time

import torch


MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
PYTHON = '/root/.cache/radio_gs/qwen3_env/bin/python'
INSTRUCTION = '''Translate the retrieval request into a JSON object with exactly these keys:
"target": a short visual noun phrase naming ONLY the object or part to retrieve,
"relations": a list of {"predicate": ..., "reference": ...},
"unresolved": a list of remaining relational or comparative phrases.
Preserve the target's explicitly stated color, material, shape, and part identity.
Do not put a nearby object, a relative location, or a comparative size into target.
Do not replace a requested part with its containing whole object.
The only supported predicates are near, far, above, below. Translate next to,
closest to, by and touching into near; farthest from and not by into far;
on top of into above and under into below. A reference is a short noun phrase
for the OTHER object, including its stated intrinsic attributes. A relation
must apply directly to the target, not to a reference object's own reference.
Put size comparisons, ordinals, room-relative directions, containment, between,
and relations without an explicit reference object in unresolved, without guessing.
Do not invent an object, attribute, or relation. You have no scene information.
Return JSON only, without a code fence or explanation.
Examples (unrelated scenes):
Request: red bottle beside a wooden box
{"target":"red bottle","relations":[{"predicate":"near","reference":"wooden box"}],"unresolved":[]}
Request: the green kettle
{"target":"green kettle","relations":[],"unresolved":[]}
Request: the tallest cup
{"target":"cup","relations":[],"unresolved":["tallest"]}
Request: handle of the yellow mug
{"target":"handle of the yellow mug","relations":[],"unresolved":[]}
Request: '''


def parse_referents(expressions, device='cuda'):
    """Use the existing Qwen3 runtime, also for a previously unseen query."""
    with tempfile.TemporaryDirectory(prefix='radio_referent_') as directory:
        root = Path(directory)
        (root/'input.json').write_text(json.dumps(expressions))
        subprocess.run([PYTHON, '-m', 'radio_gs.models.referent_language',
            '--input', str(root/'input.json'), '--output', str(root/'output.json'),
            '--device', device], check=True)
        result = json.loads((root/'output.json').read_text())
        return result['plans'], result['report']


@torch.inference_mode()
def _parse_local(expressions, device='cuda'):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    started = time.monotonic()
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(device).eval()
    plans = []
    for offset in range(0, len(expressions), 4):
        texts = [processor.apply_chat_template([dict(role='user', content=[dict(
            type='text', text=INSTRUCTION + json.dumps(expression))])],
            tokenize=False, add_generation_prompt=True)
            for expression in expressions[offset:offset + 4]]
        inputs = processor(text=texts, padding=True, return_tensors='pt').to(device)
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        outputs = processor.tokenizer.batch_decode(
            generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        # The local instruction model can still wrap JSON in Markdown fences.
        plans.extend(json.loads(value[value.index('{'):value.rindex('}') + 1]) for value in outputs)
        print('parsed query roles', len(plans), '/', len(expressions), flush=True)
    report = dict(model=MODEL, revision=model.config._commit_hash, instruction=INSTRUCTION,
        seconds=time.monotonic() - started, queries=len(expressions),
        images_read=False, map_read=False, annotations_read=False,
        handwritten_query_aliases=False, independently_verified_parse=False,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))
    del model
    torch.cuda.empty_cache()
    return plans, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    plans, report = _parse_local(json.loads(args.input.read_text()), args.device)
    args.output.write_text(json.dumps(dict(plans=plans, report=report), indent=2)+'\n')
