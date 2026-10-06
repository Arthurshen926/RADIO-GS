"""Independent source referent checks with counterbalanced answer labels.

Uses actual source RGB and the exact detector mask. Never judges target images,
never sees detector confidence, and never supplies a replacement object extent.
Three answer permutations expose label sensitivity; they are not independent
semantic witnesses or calibrated correctness probabilities.
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from scripts.research.verify_track_identity_next98 import crop

MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
CLASSES = [
    'the outlined content matches the requested identity',
    'the outlined content is a different object or contradicts a requested attribute',
    'the identity or requested attribute cannot be determined from this view or your knowledge',
]
ORDERS = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
PROMPT = ('Judge the identity of the content inside the red outline. Requested referent: {query}. '
          'Do not assume the proposed region is correct. Ignore detector quality and mask boundary precision. '
          'A partial visible surface may support an object identity; nearby or associated objects do not. '
          'Respect any specifically requested part or attribute. Do not reject a region just because part of '
          'the object is occluded. If the name is unfamiliar or the visual evidence is insufficient, choose '
          'the undetermined option.\n{choices}\nAnswer exactly one letter: A, B, or C.')
REFERENCE_PROMPT = ('The FIRST image shows the authorized reference object inside a red outline. '
    'The SECOND image shows a proposed region in another view. Judge whether the SECOND outlined content '
    'belongs to the same physical object as the reference. A partial visible surface is allowed. '
    'A nearby object, merely the same category, or a region mixing the reference with unrelated objects '
    'does not establish matching identity. Ignore small boundary errors. If correspondence is uncertain, '
    'choose the undetermined option.\n{choices}\nAnswer exactly one letter: A, B, or C.')


class LastTokenHead(torch.nn.Module):
    """Avoid allocating unused vocabulary logits for all image/prompt tokens."""
    def __init__(self, head):
        super().__init__(); self.head = head

    def forward(self, hidden):
        return self.head(hidden[:, -1:])


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-id', default=MODEL)
    p.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
    p.add_argument('--context-image', action='store_true', help='Full source ownership context plus the unchanged local crop')
    p.add_argument('--interpretations', help='Optional image-blind query interpretations')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4, help='Source candidates; three label orders each')
    a = p.parse_args(); torch.set_num_threads(2)
    manifest = json.loads(Path(a.manifest).read_text())
    interpretations = json.loads(Path(a.interpretations).read_text())['descriptions'] if a.interpretations else {}
    jobs = manifest['jobs'][a.shard::a.shards]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.model_id, local_files_only=True, min_pixels=56*56, max_pixels=448*448)
    processor.tokenizer.padding_side = 'left'
    if 'Qwen3-VL' in a.model_id:
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    else:
        model_class = Qwen2_5_VLForConditionalGeneration
    model = model_class.from_pretrained(a.model_id, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation=a.attention).cuda().eval()
    model.lm_head = LastTokenHead(model.lm_head)
    ids = [processor.tokenizer.encode(x, add_special_tokens=False)[0] for x in 'ABC']

    @lru_cache(maxsize=4)
    def evidence(path):
        return torch.load(path, map_location='cpu', weights_only=False)

    @lru_cache(maxsize=8)
    def reference_image(rgb, mask):
        values=np.asarray(Image.open(mask).convert('L')) > 0
        return crop(rgb, values)

    @lru_cache(maxsize=4)
    def source_image(path):
        im=Image.open(path).convert('RGB');im.thumbnail((672,672));return im

    def marked_context(path, mask):
        im=source_image(path)
        resized=torch.from_numpy(np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST),copy=True)).float()[None,None]
        edge=(torch.nn.functional.max_pool2d(resized,3,1,1)+torch.nn.functional.max_pool2d(-resized,3,1,1))[0,0]>.5
        rgb=np.array(im,copy=True);rgb[edge.numpy()]=[255,0,0]
        return Image.fromarray(rgb)

    start = time.monotonic(); results = []
    for offset in range(0, len(jobs), a.batch_size):
        batch = jobs[offset:offset+a.batch_size]
        images, texts = [], []
        for row in batch:
            data = evidence(row['evidence'])
            mask = data['instances'][row['query_index']]['masks'][row['instance_index']].float().numpy() > .5
            im = crop(row['source_rgb'], mask)
            context=marked_context(row['source_rgb'],mask) if a.context_image and 'reference_rgb' not in row else None
            for order in ORDERS:
                choices = '\n'.join(f'{letter}: {CLASSES[cl]}' for letter, cl in zip('ABC', order))
                is_reference='reference_rgb' in row
                instruction = (REFERENCE_PROMPT.format(choices=choices) if is_reference else
                               PROMPT.format(query=json.dumps(row['query']), choices=choices))
                if context is not None:
                    instruction=('The FIRST image is the full source view, which provides ownership and scene context. '
                        'The SECOND is a detailed crop of the SAME red-outlined region. Judge the outlined region, '
                        'not the whole surrounding object. Use the full view to identify which object a visible part belongs to.\n'
                        +instruction)
                if a.interpretations:
                    instruction += ('\nAn image-blind interpretation of the literal query is: '
                                    + interpretations[row['query']]
                                    + '\nUse this only to understand the request. It does not assert that the candidate matches. '
                                    'Do not turn typical optional appearance into a required attribute.')
                messages = [dict(role='user', content=[dict(type='image')]*(2 if is_reference or context is not None else 1)+[dict(type='text', text=instruction)])]
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                if is_reference:images.append(reference_image(row['reference_rgb'],row['reference_mask']))
                if context is not None:images.append(context)
                images.append(im)
        inputs = processor(text=texts, images=images, padding=True, return_tensors='pt').to('cuda')
        raw = model(**inputs, use_cache=False).logits[:, -1].float()
        scores = raw[:, ids].cpu().reshape(len(batch), 3, 3)
        top_tokens = processor.tokenizer.convert_ids_to_tokens(raw.argmax(-1).cpu().tolist())
        for index, row in enumerate(batch):
            canonical = torch.stack([scores[index, oi, [order.index(c) for c in range(3)]]
                                     for oi, order in enumerate(ORDERS)])
            decisions = canonical.argmax(-1).tolist()
            class_support = canonical.softmax(-1).mean(0)
            results.append(dict(key=row['key'], query=row['query'], scene=row['scene'], frame=row['frame'],
                class_logits=canonical.tolist(), decisions=decisions,
                single_veto=decisions[0] == 1, balanced_veto=all(x == 1 for x in decisions),
                class_support=class_support.tolist(), affirmative_admission=int(class_support.argmax()) == 0,
                label_sensitive=len(set(decisions)) > 1,
                raw_top_tokens=top_tokens[index*3:(index+1)*3]))
        if offset % 80 == 0:
            print(a.shard, offset+len(batch), '/', len(jobs), round(time.monotonic()-start, 1), flush=True)
    (out/f'shard_{a.shard}.json').write_text(json.dumps(dict(model=a.model_id, revision=model.config._commit_hash, attention=a.attention,
        prompt=PROMPT, reference_prompt=REFERENCE_PROMPT, classes=CLASSES, orders=ORDERS, interpretations=a.interpretations, context_image=a.context_image, manifest=a.manifest, shard=a.shard, shards=a.shards,
        candidates=len(jobs), image_forwards=3*sum(2 if 'reference_rgb' in r or a.context_image else 1 for r in jobs), target_rgb_or_gt_used=False,
        votes_are_independent=False, seconds=time.monotonic()-start, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
