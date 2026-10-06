"""Native-space lexical identity for a registered observation map.

Only the current query is parsed. Additional training labels never enter this
reader, and the original full-query semantic field is left to the caller.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .native_phrase_binding import MentionRoles, decode_mentions
from .typed_referent_binding import LexicalIdentity


@torch.inference_mode()
def lexical_query_features(expressions, encoded, tokenizer, teacher, role_checkpoint, device):
    saved = torch.load(role_checkpoint, map_location='cpu', weights_only=False)
    roles = MentionRoles(**saved['architecture']).to(device).eval();roles.load_state_dict(saved['state_dict'])
    hidden = encoded.last_hidden_state.float()
    offsets = tokenizer(expressions, padding='max_length', max_length=64, truncation=True,
        return_offsets_mapping=True)['offset_mapping']
    labels = roles(hidden).cpu().numpy()
    phrases = [decode_mentions(q, o, l) for q, o, l in zip(expressions, offsets, labels)]
    strings = [m['phrase'] for row in phrases for m in row]
    pooled = []
    for start in range(0, len(strings), 24):
        batch = tokenizer(strings[start:start + 24], padding='max_length', max_length=64,
            truncation=True, return_tensors='pt').to(device)
        pooled.append(F.normalize(F.normalize(teacher(**batch).pooler_output.float(), dim=-1).half().float(), dim=-1))
    phrase_vectors = torch.cat(pooled) if pooled else hidden.new_empty((0, encoded.pooler_output.shape[-1]))
    inputs = tokenizer(expressions, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    valid = (inputs['input_ids'] != tokenizer.pad_token_id).to(device)
    full = F.normalize(F.normalize(encoded.pooler_output.float(), dim=-1).half().float(), dim=-1)
    rows = [];offset = 0
    for i, row in enumerate(phrases):
        count = len(row)
        rows.append(dict(full=full[i].cpu(), phrases=phrase_vectors[offset:offset + count].cpu(),
            phrase_tokens=torch.stack([hidden[i, m['tokens']].mean(0) for m in row]).cpu() if row else
                torch.empty((0, hidden.shape[-1])), roles=torch.tensor([m['role'] for m in row]),
            query_tokens=hidden[i, valid[i]].mean(0).cpu(), spans=row))
        offset += count
    return rows


class LexicalIdentityReader:
    def __init__(self, checkpoint, descriptors, membership, device):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = LexicalIdentity(**saved['architecture']).to(device).eval()
        self.model.load_state_dict({key[len('identity.'):]: value for key, value in saved['state_dict'].items()
            if key.startswith('identity.')})
        self.ids = np.flatnonzero(np.asarray(membership.sum(0)).ravel() > 0)
        vectors = np.array(descriptors[self.ids], dtype=np.float32, copy=True)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        self.visual = torch.from_numpy(vectors).to(device);self.size = len(descriptors);self.device = device
        self.receipt = dict(checkpoint=str(checkpoint), selected_epoch=saved['epoch'],
            supervision='same-class positives, independent NR3D/ScanEnts3D/official3D training rooms',
            identity_rule='query-only convex weights over native full-expression and target-phrase cosine',
            original_native_field_changed=False, labels_at_inference=False, source_images_at_query=False,
            relation_correction=False, no_candidate_limit=True)

    @torch.inference_mode()
    def rank(self, rows):
        result = np.full((self.size, len(rows)), -np.inf, np.float32);details = []
        for index, row in enumerate(rows):
            query = {key: row[key].to(self.device) for key in ['phrase_tokens', 'roles', 'query_tokens']}
            query.update(native=row['full'].to(self.device) @ self.visual.T,
                phrase_scores=row['phrases'].to(self.device) @ self.visual.T)
            with torch.autocast('cuda', dtype=torch.bfloat16):scores, weights = self.model(query, return_weights=True)
            result[self.ids, index] = scores.cpu().numpy()
            details.append(dict(spans=row['spans'], full_and_target_weights=weights.cpu().tolist()))
        return result, dict(self.receipt, queries=details)
