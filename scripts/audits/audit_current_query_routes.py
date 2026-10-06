"""Cold full-task evaluation with passive route traces and replayable encodings.

No query or target data is used to update model state. Hooks record the actual
forward branches; the original evaluator owns inputs, rendering and scoring.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import shutil
import torch
from radio_gs.models.functional_query_encoder import FunctionalThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.query_input_identity import query_call_sha256


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--task',choices=['text','reference'],required=True)
    p.add_argument('--encoding-replay',type=Path)
    p.add_argument('--query-relative-scope',action='store_true')
    args,remaining=p.parse_known_args()
    mp=argparse.ArgumentParser(add_help=False);mp.add_argument('--output',type=Path,required=True)
    meta,_=mp.parse_known_args(remaining)
    tracepath=meta.output.with_suffix('.routes.json');bankpath=meta.output.with_suffix('.encodings.pt')
    if tracepath.exists() or bankpath.exists():raise FileExistsError(tracepath)
    snapshots=meta.output.with_suffix('.sources');snapshots.mkdir(exist_ok=False)
    frozen_implementations={}
    for f in [Path(__file__),Path('radio_gs/utils/query_input_identity.py'),Path('radio_gs/querying/query_relative_scope.py')]:
        shutil.copy2(f,snapshots/f.name);frozen_implementations[str(f)]=sha256_file(f)
    bank=[];cursor=0;traces=[];texts=[]
    replay=None
    if args.encoding_replay:
        replay=torch.load(args.encoding_replay,weights_only=False,map_location='cpu')
    class Encoder(FunctionalThreeQueryEncoder):
        def call(self,kind,fn,*a,**kw):
            nonlocal cursor
            identity=query_call_sha256(kind,a,kw,self.identity)
            if replay is None:value=fn(*a,**kw)
            else:
                row=replay[cursor];assert row['kind']==kind
                if row.get('query_input_sha256')!=identity:raise ValueError('query replay input or encoder identity changed')
                if kind=='text':assert row['texts']==list(a[0])
                value=row['encoding'].to(self.device)
            row=dict(kind=kind,query_input_sha256=identity,encoding=value.detach().cpu())
            if kind=='text':row['texts']=list(a[0]);texts.extend(a[0])
            bank.append(row);cursor+=1;return value
        def encode_text(self,*a,**kw):return self.call('text',super().encode_text,*a,**kw)
        def encode_image(self,*a,**kw):return self.call('image',super().encode_image,*a,**kw)
        def encode_prompt(self,*a,**kw):return self.call('prompt',super().encode_prompt,*a,**kw)
    evaluator=importlib.import_module('scripts.audits.'+('evaluate_frozen_field_lerf_labels' if args.task=='text' else 'evaluate_real_reference_queries'))
    original_loader=evaluator.load_element_retrieval_field
    def loader(*a,**kw):
        field,payload=original_loader(*a,**kw)
        if args.query_relative_scope:
            from radio_gs.querying.query_relative_scope import install_query_relative_scope_control
            install_query_relative_scope_control(field)
        original=field.retrieve
        def retrieve(query,modality,rows=None):
            capture={};hooks=[]
            if modality=='text' and field.grounded_transfer is not None:
                def head_hook(module,inputs,output):capture['scope_eligible']=output[1].detach().cpu().tolist()
                hooks.append(field.grounded_transfer.register_forward_hook(head_hook))
                if field.extent_benefit_admission is not None:
                    def gate_hook(module,inputs,output):capture['benefit_allowed']=output.detach().cpu().tolist()
                    hooks.append(field.extent_benefit_admission.register_forward_hook(gate_hook))
            try:result=original(query,modality,rows)
            finally:
                for h in hooks:h.remove()
            record=dict(field=str(a[0]),modality=modality,query_count=len(query),query_sha256=digest(query),support_sha256=digest(result['valid']),score_sha256=[digest(col) for col in result['scores'].T],membership_sha256=[digest(col>.5) for col in result['scores'].T],**capture)
            if modality=='text' and field.grounded_transfer is not None:
                index,accepted,similarity=field.grounded_transfer.route(query)
                record.update(concept_indices=index.tolist(),concept_accepted=accepted.tolist(),concept_similarity=similarity.tolist())
                eligible=capture.get('scope_eligible');allow=capture.get('benefit_allowed',[True]*len(query))
                record['final_scope_used']=[bool(e and b) for e,b in zip(eligible,allow)] if eligible is not None else None
            traces.append(record);return result
        field.retrieve=retrieve;return field,payload
    evaluator.NativeThreeQueryEncoder=Encoder;evaluator.load_element_retrieval_field=loader
    sys.argv=[sys.argv[0],*remaining];evaluator.main()
    protocolpath=meta.output.with_suffix('.protocol.json')
    protocol=json.loads(protocolpath.read_text())
    protocol['effective_readout']='query_relative_scope_frozen_policy_control' if args.query_relative_scope else 'retained_checkpoint'
    protocol['route_audit_implementation_sha256']=sha256_file(__file__)
    protocol['query_relative_implementation_sha256']=sha256_file('radio_gs/querying/query_relative_scope.py') if args.query_relative_scope else None
    protocolpath.write_text(json.dumps(protocol,indent=2)+'\n')
    evaluation=json.loads(meta.output.read_text());evaluation['protocol_sha256']=sha256_file(protocolpath)
    if 'protocol' in evaluation:evaluation['protocol']=protocol
    evaluation['effective_readout']=protocol['effective_readout']
    meta.output.write_text(json.dumps(evaluation,indent=2)+'\n')
    if replay is not None:assert cursor==len(replay)
    torch.save(bank,bankpath)
    implementations={str(f):sha256_file(f) for f in [Path(__file__),Path(evaluator.__file__),Path('radio_gs/models/element_retrieval_field.py'),Path('radio_gs/models/grounded_semantic_transfer.py'),Path('radio_gs/models/functional_query_encoder.py'),Path('radio_gs/querying/query_relative_scope.py')]}
    implementations.update(frozen_implementations)
    tracepath.write_text(json.dumps(dict(schema='full_query_route_audit.v1',query_relative_scope_control=args.query_relative_scope,training=False,target_data_used_for_model_updates=False,task=args.task,texts=texts,encoding_mode='recorded_replay' if replay is not None else 'fresh_online',encoding_replay_sha256=sha256_file(args.encoding_replay) if replay is not None else None,encoding_bank_sha256=sha256_file(bankpath),evaluation_sha256=sha256_file(meta.output),implementations=implementations,traces=traces),indent=2)+'\n')

if __name__=='__main__':main()
