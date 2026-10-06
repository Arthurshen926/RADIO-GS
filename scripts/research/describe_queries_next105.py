"""Text-only visual descriptions for proposal retrieval, never scene-conditioned."""
import json
from pathlib import Path
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from radio_gs.models.native_query_encoders import load_siglip_text
from torch.nn import functional as F
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 torch.set_num_threads(4);modelid='Qwen/Qwen2.5-VL-3B-Instruct';processor=AutoProcessor.from_pretrained(modelid,local_files_only=True);model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelid,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().cuda();records=[]
 for scene in ['figurines','ramen','teatime','waldo_kitchen']:
  for r in json.loads((Path('output/grounded_seeds_next88')/('lerf_'+scene)/'proposals.json').read_text()):
   prompt='Describe the visible appearance of the following requested object or object part for image retrieval. Preserve all stated modifiers, material, identity and whole/part scope. Use one concise sentence. Do not invent a scene, surrounding objects, a specific color not implied by the term, or an alternative target. If a name is unfamiliar, repeat it. Target: '+json.dumps(r['query']);template=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt)])],tokenize=False,add_generation_prompt=True);inputs=processor(text=[template],return_tensors='pt').to('cuda');tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False);description=processor.tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True);records.append(dict(scene=scene,query=r['query'],query_index=r['query_index'],description=description));print(r['query'],description,flush=True)
 del model;torch.cuda.empty_cache();field=load('output/taskwise_scope_composition_next38/lerf_ramen/field.pt');root=Path(field['root']);info=json.loads((root/'teachers.json').read_text());teacher=load(root/info['teacher_files']['siglip']['path']);textmodel,tokenizer=load_siglip_text(Path(teacher['encoder']['path']),torch.device('cuda'));phrases=[r['query']+'. '+r['description'] for r in records];inputs=tokenizer(phrases,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda');features=F.normalize(textmodel(**inputs).pooler_output.float(),dim=-1).cpu();out=Path('output/query_descriptions_next105');out.mkdir(parents=True,exist_ok=True);torch.save(dict(records=records,features=features),out/'descriptions.pt');(out/'descriptions.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
