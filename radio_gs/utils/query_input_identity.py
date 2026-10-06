"""Content identity for a diagnostic query encoding replay, not a model cache."""
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from radio_gs.utils.immutable_artifacts import sha256_file


def query_input_identity(value):
    if torch.is_tensor(value):
        tensor=value.detach().contiguous().cpu()
        return {'tensor_dtype':str(tensor.dtype),'shape':list(tensor.shape),'sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    if isinstance(value,Image.Image):
        return {'mode':value.mode,'size':list(value.size),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,Path):return {'path':str(value),'sha256':sha256_file(value)}
    if isinstance(value,(list,tuple)):return [query_input_identity(v) for v in value]
    if isinstance(value,dict):return {k:query_input_identity(v) for k,v in sorted(value.items())}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Unsupported query input identity: '+type(value).__name__)


def query_call_sha256(kind,args,kwargs,encoder_identity):
    # Prompt image paths may arrive as strings; bind their file content too.
    args=list(args)
    if kind=='prompt' and args:args[0]=Path(args[0])
    payload=query_input_identity({'kind':kind,'args':args,'kwargs':kwargs,'encoder':encoder_identity})
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
