"""Frozen question encoder using the official attention-mask mean-pooling recipe."""
from pathlib import Path
import hashlib,json
import numpy as np

class FrozenQuestionEncoder:
    def __init__(self, model_path, *, revision, receipt, device="cpu", max_length=256):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.path=Path(model_path).resolve()
        provenance=json.loads(Path(receipt).read_text(encoding="utf-8-sig"))
        if provenance["repo_id"]!="sentence-transformers/all-MiniLM-L6-v2" or provenance["revision"]!=revision:
            raise ValueError("Frozen encoder source mismatch")
        project=Path(receipt).resolve().parents[2]
        for record in provenance["files"]:
            source=project/record["path"]
            if hashlib.sha256(source.read_bytes()).hexdigest()!=record["sha256"]:
                raise ValueError("Encoder asset hash mismatch")
        self.torch=torch;self.device=device;self.max_length=int(max_length)
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(self.path,local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device)
        self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)

    def encode(self,texts,batch_size=16):
        torch=self.torch;out=[]
        for start in range(0,len(texts),batch_size):
            tokens=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,
                                  max_length=self.max_length,return_tensors="pt")
            tokens={key:value.to(self.device) for key,value in tokens.items()}
            with torch.inference_mode():
                vectors=self.model(**tokens).last_hidden_state
                mask=tokens["attention_mask"].unsqueeze(-1).to(vectors.dtype)
                pooled=(vectors*mask).sum(1)/mask.sum(1).clamp(min=1)
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
            out.append(pooled.cpu().float().numpy())
        return np.concatenate(out,axis=0) if out else np.empty((0,384),dtype=np.float32)
