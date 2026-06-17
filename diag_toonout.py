import torch
from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download

m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
msd = m.state_dict()
mkeys = list(msd.keys())
print("MODEL keys:", len(mkeys))
for k in mkeys[:8]:
    print("  M:", k, tuple(msd[k].shape))

path = hf_hub_download("joelseytre/toonout", "birefnet_finetuned_toonout.pth")
sd = torch.load(path, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    print("checkpoint has 'state_dict' wrapper; top keys:", list(sd.keys())[:6])
    sd = sd["state_dict"]
skeys = list(sd.keys())
print("CKPT keys:", len(skeys))
for k in skeys[:8]:
    try:
        print("  C:", k, tuple(sd[k].shape))
    except Exception:
        print("  C:", k, type(sd[k]))

mset, sset = set(mkeys), set(skeys)
print("exact overlap:", len(mset & sset))
# try common prefix transforms
for pref in ["module.", "model.", "bb.", "net."]:
    stripped = {k[len(pref):] if k.startswith(pref) else k for k in skeys}
    print(f"strip '{pref}': overlap={len(mset & stripped)}")
for pref in ["model.", "bb."]:
    added = {pref + k for k in skeys}
    print(f"add '{pref}': overlap={len(mset & added)}")
