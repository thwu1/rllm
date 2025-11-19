import os

import modal

image = modal.Image.from_registry("verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0-te2.2").run_commands("""git clone --recurse-submodules https://github.com/thwu1/rllm.git && cd rllm && git checkout 5ca63986561b98e811475dc3dabdcad3772141b2 && cd verl && git checkout 60d8a62 && cd .. && pip install -e verl/ && pip install -e . && pip install litellm[proxy] asgiref==3.10.0 aiosqlite""")
app = modal.App("rllm-v1")


@app.function(image=image)
def tutorial() -> list[dict]:
    os.system("pip list")
