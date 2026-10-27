"""Provider-agnostic VLM call via litellm. Swap providers with VLM_MODEL in .env."""

import base64
import json

import cv2
import litellm

from .config import ROOT, VLM_MODEL, VLM_REASONING

PROMPTS = ROOT / "omnimanip" / "prompts"


def query(name, images=(), model=VLM_MODEL, **fields):
    """Fill prompts/<name>.txt with fields, send with RGB uint8 images, and enforce
    prompts/<name>.schema.json as structured output (paper supp. A.2). Returns the parsed JSON."""
    content: list[dict] = [
        {"type": "text", "text": (PROMPTS / f"{name}.txt").read_text().format(**fields)}
    ]
    for im in images:
        png = cv2.imencode(".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1].tobytes()
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode()
                },
            }
        )
    schema = json.loads((PROMPTS / f"{name}.schema.json").read_text())
    r = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": content}],
        **({"reasoning_effort": VLM_REASONING[name]} if name in VLM_REASONING else {}),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        },
    )
    assert isinstance(r, litellm.ModelResponse)
    return json.loads(r.choices[0].message.content or "")
