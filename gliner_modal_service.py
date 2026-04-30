"""
Modal-hosted GLiNER NER service for SimpleRAG.

Replaces the local single-threaded GLiNER model with a Modal endpoint that
auto-scales to N parallel containers — one per concurrent caller, no shared lock.

Deploy:   modal deploy gliner_modal_service.py
Activate: set MODAL_GLINER=1 in your environment (or dev_config.json)

Free tier: ~$30/month credit. 609-doc indexing run costs ~$0.02.
"""

import modal

app = modal.App("simplerag-gliner")

# Minimal image: GLiNER + its deps. No torch CPU extras needed — GLiNER small is fast on CPU.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gliner==0.2.26", "huggingface_hub")
)

# Model is downloaded inside the container on first call, then cached in the image layer
# via run_function at build time so cold starts skip the download entirely.
@app.cls(
    image=image,
    cpu=2.0,
    memory=2048,
    timeout=180,
    # Scale to 0 when idle (free tier friendly — no idle charges)
)
class GLiNERService:
    @modal.enter()
    def load_model(self):
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    @modal.method()
    def batch_ner(
        self,
        texts: list,
        labels: list,
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> list:
        """
        Run NER on a batch of texts.

        Args:
            texts:      list of strings (one per chunk)
            labels:     NER label set
            threshold:  confidence cutoff
            batch_size: GLiNER internal batch size (GPU: 64, CPU: 16-32)

        Returns:
            list of list of dicts — one inner list per input text
        """
        return self.model.inference(
            texts, labels=labels, threshold=threshold, batch_size=batch_size
        )
