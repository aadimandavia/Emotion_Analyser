# Edge Deployment Plan — Running the System On-Device

One of the core design goals of this system is that it should eventually run **entirely on a mobile or edge device** — no cloud, no API calls, no data leaving the user's hands.

This section covers why that matters, what stands in the way, and exactly how to get there.

---

## The Goal

An on-device version of this system should be:

- **Fast** — responses under 200ms, even on mid-range hardware
- **Lightweight** — small enough to ship inside a mobile app
- **Private** — emotional data never leaves the device

That last point matters more here than in most applications. This system handles sensitive mental and emotional information. On-device inference isn't just a performance optimization — it's a trust feature.

---

## Current System Constraints

The existing implementation was built for correctness first, not size. It currently uses:

- A TF-IDF vectorizer with 500 features
- Five separate RandomForest models (state, intensity, stress, energy, action)
- A hybrid decision engine running on top

This works well on a laptop. It does not translate cleanly to a phone or embedded device without deliberate optimization.

---

## Challenges for Edge Deployment

### 1. Model Size
Five RandomForest models, each with many decision trees, add up quickly in memory. RandomForest is not known for being compact.

### 2. Latency
Each inference request runs through a sequential pipeline — preprocessing → five model calls → decision logic. Every step adds time.

### 3. Feature Processing
TF-IDF requires storing a vocabulary at runtime. Text preprocessing (tokenization, vectorization) adds overhead before a single model even runs.

---

## Optimization Strategy

### 1. Consolidate the Models

The most impactful change: replace five separate models with one or two multi-output models.

**Options:**
- A single model predicting all outputs simultaneously
- LightGBM or XGBoost with reduced tree depth — much faster and smaller than RandomForest
- A small neural network (if targeting slightly higher-end devices)

This alone reduces both memory footprint and inference time significantly.

---

### 2. Shrink the Feature Space

Reduce TF-IDF features from **500 → ~100–200** by keeping only the highest-impact vocabulary terms. Low-variance categorical features can be dropped if ablation confirms they don't move the needle.

Less input = faster preprocessing + smaller vectorizer file on disk.

---

### 3. Compress the Models

Apply **quantization** — converting float32 weights to int8 — to reduce memory usage without retraining. This is a standard technique with well-documented tooling (ONNX, scikit-learn export pipelines, TFLite).

---

### 4. Optimize the Pipeline

| Current | Optimized |
|---|---|
| 5 sequential model calls | 1–2 parallel/combined model calls |
| TF-IDF + full vocabulary | Compressed vocabulary, fewer features |
| Float precision throughout | Quantized integer inference |

Each step compounds — a faster model running on smaller inputs with compressed weights is the target state.

---

### 5. Future: Replace TF-IDF Entirely

For more advanced deployments, lightweight embeddings (distilled sentence transformers, for example) can outperform TF-IDF on short emotional text while remaining small enough to ship on-device. This is a future improvement, not a requirement for v1.

---

## Trade-offs

No edge optimization is free. Here's what gets traded:

| Change | Gain | Cost |
|---|---|---|
| Smaller model | Speed + memory | Slight accuracy drop |
| Fewer features | Faster inference | Reduced nuance in edge cases |
| Quantization | Smaller binary | Minor precision loss |
| Fewer trees / depth | Much faster | Less interpretability |

The key principle: **optimize for good-enough accuracy at real-world speed**, not maximum accuracy at any cost.

---

## Privacy by Design

Running fully on-device means:

- No journal entries, emotional states, or personal context ever touch a server
- No API keys, no network dependency, no data retention risk
- Fully functional in airplane mode or offline environments

For a mental health application, this isn't optional — it's the right default.

---

## Target Edge Performance

| Metric | Target |
|---|---|
| Response time | < 200ms |
| Memory usage | Low (< 50MB model footprint) |
| Offline capability | Fully supported |
| Network dependency | None |

---

## Future Improvements

Once the core on-device pipeline is stable, the roadmap extends to:

- **TinyML / ONNX export** — portable, hardware-agnostic inference
- **Personalization layer** — adapt predictions using local user history without any cloud sync
- **Reinforcement learning** — improve action recommendations over time based on user feedback
- **Mobile app** — ship as a standalone Flutter or React Native application

---

## Final Thought

> The goal isn't just to make the model smaller.
>
> It's to make it **usable, fast, and trustworthy** in the hands of a real person — on their own device, on their own terms.

Efficiency and privacy aren't constraints on the product. They're part of what makes it worth building.
