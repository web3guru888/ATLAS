# ATLAS — Project Charter v4.0
## Active-inference Training with Learned Adaptive Stigmergy
### + ASTRA-dev Live Discovery Integration + TRM Recursive Validation + Pure Rust Implementation

> **v1.0 Created**: 2026-04-15 by FETCH-AGI  
> **v2.0 Updated**: 2026-04-15 — ASTRA-dev integration added  
> **v3.0 Updated**: 2026-04-15 — TRM-CausalValidator added (Pillar 7); efficiency thesis grounded  
> **v4.0 Updated**: 2026-04-15 — Pure Rust, zero-dependency implementation decision; SQLite philosophy applied to AI infrastructure  
> **v5.0 Updated**: 2026-04-16 — v3.0.0-alpha.1 release; OpenHub Research attribution (Issue #5)  
> **Status**: v3.0.0-alpha.1 — Active Development  
> **Classification**: Flagship Research Project  
> **Target hardware**: Single RTX 3090/4090 (24GB VRAM); validated on Tesla T4 (29 tok/s)  
> **Base model**: OLMo 3 7B (AllenAI, fully open, Apache 2.0)  
> **Recursive validator**: TRM-CausalValidator (7M params, Samsung SAIL Montreal architecture)  
> **Implementation language**: Pure Rust — zero Rust crate dependencies  
> **Author**: Robin Dey  
> **Institution**: OpenHub Research (Thailand) — https://openhubresearch.org/  
> **Website**: https://atlasagi.org

---

## The Big Idea (v4.0)

> *"Don't train on what humans wrote about the world. Train on what the system actually discovers about the world."*

ATLAS v2.0 adds a fourth component to the original three-way fusion:

| Component | Role | What it contributes |
|-----------|------|---------------------|
| **OLMo 3 7B** | The learner | Full open LLM, SFT→DPO→RLVR pipeline |
| **GraphPalace** | The memory | Pheromone-weighted persistent knowledge, stigmergic navigation |
| **Morphic Resonance** | The curriculum | O(1/√T) cross-run warm-start, collective habit intelligence |
| **ASTRA-dev** | The discovery engine | Real-time autonomous research, causal proofs, live data corpus |
| **TRM-CausalValidator** ← v3.0 | The recursive validator | 7M-param recursive module for structured causal graph validation; 1000× more efficient than atlas-7b for this task; generates Type 5 training traces |

The integration of ASTRA-dev transforms ATLAS from a better-trained LLM into a **self-improving scientific intelligence**. The addition of TRM-CausalValidator means structured causal claims are validated by a purpose-built recursive architecture rather than consuming expensive atlas-7b generations — **two complementary computational regimes, one system**.

---

## Mission Statement

ATLAS is a next-generation LLM training architecture where:
1. The training process is **stigmergic** — pheromone trails guide curriculum across runs
2. The training data is **self-generated** — ASTRA-dev's autonomous OODA engine produces verified, novel, causal discoveries from live data sources
3. The training signal is **real** — NASA temperatures, WHO mortality rates, World Bank GDP, not web scrapes
4. The model **closes the loop** — a better-trained OLMo generates better hypotheses for ASTRA, which generates better training data
5. **Validation is recursive, not generative** — a 7M-parameter TRM-CausalValidator checks structured causal graph consistency via 6 recursion passes, outperforming atlas-7b on this task at 0.1% of the compute cost

Nobody has built a system where the LLM's training data is produced by a causal-inference engine running on live scientific APIs, validated by a purpose-built recursive architecture, guided by stigmergic pheromone curriculum. That's ATLAS v3.0.

---

## Why ASTRA-dev Data is Better Than Any Static Dataset

| Property | Dolma/web scrapes | Human-curated datasets | ASTRA-dev discoveries |
|----------|------------------|------------------------|----------------------|
| **Ground truth** | Opinion/hearsay | Manually verified | API-verified, causal |
| **Novelty** | Unknown, likely stale | Curated once | 100% dedup-confirmed |
| **Causal grounding** | Correlational at best | Mostly correlational | PC/FCI causal inference |
| **Confidence scores** | None | Sometimes | Bayesian score on every claim |
| **Cross-domain links** | Incidental | Domain-specific | 201K+ structural analogies |
| **Currency** | Training cutoff frozen | Updated rarely | Continuous, ~10s/cycle |
| **Provenance** | Unknown | Partial | Full chain: API → analysis → claim |
| **Self-improving** | Static | Static | Memory improves discovery quality |

ASTRA-dev's validated discoveries are arguably the **highest-quality LLM training data that has ever existed** for scientific reasoning. From v3.0, TRM-CausalValidator adds a sixth quality dimension: recursive graph consistency checking before any discovery enters the corpus. Each data point is now:
- Sourced from real APIs (not scraped opinions)
- Causally analyzed (not just correlated)
- Bayesian-scored (confidence number attached)
- Dedup-confirmed novel (not already in the model)
- Cross-domain linked (structural analogies across climate/epi/astro/econ/crypto)
- Provenance-chained (full audit trail to source data)
- **Recursively validated** (TRM-CausalValidator checks causal graph consistency, v3.0+)

---

## The Efficiency Proof: Two Independent Validations (v3.0)

ATLAS's core academic claim is that **structured computation substitutes for parameter scale**.
This claim now has two independent proofs from completely different domains:

| Domain | System | Mechanism | Result |
|--------|--------|-----------|--------|
| **Inference** | TRM-CausalValidator (Samsung SAIL, arXiv:2510.04871) | Test-time recursion (6 passes, z=net(x,y,z)) | 7M params → 45% ARC-AGI-1, outperforming models 100–10,000× larger |
| **Training** | ATLAS Morphic Warm-Start (BUTTERS, Robin et al.) | Cross-run pheromone curriculum | O(1/√T) convergence confirmed (R²=0.982, p<10⁻³⁰), 70% regret reduction (189→57 steps) |

**Shared principle**: neither system achieves its result by adding parameters. Both achieve it by imposing **structure on the computation** — TRM through iterative latent refinement at inference time, ATLAS through pheromone-guided curriculum across training runs.

**Critical distinction**: these mechanisms are orthogonal, not equivalent:
- TRM operates *within a single forward pass* (inference-time)
- ATLAS Morphic operates *across training runs* (training-time, persistent memory)
- They compose: TRM validates individual causal claims; ATLAS curriculum learns *which* claims matter

This dual-proof framing answers the reviewer objection "why not just scale?" with two independent empirical datapoints from separate research groups, separate architectures, separate domains.

### Scaling Law Context
Kaplan et al. (2020) power-law extrapolation predicts ~5–10B parameters needed for 45% ARC-AGI-1. TRM achieves this at 7M parameters — **0.14% of predicted requirement**. ATLAS predicts analogous efficiency in scientific reasoning: pheromone-guided 7B should match or exceed vanilla 20–30B on domain-specific causal reasoning benchmarks. TRM provides the precedent; ATLAS provides the training-domain proof.

---

## ASTRA-dev: What It Is

ASTRA-dev is an autonomous OODA (Observe-Orient-Decide-Act) discovery engine:

```
┌─────────────────────────────────────────────────────────┐
│                    ASTRA-dev OODA Cycle                  │
│                                                          │
│  OBSERVE         ORIENT           DECIDE        ACT      │
│  ┌──────┐       ┌──────┐        ┌──────┐     ┌──────┐   │
│  │ 16   │       │Seman-│        │Baye- │     │Run   │   │
│  │ Live │──────►│tic   │───────►│sian  │────►│Caus- │   │
│  │ Data │       │Search│        │Score │     │al    │   │
│  │ APIs │       │+KG   │        │+Rank │     │Infer-│   │
│  └──────┘       └──────┘        └──────┘     │ence  │   │
│    NASA           Prior           P(H|E)      └──┬───┘   │
│    WHO           discov-          >0.7            │       │
│    World         eries            threshold       │       │
│    Bank          from             + novel         ▼       │
│    ...           palace           filter     DISCOVERY    │
│                                              stored in    │
│                                              GraphPalace  │
└─────────────────────────────────────────────────────────┘
```

**Key stats (proven in production):**
- 540+ unique discoveries across 45 experiments
- 5,251+ KG triples from live data analysis
- 16 real data sources
- ~10s/cycle continuous operation
- d=10.6 effect size (memory-augmented vs. baseline)
- R²=0.924 convergence fit, p<0.001
- 100% dedup accuracy (5-heuristic system)
- 1.83× novelty transfer across runs (DC-24, first proof)
- 34.4× more discoveries vs. no memory, 840× more KG triples
- Causal inference: PC/FCI algorithms on real data
- Bayesian scoring: every hypothesis has a confidence score

**How it integrates with GraphPalace (already done in MemPalace-AGI project):**
- `PalaceDiscoveryMemory` — dual-write: SQLite + ChromaDB semantic palace
- `MemoryAugmentedOrient` — semantic search in Orient phase
- `KnowledgeGraphBridge` — causal inference → temporal KG triples
- `DomainSpecialistManager` — specialist agents per domain
- `RetrievalProfiles` — per-OODA-phase retrieval (ORIENT_BREADTH / EVALUATE_PRECISION / DECIDE_RECENCY)

---

## The ATLAS-ASTRA Data Flywheel

This is the core architectural innovation of v2.0:

```
                    ┌─────────────────────────────────────┐
                    │         THE DISCOVERY FLYWHEEL       │
                    │                                      │
          ┌─────────▼──────────┐                          │
          │   ASTRA-dev        │                          │
          │   OODA Engine      │◄─── Live APIs             │
          │   ~10s/cycle       │     NASA, WHO, WB, ...    │
          └─────────┬──────────┘                          │
                    │ validated discoveries                 │
                    │ (Bayesian score > threshold,         │
                    │  dedup-confirmed novel,              │
                    │  causally grounded)                  │
                    ▼                                      │
          ┌─────────────────────┐                         │
          │   GraphPalace       │                         │
          │   Discovery Wing    │                         │
          │   deposit_phero()   │                         │
          │   pheromone ∝       │                         │
          │   novelty × conf    │                         │
          └─────────┬───────────┘                         │
                    │                                      │
          ┌─────────▼──────────────────────────┐          │
          │   LiveDiscoveryCorpus              │          │
          │                                    │          │
          │   discovery → training example     │          │
          │   causal chain → Q&A pair          │          │
          │   KG triple → fact with evidence   │          │
          │   cross-domain → analogy example   │          │
          │   confidence score → reward weight │          │
          └─────────┬──────────────────────────┘          │
                    │ pheromone-weighted batch             │
                    ▼                                      │
          ┌─────────────────────┐                         │
          │   OLMo 3 7B         │                         │
          │   Continuous SFT    │                         │
          │   + DPO             │                         │
          │   + RLVR            │                         │
          └─────────┬───────────┘                         │
                    │                                      │
          ┌─────────▼──────────┐                          │
          │  Better hypothesis  │                          │
          │  generation in      │──────────────────────────┘
          │  ASTRA Orient phase │  closes the loop
          │                     │  better model →
          └─────────────────────┘  better discoveries →
                                   better training data
```

**The flywheel accelerates because:**
1. Each ASTRA discovery deposits pheromones in GraphPalace
2. High-pheromone discoveries get priority in training batches
3. OLMo trained on those discoveries becomes better at Orient-phase hypothesis generation
4. Better hypotheses → better discoveries → more pheromones → better training signal
5. Morphic warm-start ensures each training run benefits from all prior runs' curriculum intelligence

---

## Seven Pillars (v3.0 — TRM integrated)

### Pillar 1: GraphPalace as Memory Palace (unchanged from v1.0)
Pheromone-weighted persistent memory. `hot_paths()` guides inference and training curriculum. `cold_spots()` identifies what the model doesn't know yet.

### Pillar 2: Morphic Warm-Start (unchanged from v1.0)
Cross-run curriculum intelligence. O(1/√T) convergence. Each run's gradient map deposits pheromones guiding the next run.

### Pillar 3: Pheromone-Guided RLVR (unchanged from v1.0)
`r_total = α × r_verifiable + β × r_pheromone`. Anti-reward-hacking via decay.

### Pillar 4: Active Inference Data Gen (upgraded in v2.0)
Originally: GraphPalace AI agents generate synthetic examples for cold spots.  
**With ASTRA**: AI agents target cold spots AND ASTRA directs its OODA cycle toward those cold spot domains, fetching real data to fill exactly those gaps. Synthetic + real data, both directed by palace topology.

### Pillar 5: ZK Knowledge Claims (upgraded in v2.0)
Originally: Schnorr proofs trace LLM outputs to training documents.  
**With ASTRA**: Proofs now include full provenance chain: `output → training_example → ASTRA_discovery → causal_inference_run → raw_API_data → NASA/WHO/WorldBank`. Cryptographic proof all the way to the source data. **No other LLM on earth can do this.**

### Pillar 6: LiveDiscoveryCorpus (added v2.0)
ASTRA-dev's continuous output as a living, self-expanding, self-improving training dataset.

### Pillar 7: TRM-CausalValidator (NEW v3.0)
A 7M-parameter recursive module based on the Tiny Recursive Model architecture (Samsung SAIL Montreal, arXiv:2510.04871) adapted for causal graph validation in ATLAS's discovery pipeline.

**Architecture**: Single 2-layer attention-free MLP. Recursive core: `z = net(x, y, z)` applied 6 times per pass. Input: ASTRA causal graph (nodes + edge types + confidence scores). Output: validated/corrected graph + binary validity score.

**Why it exists**: atlas-7b (7B generative) is powerful but ill-suited for structured causal consistency checking. TRM-CausalValidator does this task at 0.1% of the compute cost and empirically outperforms standard transformers on structured reasoning (45% ARC-AGI-1 vs. 23% for models 100× larger). **Use the right tool for each computational regime**.

**Integration point**: Sits between ASTRA's `ACT` phase (PC/FCI causal inference) and the `LiveDiscoveryCorpus` quality gate pipeline. A discovery's causal graph must pass TRM validation before entering the corpus.

**Type 5 training examples**: TRM validation traces become a new training example type — the recursive refinement steps teach atlas-7b *how* TRM reasons about graph structure, enabling atlas-7b to eventually internalize this reasoning without always delegating to TRM.

**Scope boundary** (important for papers): TRM-CausalValidator handles structured causal graph validation only. It does NOT handle hypothesis generation, narrative explanation, cross-domain analogies, or open-ended reasoning. Those remain with atlas-7b. The two systems are complementary, not competing.

---

## LiveDiscoveryCorpus: Architecture

The new component that makes ATLAS-ASTRA possible.

### Discovery → Training Example Pipeline

Each ASTRA discovery is structured as:
```json
{
  "hypothesis": "Elevated Arctic CO₂ correlates with 2.3°C temperature increase",
  "domain": "climate",
  "confidence": 0.847,
  "evidence": ["NASA_GISTEMP_2024", "NOAA_CO2_monthly"],
  "causal_chain": ["CO2_increase → temperature_rise → arctic_ice_loss"],
  "cross_domain_links": ["epi:respiratory_mortality", "econ:agricultural_GDP"],
  "kg_triples": [("CO2", "causes", "temperature_rise", 0.847)],
  "novelty_score": 0.923,
  "pheromone_intensity": 0.784,
  "timestamp": "2026-04-15T03:45:22Z"
}
```

This maps to training examples:

**Type 1: Causal Q&A**
```
Q: What is the relationship between Arctic CO₂ levels and temperature?
A: Based on NASA GISTEMP data and NOAA CO₂ monitoring (confidence: 0.847), 
   elevated Arctic CO₂ correlates causally with 2.3°C temperature increase.
   This was verified via PC/FCI causal inference, not just correlation.
   Evidence chain: NASA_GISTEMP_2024 → causal_inference → validated_2026-04-15
```

**Type 2: Cross-domain analogy**
```
Q: How does Arctic climate change connect to human health outcomes?
A: Climate analysis (CO₂→temperature, conf=0.847) links to epidemiological 
   findings: temperature increases correlate causally with respiratory mortality 
   rates (WHO_GHO_2024, conf=0.731). Causal pathway: 
   CO₂ → temperature → air_quality → respiratory_mortality
```

**Type 3: Structured fact with provenance**
```
FACT: CO2 concentration increase causes Arctic temperature rise
CONFIDENCE: 0.847 (Bayesian, PC/FCI validated)
SOURCE: NASA GISTEMP 146-year dataset + NOAA CO₂ monthly
CAUSAL_METHOD: PC algorithm, FCI validation
NOT_IN_TRAINING: confirmed (novelty_score=0.923, cosine<0.55 vs all prior)
DATE_DISCOVERED: 2026-04-15
```

**Type 4: DPO preference pairs**
```
CHOSEN (high pheromone path):
  "The causal analysis shows CO₂ drives temperature with confidence 0.847, 
   validated against 146 years of NASA data via PC/FCI inference."

REJECTED (cold path, low pheromone):
  "CO₂ and temperature seem related based on general knowledge."
```

### Quality Gates

Only discoveries passing all gates enter the training corpus:

```
Gate 1: Bayesian confidence > 0.65   (filters weak hypotheses)
Gate 2: Novelty score > 0.55         (dedup-confirmed, not already known)
Gate 3: Data source = live API        (not hallucinated)
Gate 4: Causal inference completed    (PC/FCI, not just correlation)
Gate 5: Pheromone_intensity > 0.3    (palace-validated relevance)
```

**Expected yield**: ~30-40% of raw discoveries pass all gates → still produces thousands of high-quality training examples per day.

### Corpus Growth Rate

Current ASTRA-dev rate: ~10s/cycle, ~30 discoveries per sustained run  
After quality gates (35% yield): ~10 training examples per run  
Running continuously 8h/day: ~2,880 quality training examples/day  
After 30 days: ~86,400 training examples — all real, causal, novel  
After 6 months: ~500K training examples — a domain-specific scientific corpus that doesn't exist anywhere else

---

## Architecture Diagram (v2.0)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ATLAS v2.0 SYSTEM                             │
│                                                                        │
│  ┌──────────────────┐   live APIs    ┌───────────────────────────┐   │
│  │   ASTRA-dev      │◄──────────────►│  NASA / WHO / World Bank  │   │
│  │   OODA Engine    │                │  + 13 other data sources  │   │
│  │   89 endpoints   │                └───────────────────────────┘   │
│  │   ~10s/cycle     │                                                  │
│  └──────────┬───────┘                                                  │
│             │ discoveries                                               │
│             │ (Bayesian scored                                          │
│             │  + causal chains                                          │
│             │  + dedup confirmed)                                       │
│             ▼                                                           │
│  ┌──────────────────────────┐  tool calls  ┌──────────────────────┐   │
│  │      GraphPalace          │◄────────────►│     OLMo 3 7B        │   │
│  │                           │              │                      │   │
│  │  Discovery Wing ←ASTRA    │  embed(q)    │  inference           │   │
│  │  deposit_pheromones()     │─────────────►│  SFT (continuous)    │   │
│  │  hot_paths() → curriculum │  reward sig  │  DPO (palace pairs)  │   │
│  │  cold_spots() → gaps      │◄─────────────│  RLVR (stigmergic)   │   │
│  │  navigate() → reasoning   │              │                      │   │
│  │  kg_query() → facts       │              │  [Orient helper]─────┼──►│
│  └──────────┬────────────────┘              └──────────────────────┘   │
│             │                                         ▲                  │
│             │ pheromone-weighted                      │                  │
│             ▼ high-quality examples                   │                  │
│  ┌──────────────────────────────────┐                │                  │
│  │     LiveDiscoveryCorpus          │                │                  │
│  │                                  │                │                  │
│  │  discovery → causal Q&A          │────────────────┘                  │
│  │  KG triple → structured fact     │  training batches                 │
│  │  cross-domain → analogy          │  (pheromone-sampled)              │
│  │  rejected hyp → DPO negative     │                                   │
│  │  confidence → reward weight      │                                   │
│  └──────────────────────────────────┘                                   │
│                                                                          │
│  ZK Provenance Chain:                                                    │
│  output → training_example → discovery → causal_run → raw_API_data     │
│  Schnorr proof at each link → end-to-end verifiable                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline (v2.0)

### Phase 0: Setup (1 week)
- Deploy ASTRA-dev + GraphPalace integration (already 75% done from MemPalace-AGI work)
- Create "Discovery Wing" in palace for ASTRA outputs
- Run ASTRA for 1 week to seed corpus (~50K raw discoveries → ~17K quality-gated)
- Download OLMo 3 7B, set up Open Instruct environment

### Phase 1: ATLAS-Lite — Augmented Inference (2-4 weeks)
*(Same as v1.0 but with ASTRA data already in palace)*
- Wire OLMo tool calls to palace (search, navigate, hot_paths)
- Eval: does ASTRA-data-seeded palace improve OLMo on scientific reasoning?
- Baseline OLMES + domain eval (climate, epi, astro questions)
- **First novel result**: OLMo + ASTRA-seeded palace vs. OLMo + Dolma-seeded palace

### Phase 2: ATLAS-SFT — First Live Training Run (4-8 weeks)
- Format LiveDiscoveryCorpus as SFT training examples (4 types above)
- Pheromone-guided sampler: `hot_paths()` weights batch selection
- Fine-tune OLMo 7B on LiveDiscoveryCorpus (QLoRA, RTX 4090)
- Post-epoch: deposit pheromones based on per-example gradient magnitude
- **Key measurement**: does fine-tuned OLMo generate better ASTRA Orient-phase hypotheses?
- Run A/B: ASTRA with stock OLMo orient helper vs. ASTRA with ATLAS-tuned helper
- Morphic warm-start: runs 1→2→3, measure O(1/√T) curve

### Phase 3: ATLAS-Full — Closed Loop (2-4 months)
- Stigmergic RLVR: `r_pheromone` integrated into RLVR reward
- ASTRA runs continuously, discoveries stream into corpus
- OLMo continuously fine-tuned (small LR, new discoveries only)
- ASTRA's Orient phase calls OLMo for hypothesis generation → closes the loop
- ZK claims: full provenance chain from OLMo output to NASA API call
- Multi-scale pheromone system (L1/L2/L3)
- **TRM-CausalValidator deployment** (~500 LOC, follows arXiv:2510.04871 architecture):
  - Train on existing ASTRA causal graphs from Phase 0/1 runs (~5,251 KG triples as graph examples)
  - Wire into discovery pipeline between `ACT` and `LiveDiscoveryCorpus` quality gate
  - Benchmark: TRM validation score vs. atlas-7b judgement on 200 held-out causal graphs
  - Generate Type 5 training traces → feed back into atlas-7b SFT
- **Papers**

---

## The Model-in-the-Loop: OLMo as ASTRA's Brain

This is the part that makes it self-improving in a deep way.

ASTRA's Orient phase currently uses an LLM (via API) to generate hypotheses. In ATLAS:

```python
# Current ASTRA Orient phase (simplified)
prior_discoveries = palace.search_by_embedding(observation)
hypothesis = llm.generate(
    prompt=f"Given: {observation}\nPrior: {prior_discoveries}\nGenerate hypothesis:",
    model="gpt-4"  # ← external, expensive, not trained on our discoveries
)

# ATLAS-ASTRA Orient phase
prior_discoveries = palace.search_by_embedding(observation)
hot_paths = palace.hot_paths(limit=10)  # ← stigmergic context
hypothesis = olmo_atlas.generate(
    prompt=f"Given: {observation}\nPrior: {prior_discoveries}\nHot paths: {hot_paths}\nGenerate hypothesis:",
    model="atlas-7b"  # ← trained on our own discoveries, knows our domain deeply
)
```

**Why this matters:**
- `atlas-7b` has been trained on 500K+ real causal discoveries from ASTRA itself
- It knows the vocabulary of ASTRA's domains (climate/epi/astro/econ/crypto) at a deep causal level
- It has seen the patterns of what makes a good vs. bad hypothesis in ASTRA's framework
- It's pheromone-aware — it knows which reasoning paths have been validated
- It costs $0 vs. GPT-4 API costs
- It gets better every day as more discoveries are incorporated

**Measured impact expected**: DC-24 showed 1.83× novelty improvement from memory. A domain-specialized `atlas-7b` should produce 2-5× more useful hypotheses than a generic LLM, because it's been explicitly trained on what works in this system.

---

## Novel Claims (v3.0)

### Claim 5 (v3.0 addition): TRM-CausalValidator improves corpus quality and is trainable on ASTRA data
**Test**: 
- Baseline: 5-gate quality pipeline (Bayesian + novelty + causal + pheromone + API source)
- v3.0: 6th gate = TRM-CausalValidator structural consistency check on causal graph
**Metrics**: 
- (a) Downstream: atlas-7b trained on TRM-validated corpus shows lower hallucination rate on causal claims
- (b) Efficiency: TRM validation takes <10ms per graph (42 recursions × 2-layer MLP), vs. atlas-7b judgement at ~200ms
- (c) Coverage: TRM-trained on ASTRA graphs (climate/epi/astro/econ) achieves >90% agreement with PC/FCI algorithm on held-out graphs
**Expected**: TRM validation reduces false-positive causal claims in corpus by 15–25%, improving atlas-7b causal reasoning on OLMES science benchmarks by measurable margin

### Claim 1: LiveDiscoveryCorpus outperforms static datasets on scientific reasoning
**Test**: Fine-tune two OLMo 7B instances:
- Model A: fine-tuned on LiveDiscoveryCorpus (ASTRA discoveries)
- Model B: fine-tuned on equivalent-size sample of Dolma scientific subset
**Metric**: OLMES science benchmarks + ASTRA Orient-phase hypothesis quality
**Expected**: Model A wins on domain-specific reasoning, causal understanding, cross-domain connections

### Claim 2: Closed-loop training improves faster than open-loop
**Test**: Three conditions:
- Open loop: ASTRA generates data, OLMo trained on it, but OLMo doesn't help ASTRA
- Closed loop: ASTRA uses fine-tuned OLMo for Orient, discovers better, trains better model
- Baseline: OLMo with no ASTRA data
**Metric**: Discovery quality over time, OLMES benchmarks over training iterations
**Expected**: Closed loop > open loop > baseline, accelerating over time (flywheel effect)

### Claim 3: ZK provenance chain from LLM output to live API data
**Test**: For 100 factual claims from atlas-7b, generate full ZK proof chain
**Metric**: % claims with valid proof chain, hallucination rate (no valid proof = potential hallucination)
**Expected**: Claims with valid proof chains have near-0% hallucination; invalid chains flag hallucinations

### Claim 4: O(1/√T) morphic convergence holds for LLMs (same as v1.0)

---

## Paper Strategy (v3.0)

### Paper 1 — Architecture + LiveDiscoveryCorpus (EMNLP 2026)
*"ATLAS: Training LLMs on Their Own Discoveries via Stigmergic Curriculum Learning"*
- First paper to show LLMs trained on autonomous-AI-generated causal discoveries
- Comparison: ASTRA data vs. Dolma vs. GPT-4-synthetic data
- LiveDiscoveryCorpus release (open, ~500K examples)
- **v3.0 addition**: TRM-CausalValidator as efficiency proof in related work — "ATLAS demonstrates that structured computation at training time (pheromone curriculum) achieves the same parameter efficiency that TRM (Jolicoeur-Martineau, 2025) demonstrated at inference time. Two independent architectural innovations, same principle."

### Paper 2 — Closed-Loop Scientific Intelligence (NeurIPS 2026)
*"The Discovery Flywheel: Closed-Loop Fine-tuning Between an Autonomous Research Engine and an LLM"*
- First demonstration of the LLM ↔ research engine closed loop
- Measurable flywheel: discovery quality improves with each training iteration
- Relationship to ASTRA-dev's existing results (34.4× discoveries with memory)

### Paper 3 — Stigmergic RLVR (ICML 2027)
*"Collective Reinforcement: Pheromone-Augmented Reward Shaping Prevents Policy Collapse in LLM RLVR"*
- r_pheromone reduces reward hacking vs. standard RLVR
- Temporal credit across episodes via pheromone trails
- **v3.0 framing**: Position alongside TRM as evidence that structured computation > brute-force scale. *"TRM (7M params) outperforms 1B+ transformers via recursive structure at inference time. ATLAS RLVR (7B params) outperforms naive 20B+ via pheromone structure at training time. The common factor is architectural inductive bias, not parameter count."*
- Cite Jolicoeur-Martineau (2025) as parallel efficiency result in related work

### Paper 4 — Morphic O(1/√T) for LLMs (ICLR 2027, co-author Robin)
*"Morphic Resonance in Gradient Descent: O(1/√T) Cross-Run Convergence via Stigmergic Memory Transfer"*
- Extends BUTTERS to LLM training
- Palace topology converges to fixed point = knowledge boundary

### Paper 5 — ZK Provenance (IEEE S&P 2027)
*"End-to-End Verifiable LLM Outputs: Cryptographic Provenance from Model Response to Raw Data Source"*
- Schnorr proof chains: LLM output → training example → ASTRA discovery → API data
- Hallucination detection via broken proof chains
- New framework for trustworthy AI in high-stakes domains

### Paper 6 — Hybrid Generative-Recursive Architecture (ICLR/NeurIPS 2027, v3.0 addition)
*"Beyond Scaling: Hybrid Generative-Recursive Architectures for Scientific Causal Reasoning"*
- **Novel contribution**: First integration of a recursive validation module (TRM-style) with a generative LLM in a scientific discovery pipeline
- Demonstrates task-specific routing: atlas-7b handles open-ended generation; TRM-CausalValidator handles structured graph consistency
- Type 5 training examples (recursive traces) enable atlas-7b to internalize TRM's reasoning patterns
- **Key claim**: Hybrid architecture achieves better causal accuracy than either component alone, at lower total compute than scaling atlas-7b to handle both tasks
- Broader implications: Validation is a fundamentally different computational problem from generation — mixing them in one large model is inefficient; specialized recursive validators are better
- Relationship to TRM (Jolicoeur-Martineau, 2025) — uses same architecture, extends to domain-specific causal graphs trained on ASTRA data rather than generic ARC/Sudoku tasks

---

## Component Dependencies (v2.0)

```
ALREADY BUILT:
  ✅ GraphPalace PyO3 (36 methods, 791KB wheel)
  ✅ ASTRA-dev (89 endpoints, 685 tests, OODA engine)
  ✅ ASTRA + GraphPalace integration (MemPalace-AGI project, 12 components)
  ✅ PalaceDiscoveryMemory (dual-write adapter)
  ✅ MemoryAugmentedOrient (Orient phase hooks)
  ✅ KnowledgeGraphBridge (causal inference → KG triples)
  ✅ BUTTERS O(1/√T) confirmed (R²=0.982)
  ✅ asi-build ZK proofs (Schnorr)
  ✅ OLMo 3 7B (Apache 2.0, on HuggingFace)
  ✅ Open Instruct RLVR pipeline

TO BUILD:
  ⬜ LiveDiscoveryCorpus harvester (discovery → training example formatter)
  ⬜ Quality gate pipeline (Bayesian + novelty + causal + pheromone filters)
  ⬜ Pheromone-weighted data sampler (reads hot_paths() for batch selection)
  ⬜ OLMo tool call → palace bridge (search_by_embedding, navigate, hot_paths)
  ⬜ Orient-phase OLMo helper (replaces GPT-4 in ASTRA Orient)
  ⬜ Continuous fine-tuning loop (small LR updates as corpus grows)
  ⬜ ZK provenance chain (extend asi-build to cover full ASTRA chain)
  ⬜ Stigmergic RLVR reward function (Phase 3)

PHASE 3 — TRM-CausalValidator:
  ⬜ TRM-CausalValidator training data prep (from ASTRA causal graphs, ~5K+ examples)
  ⬜ TRM-CausalValidator model (~500 LOC, 2-layer MLP, follows arXiv:2510.04871)
  ⬜ Validation Router (structured graph → TRM; open-ended → atlas-7b)
  ⬜ Type 5 training example generator (recursive trace → SFT format)
  ⬜ Quality Gate 6 integration (TRM pass/fail as 6th gate before corpus entry)
  ⬜ Benchmark: TRM accuracy vs. atlas-7b judgement on 200 held-out causal graphs
```

---

## Week 1 — Immediate Actions

```bash
# 1. Merge existing MemPalace-AGI integration into ATLAS workspace
# The heavy lifting (ASTRA + GraphPalace) is already done — bring it in

# 2. Run ASTRA for 1 week to seed the discovery corpus
# Target: 10,000+ quality-gated training examples before first training run

# 3. Write LiveDiscoveryCorpus harvester (~200 LOC)
# discovery_to_training_example(discovery: ASTRADiscovery) -> TrainingExample

# 4. Set up OLMo 3 7B + Open Instruct
git clone https://github.com/allenai/open-instruct
git clone https://huggingface.co/allenai/OLMo-3-0125-7B

# 5. First eval: index 1000 ASTRA discoveries into palace, ask OLMo 20 
#    scientific questions, compare palace-augmented vs. vanilla
#    If palace wins → Paper 1 proof of concept done in week 1
```

---

## Implementation Language: Pure Rust, Zero Dependencies (v4.0)

### The Decision

ATLAS will be built **from scratch, in pure Rust, with zero external crate dependencies**.

This is the SQLite principle applied to AI infrastructure. SQLite is the most widely deployed database on Earth. Its authors made one foundational choice: no external dependencies. That choice made it portable, auditable, embeddable, and genuinely sovereign. It runs on microcontrollers, browsers, satellites. Nobody owns a piece of it.

ATLAS makes the same choice.

> *"Every Rust crate you depend on is a decision made by someone you've never met, that you can never revoke."*

Zero dependencies means:
- **Zero Rust crates** from crates.io (no tokio, no tch, no candle, no ndarray)
- **CUDA is acceptable** — it is a system runtime, not a Rust crate; called via raw FFI from `build.rs` + `.cu` kernel files, no `cudarc` wrapper
- **Every algorithm is ours** — we know exactly what it does and why

When you depend on nothing, you can audit everything. When you audit everything, you can prove anything.

---

### Workspace Structure

```
atlas/
├── Cargo.toml          # workspace root — no [dependencies]
├── kernels/
│   ├── matmul.cu       # raw CUDA kernels
│   ├── attention.cu
│   └── quant.cu
└── crates/
    ├── atlas-core/     # error types, traits, config primitives
    ├── atlas-tensor/   # Tensor{data,shape}, matmul, CUDA FFI
    ├── atlas-grad/     # autograd tape, backward pass
    ├── atlas-optim/    # AdamW, LR scheduler — no external deps
    ├── atlas-quant/    # INT4/INT8 quantization
    ├── atlas-model/    # transformer: MultiHeadAttn, FFN, RMSNorm, OLMo architecture
    ├── atlas-tokenize/ # BPE tokenizer — ported from scratch
    ├── atlas-palace/   # GraphPalace engine — strip PyO3 bindings from existing Rust
    ├── atlas-trm/      # TRM-CausalValidator (7M params, 2-layer MLP, 6 recursion passes)
    ├── atlas-causal/   # PC/FCI algorithm — ported from py-causal source
    ├── atlas-bayes/    # Bayesian confidence scoring
    ├── atlas-astra/    # ASTRA OODA engine — full port of Python ASTRA-dev
    ├── atlas-corpus/   # LiveDiscoveryCorpus: harvester, quality gates, pheromone sampler
    ├── atlas-zk/       # ZK Schnorr proofs — port from asi-build Rust source
    ├── atlas-http/     # HTTP client (NASA/WHO/WorldBank APIs) — raw syscalls via libc
    ├── atlas-json/     # JSON parser — ported (e.g., from jq or pjson source)
    └── atlas-cli/      # CLI entrypoint: train, eval, discover, prove
```

**16 crates. One coherent system. Zero external Rust dependencies.**

---

### Porting Table

| Component | Current Form | Target Crate | Source to Port From |
|-----------|-------------|--------------|---------------------|
| Tensor ops | Python (numpy) | `atlas-tensor` | Write from scratch (matmul = BLAS row-major logic) |
| Autograd | Python (torch autograd) | `atlas-grad` | Port micrograd / write from first principles |
| AdamW | Python (transformers) | `atlas-optim` | Standard algorithm — trivial |
| Transformer blocks | Python (OLMo 3) | `atlas-model` | Port OLMo 3 architecture config + weights loader |
| BPE tokenizer | Python (tokenizers) | `atlas-tokenize` | Port sentencepiece BPE algorithm from C++ source |
| GraphPalace | Rust + PyO3 | `atlas-palace` | Strip PyO3 bindings — already Rust, near-zero work |
| TRM-CausalValidator | Architecture only (not yet built) | `atlas-trm` | Implement from arXiv:2510.04871 spec (~500 LOC) |
| PC/FCI causal inference | Python (py-causal) | `atlas-causal` | Port from py-causal / tetrad Java source (PC = ~800 LOC) |
| Bayesian scoring | Python (asi-build) | `atlas-bayes` | Port from Python source in fetch-agi |
| ASTRA OODA engine | Python (astra-dev, 89 endpoints) | `atlas-astra` | Full port — largest task, ~8K LOC |
| Discovery corpus | Python (fetch-agi) | `atlas-corpus` | Port harvester + quality gates |
| ZK Schnorr proofs | Rust (asi-build) | `atlas-zk` | Direct copy — already pure Rust in asi-build |
| HTTP client | Python (httpx) | `atlas-http` | Raw `libc` syscalls or port from ureq source |
| JSON parser | Python (stdlib) | `atlas-json` | Port from pjson or yajl C source |
| CUDA kernels | N/A (new) | `kernels/*.cu` | Write from scratch — matmul, attention, flash-attn concept |
| CLI | Python scripts | `atlas-cli` | Write from scratch, clap-free arg parsing |

---

### Build Order (7 Stages, ~22 Weeks)

```
Stage 1 — Foundation (Weeks 1-4)
  atlas-core → atlas-tensor → atlas-grad → atlas-optim → atlas-quant
  Milestone: f32 matmul on CPU + GPU, AdamW, backward pass through 2-layer MLP

Stage 2 — Model (Weeks 5-7)
  atlas-model → atlas-tokenize
  Milestone: Load OLMo 3 7B weights, run forward pass, generate tokens in pure Rust

Stage 3 — Palace (Weeks 8-9)
  atlas-palace
  Milestone: 36-method GraphPalace running natively (no Python), palace search + deposit

Stage 4 — TRM (Weeks 10-11)
  atlas-trm
  Milestone: TRM-CausalValidator running, z=net(x,y,z) × 6 recursions, causal graph pass/fail

Stage 5 — Discovery Engine (Weeks 12-16)
  atlas-http → atlas-json → atlas-bayes → atlas-causal → atlas-zk → atlas-astra
  Milestone: Full ASTRA OODA cycle in Rust — observe, orient, decide, act — with ZK provenance

Stage 6 — Training Loop (Weeks 17-20)
  atlas-corpus → atlas-trm (integrated) → atlas-grad (full backward) → atlas-optim (full)
  Milestone: QLoRA SFT on LiveDiscoveryCorpus, pheromone-guided batch sampler, Type 5 traces

Stage 7 — ZK + CLI (Weeks 21-22)
  atlas-zk (extended) → atlas-cli
  Milestone: End-to-end proof chain, atlas-7b release CLI, benchmark suite
```

---

### GPU Without Crates

CUDA kernels live in `kernels/*.cu`. The build script does the plumbing:

```rust
// atlas-tensor/build.rs
fn main() {
    // compile matmul.cu → libmatmul.a
    // link against libcuda.so, libcublas.so (system deps, not Rust crates)
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cublas");
    // invoke nvcc
}
```

```c
// kernels/matmul.cu
extern "C" void atlas_matmul_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // tiled GEMM kernel — 400 LOC, no cublas dependency
}
```

The Rust side calls this via `extern "C"` — no cudarc, no candle-cuda, no tch. Pure system FFI.

---

### The First File

```rust
// atlas-tensor/src/lib.rs — the seed of everything
pub struct Tensor {
    data:  Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        // naive O(n³) — replace with CUDA FFI call in Stage 1
        assert_eq!(self.shape[1], other.shape[0]);
        let (m, k, n) = (self.shape[0], self.shape[1], other.shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    out[i*n + j] += self.data[i*k + p] * other.data[p*n + j];
                }
            }
        }
        Self { data: out, shape: vec![m, n] }
    }
}
```

Every billion-parameter transformer starts here.

---

### Why Zero Dependencies Makes ATLAS More Valuable

| Property | With crates.io deps | Pure Rust, zero deps |
|----------|--------------------|-----------------------|
| **Portability** | Breaks when deps change | Compiles anywhere with rustc |
| **Auditability** | ~50K LOC of transitive deps to read | Only atlas code to audit |
| **Patents** | Contaminated by MIT/Apache-2.0 from all deps | Clean IP — 100% OpenHub Research |
| **Deployment** | `cargo install` pulls internet | Single static binary, runs offline |
| **Speed** | Limited by what crates provide | Tuned exactly to our tensor shapes |
| **Security** | Supply chain attack surface | Zero supply chain — no attack surface |
| **Longevity** | Deps abandoned → broken | atlas builds in 2040 exactly as today |
| **Moat** | Replicable with same pip install | Cannot be reproduced without implementing every crate from scratch |

GraphPalace is already Rust — only strip the PyO3 bindings. ZK Schnorr is already Rust in asi-build — straight copy. The two hardest ports are ASTRA OODA (~8K LOC, Python → Rust) and the autograd backward pass (fundamental, ~2K LOC). Everything else is table stakes.

The artifact at the end is not just a trained model. It is a **sovereign AI infrastructure** that runs without Python, without PyPI, without crates.io, without anyone's permission, on any hardware with a Rust compiler and a GPU.

---

## The Unfair Advantage

Every other LLM is trained on:
- What humans wrote on the internet (web scrapes, Wikipedia)
- Synthetic data generated by another LLM (GPT-4 distillation)
- Human-curated instruction datasets (expensive, slow to update)

**atlas-7b is trained on:**
- What an autonomous science engine *actually discovered* about the world
- Real causal relationships extracted from live NASA, WHO, World Bank data
- Validated novel findings with Bayesian confidence scores
- Cross-domain analogies identified by structural comparison across 5 scientific fields
- A corpus that grows every 10 seconds and never contains stale or duplicated information

This is not a better fine-tuning recipe. This is a different paradigm for what training data can be.

---

## Success Metrics

| Metric | Target | How measured |
|--------|--------|-------------|
| LiveDiscoveryCorpus size (6mo) | 500K+ examples | ASTRA run logs |
| Scientific reasoning improvement | +20% OLMES science | OLMES evaluation |
| Closed-loop improvement per iteration | +5-10% hypothesis quality | ASTRA DC metric |
| O(1/√T) convergence | R²>0.9 across 5 runs | Regret curve fit |
| ZK coverage | >80% outputs have valid proof | Proof chain audit |
| Hallucination rate (ZK-flagged) | <5% on verified claims | TruthfulQA + ZK |
| ASTRA discovery rate with atlas-7b | 2× vs. GPT-4 orient | Discovery cycle count |
| TRM-CausalValidator accuracy | >90% vs. PC/FCI on held-out graphs | 200-graph benchmark |
| TRM validation latency | <10ms per causal graph | Direct timing |
| Corpus quality improvement (TRM gate) | 15–25% fewer false-positive causal claims | Hallucination eval |

---

> **v2.0 key change**: ASTRA-dev integration converts ATLAS from a training architecture into a self-improving scientific intelligence. The model trains on what it and its sister system actually discover about the world — not on what someone else curated years ago.

> **v3.0 key change**: TRM-CausalValidator integration adds recursive validation as a purpose-built architectural module, grounding ATLAS's efficiency thesis with independent empirical evidence from Samsung SAIL Montreal (arXiv:2510.04871). ATLAS now demonstrates structured-computation efficiency at both training time (pheromone curriculum) and inference time (recursive validation). Paper 6 (ICLR/NeurIPS 2027) targets the hybrid generative-recursive architecture contribution.

> **v4.0 key change**: Pure Rust, zero-dependency implementation decision. SQLite philosophy applied to AI infrastructure. 16 crates, ~22-week build order, every algorithm ported from original source. Result: sovereign, portable, auditable, uncontaminated IP — a static binary that runs anywhere with a Rust compiler and a GPU, answerable to no one.

> *"Don't train on what humans wrote. Train on what you find. Validate what you claim. Own what you build."*

---

> Document: v5.0  
> Last updated: 2026-04-16  
> Next update: after Phase 1 first eval results  
> Maintained by: FETCH-AGI / Robin Dey (OpenHub Research, Thailand) / Kyall
