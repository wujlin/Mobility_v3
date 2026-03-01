# Way-CASD Framework Figure Prompt (Journal-Grade)

## Objective

Create a clear, professional model architecture figure for a Nature-family journal paper, intended as **Figure 1 (Method Overview)**.

---

## Overall Figure Requirements

- **Style**: clean and modern; white background; soft gradient blocks
- **Color palette**:
  - Inputs/outputs: deep blues (`#1a365d`, `#2c5282`)
  - Encoder modules: teal/blue-green (`#285e61`, `#319795`)
  - Decoder modules: purple (`#553c9a`, `#805ad5`)
  - Key innovation (Candidate-Aware Attention): orange highlight (`#dd6b20`, `#ed8936`)
  - Feature/data flow: gray arrows (`#4a5568`)
- **Typography**: sans-serif (e.g., Helvetica, Arial); use **sentence case** for labels
- **Canvas**: ~16:9 aspect ratio; suitable for single- or double-column layouts

---

## Figure Layout (Left to Right: Three Main Regions)

### Region A: Input (Left)

**Elements**:
1. **Road Network Graph** (top-left)
   - A simplified road graph (5–7 nodes; edges as road segments)
   - Label: $G = (V, E)$
   - Use different colors or line weights to indicate road hierarchy (tiers)

2. **Route Sequence** (middle-left)
   - A ground-truth route sequence: $\mathbf{r} = (w_1, w_2, \ldots, w_T)$
   - Visualize as consecutive blocks with arrows; each block shows a way ID
   - Mark origin **O** (green) and destination **D** (red)

3. **Multi-Source Features** (bottom-left)
   - A small table or icon group showing three feature families:
     - Geometry: position, direction, length
     - Category: road tier, highway type
     - Semantics: POI, road probability, entropy
   - Use small icons or colored tags to distinguish categories

### Region B: Model Architecture (Center, Core)

Use a large rounded rectangle container with two stacked blocks:

**Top: WayEncoder**
- Title: `WayEncoder` (teal/blue-green background)
- Flow (left → right):
  ```
  Way embeddings → Transformer encoder → Latent tokens z_enc
       (T×d)              ↓               (L×d)
                   [Self-attention]
  ```
- On the right, show a small output box: `z_enc (64×256)`, connected downward to the decoder with a dashed line

**Bottom: WayDecoder** (purple background)
- Title: `WayDecoder (Constrained AR)`
- Show one unrolled decoding step (step t), containing:

  1. **Candidate Set** (left)
     - From the current way $w_{t-1}$, branch to multiple candidates $\{c_1, c_2, c_3\}$
     - Label: `Topologically valid successors`

  2. **Candidate-Aware Cross-Attention** (center; orange highlighted box)
     - This is the core novelty and must be visually emphasized
     - Structure:
       ```
       Base query (cur_emb + past_ctx + dest)
              ↓
       + Candidate embedding → Candidate-specific query
              ↓
       Cross-attention with z_enc
              ↓
       Candidate-specific context (ctx_c)
       ```
     - Use an orange border and a light-orange fill
     - Annotation: **"Each candidate queries z_enc separately"**

  3. **Scorer** (right)
     - Input: `[ctx_c, cur_h, cand_h, diff_from_mean, dist]`
     - MLP → logits → softmax → select
     - Arrow points to the selected candidate (with a check mark)

**Connections**
- From `z_enc` to the cross-attention: thick dashed arrow
- From current way to candidate set: solid arrow
- Between modules: thin arrows

### Region C: Output (Right)

1. **Generated Route**
   - A predicted sequence: $\hat{\mathbf{r}} = (\hat{w}_1, \hat{w}_2, \ldots, \hat{w}_T)$
   - Optionally overlay the predicted route on the road graph, using a distinct color

2. **Key insight** (optional; small text box)
   - "Multi-source information directly participates in candidate ranking"

---

## Key Visual Elements

### 1) Candidate-Aware Attention Detail (Inset or Panel B')

Zoom-in to contrast candidate-agnostic vs. candidate-aware mechanisms:

```
┌─────────────────────────────────────────────────────────┐
│  Traditional (candidate-agnostic)                       │
│                                                         │
│  query ───→ Cross-attn ───→ ctx ─→ expand ─→ [ctx, ctx, ctx] │
│             with z_enc                 (same for all)    │
└─────────────────────────────────────────────────────────┘
                      vs
┌─────────────────────────────────────────────────────────┐
│  Ours (candidate-aware)                     ⭐ HIGHLIGHT │
│                                                         │
│  query + cand₁ ─→ Cross-attn ─→ ctx₁ ─┐                  │
│  query + cand₂ ─→ Cross-attn ─→ ctx₂ ─┼→ [ctx₁, ctx₂, ctx₃] │
│  query + cand₃ ─→ Cross-attn ─→ ctx₃ ─┘     (different!)  │
└─────────────────────────────────────────────────────────┘
```

Mark the traditional approach with a red ✗ and ours with a green ✓.

### 2) Data-Flow Arrows

- Main data flow: thick arrows (3–4 pt)
- Auxiliary conditioning flow: thin dashed arrows (1–2 pt)
- Key innovation: orange arrows

### 3) Legend (Bottom-right)

```
────→  Data flow
- - →  Latent information
█████  Encoder module
█████  Decoder module
█████  Key innovation (Candidate-Aware Attn)
```

---

## Optional Enhancements

1. **Mini performance panel** (as Figure 1b)
   - A simple two-bar comparison: `candq=0` vs. `candq=1`
   - Success Rate: 58% → 82%
   - Keep it minimal (two bars only)

2. **Attention visualization** (as Figure 1c)
   - A heatmap showing different candidates attending to different parts of `z_enc`
   - Title: `Different candidates attend to different latent positions`

---

## Technical Annotations

Add these math snippets (LaTeX-rendered) where appropriate:

- $\mathbf{z}_{\text{enc}} \in \mathbb{R}^{L \times d}$
- $\mathbf{q}_c = \mathbf{q}_{\text{base}} + \text{Proj}(\mathbf{e}_c)$
- $\text{ctx}_c = \text{CrossAttn}(\mathbf{q}_c, \mathbf{z}_{\text{enc}})$
- $\text{logit}_c = \text{MLP}([\text{ctx}_c; \mathbf{h}_{\text{cur}}; \mathbf{h}_c; d_c])$

---

## Reference Style Targets

- Method figures in *Nature Machine Intelligence*
- Transformer architecture diagrams from NeurIPS/ICML papers
- Keep it minimal but information-complete; avoid heavy decoration

---

## Output Formats

- Vector: PDF or SVG (for the paper)
- Raster: PNG at 300 dpi (for slides)
- Width: 180 mm (double-column) or 90 mm (single-column)

---

## Summary (What the Figure Must Communicate)

1. **Inputs**: road graph + multi-source features + route sequence
2. **Encoding**: Transformer compresses into latent tokens
3. **Decoding (key)**: candidate-aware cross-attention lets multi-source information directly participate in ranking
4. **Output**: a topologically valid generated route

Highlight the key novelty in orange so readers immediately see what is new.
