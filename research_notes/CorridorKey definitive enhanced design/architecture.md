# Matting architectures for an enhanced CorridorKey 2.0 (2023 to Oct 2026)

Scope note: about 20 tool calls. Many paper pages fetched were abstract-only, so several mechanisms (MatteFormer, MAM, ZIM, Cutie/XMem, Transparent-460 modern SOTA, smoke) have no cited numbers here and are listed under Gaps. Nothing below is invented; recommendations are flagged as inferences.

## 1. Full-resolution detail: how do top methods recover sub-pixel detail, and is a linear 16x depatchify adequate?

### Takeaway
Every strong detail method adds a cheap high-resolution convolutional or patch-selective path that sees raw pixels at /1 to /4; none relies on a single linear depatchify from /16 tokens. ViTMatte (plain ViT at /16 plus a small conv "detail capture module") and BGMv2 (base net at low res, refinement net on selected patches only) are the canonical designs. Quantified gain from a conv detail path or from generative detail is mostly reported as end-to-end SOTA, not isolated ablation, in the pages I could read.

### Cited Findings
- ViTMatte's "detail capture module" is "simple lightweight convolutions to complement the detailed information required by matting" bolted onto a plain ViT backbone — [ViTMatte arXiv 2305.15272](https://arxiv.org/abs/2305.15272)
- ViTMatte-B: Composition-1k SAD 20.33 / MSE 3.0 / Grad 6.74 / Conn 14.78; Distinctions-646 SAD 17.05 / Grad 7.03; ViTMatte-S Comp-1k SAD 21.46; repo is MIT licensed — [hustvl/ViTMatte](https://github.com/hustvl/ViTMatte)
- BGMv2: a base network produces a low-res result, then a second network refines only selected patches at high resolution; reported 30 fps at 4K and 60 fps at HD on a modern GPU; introduced VideoMatte240K and PhotoMatte13K/85 — [BGMv2 arXiv 2012.07810](https://arxiv.org/abs/2012.07810)
- AEMatter (hybrid-transformer encoder with axis-wise blocks, Swin decoder, trained on 1024x1024 crops to widen context): aggregator-reported Composition-1k SAD 17.53, MSE 2.26, Grad 4.76, Conn 12.46; authors claim ~25% lower SAD and ~40% lower MSE than MatteFormer. Numbers come from aggregator sites, not verified in paper — [Revisiting Context Aggregation (arXiv 2304.01171v2)](https://arxiv.org/html/2304.01171v2); [HyperAI table](https://vercel.hyper.ai/en/sota/tasks/image-matting/benchmark/image-matting-on-composition-1k-1)
- DiffMatte (diffusion decoder for matting, ECCV 2024): aggregator lists Composition-1k SAD 17.15, MSE 2.26; AIM-500 SAD 16.31; Distinctions-646 SAD 15.50 (marked unverified). Paper abstract claims 5% SAD / 15% MSE gain (arXiv v1) vs 8% / 15 (ECCV version): the two sources disagree — [DiffMatte arXiv 2312.05915](https://arxiv.org/abs/2312.05915v1); [ECCV 2024 page](https://eccv2024.ecva.net/virtual/2024/poster/1785); [Papers-with-code mirror](https://astro.paperswithcode.com/paper/diffusion-for-natural-image-matting)
- So the diffusion refiner buys only about 0.4 SAD (17.53 to 17.15) over AEMatter on Composition-1k in the aggregator listing, i.e. a context-wide conv/Swin decoder at full-res crops is nearly as good as a diffusion decoder — derived from the two entries above.
- BiRefNet: localization module plus a reconstruction module that uses hierarchical image patches as a source reference and gradient maps as a target reference (gradient supervision for fine detail); a lite-2K variant accepts 2560x1440 input — [BiRefNet arXiv 2401.03407](https://arxiv.org/pdf/2401.03407); [Aalto record](https://research.aalto.fi/en/publications/bilateral-reference-for-high-resolution-dichotomous-image-segment/); [BiRefNet GitHub fork](https://github.com/whether1/BiRefNet)
- Matting-specific BiRefNet variant model card reports Smeasure 0.979 on TE-P3M-500-NP (model card figure, not paper) — [HF BiRefNet-matting](https://huggingface.co/frankjoshua/BiRefNet-matting)
- CorridorKey 2.0 itself: Stage 2 streams full-resolution predictions from Stage 1 states; the repo gives no accuracy benchmark — [CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)
- MaGGIe uses "transformer attention and sparse convolution" so cost does not grow with instance count (sparse refinement at boundaries) — [MaGGIe arXiv 2404.16035](https://arxiv.org/abs/2404.16035)

### Inferences
- A single linear PixelShuffle from d=1280 /16 tokens must encode a 16x16x4 output block (alpha, FG RGB) in one affine map; this is the structural reason sub-pixel structures (web, hair) fail. The consistent remedy across ViTMatte, BGMv2, BiRefNet and MaGGIe is a high-res path with local convolutions fed by raw pixels. Switching on the already-existing `refine4` /4 ConvNeXt head is the lowest-risk first step, but the shipped checkpoint was trained without it, so it needs training (at least the head, with frozen backbone).
- Better than a global /4 refine at 4K: BGMv2-style or MaGGIe-style sparse refinement that only processes pixels where predicted alpha is in (eps, 1-eps) or uncertainty is high, applied at full res with a small conv net. At 4K most pixels are solid, so this is cheap.
- Training on large crops (AEMatter 1024x1024) matters for context; Stage 2's Swin/CSwin blocks already provide this.

### Gaps
- Isolated ablation numbers for "linear head vs conv decoder" were not found; MatteFormer, MAM (Matting Anything) and ZIM pages were not retrieved.
- No verified paper-level (non-aggregator) numbers for AEMatter or DiffMatte.

## 2. Temporal coherence: recurrent, memory propagation, attention windows, flow; should the detail stage be temporal?

### Takeaway
On 1080p benchmarks, memory-propagation methods (MatAnyone 1/2) beat RVM-recurrent and MaGGIe on dtSSD by 10 to 35%, and generative video-prior GVM sits between them. CorridorKey's temporal reasoning is only at /64 (Stage 1) with Stage 2 per-frame, which fits the benchmark result of per-frame flicker of fine detail; the evidence supports a light temporal path in the detail stage (stateful or short-window), not necessarily full attention.

### Cited Findings
- VideoMatte 1920x1080 (MAD / MSE / Grad / dtSSD): MatAnyone 2 4.10/0.28/3.45/1.15; MatAnyone 4.24/0.33/4.00/1.19; MaGGIe 4.42/0.40/4.03/1.31; RVM-Large 5.81/0.97/9.65/1.78; RVM 6.57/1.93/10.55/1.90; GVM 6.33/2.08/8.04/1.59 — [MatAnyone 2 arXiv 2512.11782v2](https://arxiv.org/html/2512.11782v2)
- YouTubeMatte 1920x1080: MatAnyone 2 1.61/0.50/7.14/1.53; MatAnyone 1.99/0.71/8.91/1.65; MaGGIe 2.37/0.98/7.69/1.77; GVM 2.68/1.34/8.38/1.72; RVM-Large 3.58/1.23/12.97/2.04; RVM 4.37/2.25/15.10/2.28 — same source
- Real-world CRGNN set: MatAnyone 2 4.24/2.00/11.74/4.54; GVM 5.03/2.15/14.28/4.86; RVM-Large 5.75/2.47/13.26/5.17; MatAnyone 5.76/3.04/15.55/5.44; RVM 5.98/2.79/13.68/5.36; MaGGIe 9.50/6.11/16.51/6.02 — same source. Note MaGGIe is worst on real data despite good synthetic numbers.
- MatAnyone 2 ablation on YouTubeMatte (MAD/MSE/Grad/dtSSD): baseline 1.99/0.71/8.91/1.65; + online quality-evaluator guidance 1.90/0.62/8.20/1.63; + VMReal data 1.76/0.61/7.65/1.54; + reference-frame training 1.61/0.50/7.13/1.53. Reference-frame strategy adds no extra memory cost. So data and training strategy (-12% MAD from data, -9% from reference frames) outweigh evaluator guidance (-5%) — same source
- Training RVM on VMReal improved every metric (their Table 6), indicating data matters independent of architecture — same source
- MatAnyone 1: memory-based, target-assigned framework with "consistent memory propagation module via region-adaptive memory fusion" that keeps core regions semantically stable while preserving boundary detail; trained with large-scale segmentation data for stability — [MatAnyone arXiv 2501.14677](https://arxiv.org/abs/2501.14677)
- VMReal: 28K clips, 2.4M frames; a learned Quality Evaluator gives a per-pixel reliability map used for training feedback and offline data curation; CVPR 2026 — [arXiv 2512.11782](https://arxiv.org/pdf/2512.11782); [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36883)
- GVM (Stable Video Diffusion base) vs RVM on VideoMatte240K: MAD 5.88 vs 6.39, Grad 5.00 vs 9.95, Conn 5.27 vs 6.53, dtSSD 1.11 vs 1.85 (-40% dtSSD, -50% Grad); on V-HIM60 Hard MAD 11.04 vs 17.96 — [GVM arXiv 2508.07905](https://arxiv.org/html/2508.07905v1). Note: the numbers for GVM differ between its own paper (VM240K MAD 5.88) and the MatAnyone 2 paper (6.33); evaluation protocols differ, so cross-paper numbers are not comparable.
- CorridorKey 2.0 known issue: Hann carry can drift on long crowded shots or large occlusions; small windows weaken long-range consistency — [CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)

### Inferences
- dtSSD gains in the literature come from (a) alpha-memory with a stable core/boundary split and (b) data; none of the top video methods use optical-flow warping in their latest versions (I found no evidence either way for flow in 2025/26 papers; not claimed).
- CorridorKey's Stage 1 at /64 gives correct semantic/temporal context but boundary-level flicker arises because Stage 2 sees each frame alone. A cheap fix: feed Stage 2 the previous frame's final output (alpha/FG) warped-free as extra channels, or add 3-frame temporal attention (center frame plus prev/next, 1x1xT axial) only inside a small number of Stage 2 blocks restricted to the uncertain band. This mirrors MatAnyone's use of previous-frame alpha memory fused region-adaptively (core vs boundary).
- Reference-frame training (sampling frames beyond the local window during training) is free at inference and gave the single largest MAD/MSE gain among MatAnyone 2 components after data; it applies directly to Hann-window training.

### Gaps
- Cutie/XMem architecture numbers and MESSDdt were not retrieved; no direct evidence on flow warping for matting; no head-to-head against 80-frame full space-time attention.

## 3. Subject selection / target identity

### Takeaway
The best-performing recent methods are mask-guided (first-frame or per-frame mask from SAM2) with target-assigned memory; guidance reduces wrong-subject failures. CorridorKey already has hint modes; its documented failure (keyed a different subject with no hint) matches the no-hint ambiguity the README admits.

### Cited Findings
- CorridorKey 2.0: with no hint the model may pick a different subject when several are plausible; hint modes are none / first-frame matte with Hann carry / full hint video; full hint video biases output toward the hint's systematic errors; `--hint_quality` sets trust — [CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)
- MatAnyone is "target-assigned": a first-frame mask selects the target and consistent memory propagation keeps it — [arXiv 2501.14677](https://arxiv.org/abs/2501.14677)
- VideoMaMa (mask-guided, SVD-based) turns a coarse SAM2 mask into a fine matte. On V-HIM60 Hard, MAD/Grad: SAM2 7.85/26.34; MatAnyone 5.72/9.82; SAM2+VideoMaMa 2.78/4.83; SAM2-Matte (SAM2 fine-tuned on their pseudo-labeled MA-V set) 2.61/5.09 — [VideoMaMa arXiv 2601.14255](https://arxiv.org/html/2601.14255v1)
- MaGGIe: instance-guided, simultaneous multi-instance mattes with cost independent of instance count — [arXiv 2404.16035](https://arxiv.org/abs/2404.16035)

### Inferences
- The tracking-marker failure is a prior problem, not an architecture limit: a foreground-selection hint (SAM2 mask or a keyer-style "garbage matte" box) is the practical solution; in VFX workflows a rough garbage matte is normal. Training with randomized hint damage (the existing AdaLN hint-quality conditioning) already follows MatAnyone/VideoMaMa practice. Add an explicit "negative" training class: small high-contrast non-subject objects (markers, stands) labelled alpha=0 in the hint.
- Hint conditioning at patchify (concatenate) plus AdaLN is adequate; adding cross-attention from frozen-feature tokens at the masked region (mask-pooled prototype, as in memory-based methods) is a cheap strengthening but untested in sources.

### Gaps
- No source on SAM2-guided matting numbers beyond VideoMaMa; ZIM and Matting Anything not retrieved.

## 4. Transparency and semi-transparent media (glass, smoke)

### Takeaway
Little modern architecture-specific evidence was found. The literature points to confidence/known-region conditioning and tri-token (learned) context; the fundamental issue for glass and smoke is that alpha alone is ill-posed (refraction, additive emission), so output parameterization matters more than backbone.

### Cited Findings
- TransMatting (ECCV 2022): Transformer with learnable "tri-tokens" replacing a trimap, plus Transparent-460 dataset; aggregator lists SAD 192.4 on Transparent-460 test — [TransMatting arXiv 2208.03007](https://arxiv.org/pdf/2208.03007); [aggregator](https://astro.paperswithcode.com/paper/transmatting-enhancing-transparent-objects)
- TOM-PDFB (TCSVT Jan 2025): estimates confidence in definite foreground/background regions and feeds it to a refinement network, for cases where known foreground is small or missing — [TOM-PDFB project](https://github.com/yihuiliang/TOM-PDFB) (via search summary)
- TOM-Net (2018) assumes colorless, specular transparent objects; does not handle colored translucent or refraction plus reflection — [arXiv 1803.04636](https://arxiv.org/abs/1803.04636v1)
- I found no dedicated 2024 to 2026 smoke matting benchmark in search.

### Inferences
- For green-screen footage, glass and smoke obey the compositing equation C = alpha*F + (1-alpha)*B only if F is allowed to be emissive/colored; glass tint is a multiplicative (not additive) effect. Best-practice parameterization (inference, not from a source): predict alpha plus straight F as now, but add a per-pixel transmission color (3-channel "alpha") head for tinted glass; with a known green screen, B is nearly known, so a background (clean plate) prediction head provides a self-consistency loss C ≈ alpha*F + (1-alpha)*B_pred.
- Huge edge error on glass is consistent with a loss dominated by opaque pixels; weighting loss by uncertainty-band mask and including synthetic glass/smoke renders is cheap and architecture-independent.

### Gaps
- No quantitative source for smoke; no modern transparent-matting SOTA table beyond TransMatting's aggregator entry.

## 5. Output parameterization (straight vs premultiplied FG, logits, clean plate, uncertainty)

### Takeaway
Little direct sourced evidence; the MatAnyone 2 result is the strong sourced precedent for per-pixel quality estimation. Others are design inferences.

### Cited Findings
- MatAnyone 2's Quality Evaluator produces a pixel-wise reliability map without ground truth and is used both as training loss guidance and data selection — [arXiv 2512.11782](https://arxiv.org/pdf/2512.11782)
- Its ablation attributes -5% MAD / -8% Grad to online evaluator guidance (1.99 to 1.90 MAD, 8.91 to 8.20 Grad) — [arXiv 2512.11782v2](https://arxiv.org/html/2512.11782v2)
- CorridorKey 2.0 runs bf16 in an asinh HDR domain; README states this bounds accuracy to roughly 1 to 2% relative error regardless of EXR pixel type — [CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)

### Inferences
- The 1 to 2% bf16 floor limits fine-edge precision (alpha steps near 0 or 1 at web strands). Running the final conv/linear head and depatchify in fp16/fp32 (small cost, as it is one layer plus refine head) removes that ceiling without touching the backbone. Alpha-logit output with a sigmoid, or predicting alpha in a space with finer resolution near 0/1, would help semi-transparent tails (unsourced design suggestion).
- An uncertainty head (aleatoric variance, or MatAnyone-2-style learned evaluator) enables the sparse refiner (Section 1) and a deployable "needs review" mask for VFX users.

### Gaps
- No source comparing straight versus premultiplied FG prediction; no source on clean-plate prediction in matting.

## 6. Generative / diffusion-prior matting: worth the fidelity per cost?

### Takeaway
Video-diffusion priors give real generalization and consistency gains (GVM, VideoMaMa) but cost roughly 30x to 60x the runtime of a regression network and have licensing friction; their benefit over MatAnyone-2 class memory methods on standard benchmarks is not established. Not recommended as the core; a distilled prior is cheaper.

### Cited Findings
- GVM latency on A100 (single image): RVM-R50 0.01 s at 512x512, 0.02 s at 1024x1024; GVM 1-step 0.32 / 0.65 s; GVM 3-step 0.53 / 0.90 s; full SVD 50-step 5.44 / 7.33 s. Authors acknowledge speed is not comparable to regression — [GVM arXiv 2508.07905](https://arxiv.org/html/2508.07905v1)
- GVM's own repo: one-step model, 960 max resolution in the sample command, warns results differ slightly from the paper; BSD-2-Clause for academic use, commercial requires contacting the author — [aim-uofa/GVM](https://github.com/aim-uofa/GVM)
- Against MatAnyone 2 (their tables), GVM is behind on all four metrics on VideoMatte, YouTubeMatte, and CRGNN (e.g. CRGNN MAD 5.03 vs 4.24) — [arXiv 2512.11782v2](https://arxiv.org/html/2512.11782v2)
- VideoMaMa: SVD-based, single-pass v-parameterized, DINO feature injection; stage 1 trained at 1024x1024 images, stage 2 on 3-frame clips at 704x704; evaluated at 1920x1080. Code CC BY-NC 4.0; weights fall under the Stability AI Community License; MA-V dataset not yet released — [paper](https://arxiv.org/html/2601.14255v1); [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
- Generative Video Matting is the first listed use of SVD priors for matting to close the synthetic-to-real gap; VRMDiff does text-referred matting with video diffusion — [search summary](https://arxiv.org/abs/2503.10678v1)
- GVM and VideoMaMa training at 704 to 1024 px means native 4K detail requires tiling or a separate refiner.

### Inferences
- Diffusion generalization benefit (real footage, ambiguous scenes) is already partially obtained by CorridorKey from frozen foundation features (C-RADIO, MoGe-2, RVM). Adding an SVD prior (about 1.5B-param class) would add >10x compute for unproven gain over a well-trained memory model. The data lesson (VMReal, MA-V) is more transferable: pseudo-labeled real footage improved even RVM across all metrics.
- Licensing: Stability AI Community License on VideoMaMa weights and BSD-academic GVM are not clearly compatible with CorridorKey's CC BY-NC-SA variant; flag for legal check before any weight reuse.

### Gaps
- No found head-to-head of a diffusion refiner on top of a regression keyer at 4K.

## 7. Chroma-key / green-screen specific ML, and licensing

### Takeaway
Little published ML specific to chroma keying exists; the closest are Netflix's Magenta Green Screen (data generation) and Background Matting. The useful lever is synthetic and physical ground-truth data, not architecture.

### Cited Findings
- Magenta Green Screen (Netflix, DigiPro 2023): actor lit only by red and blue light against a green screen; the green channel acts as a holdout matte (inverse alpha), and a learned colorization model restores green. The authors say the resulting alpha could be higher-quality training data for matting; no quantitative metrics in the abstract; no dataset release stated — [arXiv 2306.13702](https://arxiv.org/abs/2306.13702); [PetaPixel](https://petapixel.com/2023/07/11/netflixs-new-magenta-method-makes-removing-green-screens-much-easier)
- Background Matting (CVPR 2020) estimates alpha and foreground color from the image plus a captured empty background plus soft segmentation — [Sengupta et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sengupta_Background_Matting_The_World_Is_Your_Green_Screen_CVPR_2020_paper.pdf)
- My search found nothing on learned spill suppression or foreground decontamination from 2025.
- Licenses relevant to a free, non-commercial derivative: CorridorKey 2.0 is a CC BY-NC-SA 4.0 variant; derivatives must be freely accessible under the same license and keep the name — [CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0). MatAnyone and MatAnyone 2 use NTU S-Lab License 1.0 (weights terms unstated) — [MatAnyone](https://github.com/pq-yang/MatAnyone); [MatAnyone2](https://github.com/pq-yang/MatAnyone2). ViTMatte MIT — [hustvl/ViTMatte](https://github.com/hustvl/ViTMatte). VideoMaMa code CC BY-NC 4.0, weights Stability Community License — [repo](https://github.com/cvlab-kaist/VideoMaMa). GVM BSD-2 academic only — [repo](https://github.com/aim-uofa/GVM).
- MatAnyone 2 released inference/eval code and weights (2026-03) but not training code, the quality evaluator checkpoint, or VMReal — [MatAnyone2 repo](https://github.com/pq-yang/MatAnyone2)

### Inferences
- Because green-screen background B is near-uniform, a clean-plate/B head and physically-based synthetic green-screen composites (refraction, smoke, spill) are far cheaper sources of exact alpha for glass and smoke than real data. Magenta Green Screen-style multiplexed capture is a possible route to real ground truth but needs a lighting rig.
- Architecture ideas can be reused freely (not licensed); weights/data from NTU S-Lab, Stability, and GVM need individual license review.

### Gaps
- No commercial keyer (Keylight, Primatte, Ultimatte) architecture details found; no spill-suppression ML source.

## 8. Recommendation: single best enhanced architecture (what to keep, change, add)

This section is inference built on the cited findings above; effect sizes marked "reported" come from the cited papers, others are expectations.

### Takeaway
Keep the Stage 1 /64 space-time transformer, the frozen foundation features, the two-stage split and Hann windows. Replace the single linear depatchify with a hybrid high-resolution detail head: a /4 to /1 conv refiner fed raw RGB (and pre-clean B estimate), activated sparsely on an uncertainty band, in fp16/fp32, with light temporal conditioning on the previous output. Fix data (glass/smoke/web/hair synthetic plus pseudo-labeled real, reference-frame training) alongside. Do not add a diffusion prior.

### Cited Findings
(see Sections 1 to 7; key numbers: MatAnyone 2 over MatAnyone on YouTubeMatte MAD 1.99 to 1.61 via data + reference-frame + evaluator; GVM vs RVM dtSSD 1.85 to 1.11; GVM 30x to 60x slower than RVM; DiffMatte only 0.4 SAD better than AEMatter.)

### Inferences (ranked by expected value per cost)
1. Detail head (addresses spider web, hair, sub-pixel, glass edges). Enable and train `refine4`; add a /1 sparse conv stage on pixels where predicted uncertainty or 0.02<alpha<0.98 (BGMv2/MaGGIe pattern). Input: raw full-res RGB, upsampled Stage 2 alpha/FG, baton features. Run in fp32. Expected: the main fix for sub-pixel detail; magnitude unmeasured here. ViTMatte's lightweight conv detail module is evidence that a few conv layers recover most of the loss of a /16 ViT.
2. Training data and curriculum (glass, smoke, compression, low light, markers). MatAnyone 2 attributes -12% MAD (1.90 to 1.76) to better data and -9% MAD (1.76 to 1.61) to reference-frame training, and training RVM on that data improved it too. Build physically-based synthetic green-screen composites with refractive glass, volumetric smoke, spill; add compression/noise augmentation; add non-subject marker negatives.
3. Temporal stabilization of fine detail. Feed previous-frame final alpha/FG (stop-grad, region-adaptive gating à la MatAnyone) into the detail head, and/or give 2 to 3 Stage 2 blocks a 3-frame axial attention restricted to the uncertainty band. Expected: dtSSD reduction in the 10% to 35% range seen between RVM-class and memory-class methods (reported spread in the MatAnyone 2 tables: 1.78 to 1.15 on VideoMatte), unverified for this model.
4. Output heads: add (a) uncertainty/quality head (MatAnyone 2 evaluator precedent: -5% MAD, -8% Grad as training guidance), (b) clean-plate B head with recomposition loss, (c) optionally 3-channel transmission for tinted glass. Do final linear/conv output layers in fp32 to remove the bf16 1 to 2% floor.
5. Subject selection: keep AdaLN hint conditioning; make the first-frame SAM2 mask the default recommended workflow; add marker negatives; add reference-frame training for long clips so drift under Hann carry shrinks.
6. Skip: diffusion/SVD prior (30x+ slower, licensing friction, behind MatAnyone 2 in its own tables), optical-flow warping (no evidence), larger Stage 1 (no evidence it is the bottleneck).

### Gaps
- None of the expected effect sizes for items 1, 3, 4 are measured on CorridorKey; they require ablation on the user's own benchmark. MatteFormer, MAM, ZIM, Cutie/XMem, MESSDdt, and a direct linear-vs-conv decoder ablation were not retrieved.
