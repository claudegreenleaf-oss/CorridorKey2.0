# Frozen foundation models and extra input signals for an enhanced CorridorKey 2.0 keyer (as of 2026-10-21)

Method note: about 25 fetches/searches. Where a fetch returned only the repo README or arXiv abstract, numbers that were not on the page are listed under Gaps rather than guessed. Claims about CorridorKey 2.0's own architecture come from the task brief and were not independently verified. Items marked (unverified) are from search snippets or secondary blogs.

## 1. Dense feature backbones (DINOv3, C-RADIO, PE-spatial, SAM 3, V-JEPA 2.1, RVM)

### Takeaway
The current trio (C-RADIOv4-SO400M, MoGe-2, RVM) is already a strong, license-clean set. C-RADIOv4 already distills DINOv3-7B, SigLIP2-g and SAM3, so adding a raw DINOv3 or SAM 3 encoder is largely redundant with it. The best remaining upgrade is not a new backbone but a better way to feed the existing ones at higher effective resolution (feature upsampler or tiling) plus an explicit motion signal. If one backbone is added, DINOv3 ViT-L/16 (or distilled H+) is the most proven in matting papers (CineMatte freezes DINOv3), but it carries the DINOv3 License rather than NVIDIA's.

### Cited Findings
- DINOv3 family: ViT-S/16 21M, S+ 29M, B 86M, L 300M, H+/16 840M, 7B 6,716M; ConvNeXt T/S/B/L 29M/50M/89M/198M (distilled from the 7B); satellite ViT-L and 7B variants; the 7B training includes a high-resolution adaptation stage — [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- DINOv3 ADE20k linear: ViT-7B/16 55.9, ViT-L/16 54.9 (model cards); distilled ConvNeXt 42.7 (T) to 47.8 (L) at 256px — [HF model card ViT-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- DINOv3 paper: Gram anchoring targets dense feature degradation over long training; claims dense features beat specialized SOTA without task fine-tuning — [arXiv 2508.10104](https://arxiv.org/abs/2508.10104)
- C-RADIOv4-SO400M: about 431M params, patch 16, teachers SigLIP2-g, DINOv3-7B and SAM3; inputs up to 2048 px in steps of 16 (tested 256 to 2048); ADE20k mIoU 55.14 (C-RADIOv4-H 55.20, C-RADIOv3-H 52.75, v3-L 51.87, v3-B 49.79); VOC 87.22; also a C-RADIOv4-H at 653M; no VRAM/throughput listed — [HF nvidia/C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M)
- C-RADIOv4-SO400M's 55.14 ADE20k is within 0.8 of the DINOv3 7B (55.9) at about 6% of the parameters (431M vs 6.7B), and above DINOv3 L (54.9) — [C-RADIOv4 card](https://huggingface.co/nvidia/C-RADIOv4-SO400M) vs [DINOv3 cards](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). Caveat: probe protocols may differ between the two cards (unverified comparability).
- Perception Encoder spatial (PE-spatial): G/14 at 448px ADE20k 49.3, DAVIS zero-shot J&F 61.5; distilled L/14 448px 48.1 / 60.6, B/16 512px 44.4 / 58.9, S/16 37.5 / 57.5, T/16 27.6 / 55.0; Apache-2.0 — [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models). Below DINOv3/C-RADIO on ADE20k.
- SAM 3: 848M params, detector and tracker share one vision encoder (Perception Encoder), SAM License; SAM 3.1 (2026-03-27) adds Object Multiplex joint multi-object tracking — [facebookresearch/sam3](https://github.com/facebookresearch/sam3). Reported ~7x faster at 128 tracked objects on H100 (secondary source) — [Labellerr](https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/)
- V-JEPA 2.1 (arXiv 2603.14482, 2026-03): dense predictive loss, deep self-supervision, claims "temporally consistent dense features"; NYUv2 linear-probe depth RMSE 0.307, SSv2 77.7; abstract gives no segmentation numbers — [arXiv](https://arxiv.org/abs/2603.14482)
- RVM (current stack): variants ViT-S 34M, B 117M, L 375M, H 743M; software Apache-2.0, other materials CC-BY 4.0; matches or exceeds DINOv2 on geometric/dense tasks and VideoMAE/V-JEPA on tracking; recurrence gives linear-cost long-horizon propagation — [representations4d](https://github.com/google-deepmind/representations4d), [arXiv 2512.13684](https://arxiv.org/abs/2512.13684)
- Matting use of backbones: CineMatte (arXiv 2605.18328, 2026-05) uses a frozen, weight-shared DINOv3 ViT in a Siamese setup for frame plus clean plate, and replaces the conv detail branch with a pretrained image-guided feature upsampler to fix boundary artifacts from semantic misalignment — [arXiv](https://arxiv.org/abs/2605.18328)
- SAM2Matting (arXiv 2606.27339): keeps SAM2/SAM3 tracker intact, adds a region-proposal bridge and matting heads, trained on images only, claims SOTA video matting — [arXiv](https://arxiv.org/abs/2606.27339)
- MatAnyone 2 reportedly uses a DINOv3-based quality evaluator with a DPT head for training-data curation (secondary source, unverified) — [gaga.art](https://gaga.art/blog/matanyone-2/)

### Inferences
- Because C-RADIOv4 already contains DINOv3-7B, SigLIP2 and SAM3 knowledge, a second image backbone will add mostly redundant semantics; the marginal gain is larger from resolution and motion than from another encoder.
- C-RADIOv4-H (653M) gains only 0.06 ADE20k over SO400M (431M), so SO400M is the right size/cost point; no reason to upgrade.
- RVM is the only stateful video encoder in the set and is the one piece with no drop-in replacement; V-JEPA 2.1 is the only credible alternative but lacks dense segmentation evidence and a verified license, so keep RVM and consider V-JEPA 2.1 only as an experiment.
- PE-spatial and VideoMAE v2 are dominated on dense metrics and not worth adding.
- Edge quality (hair, fur) is limited by the /16 patch grid, not by feature semantics; this is why CineMatte pairs a frozen ViT with an image-guided upsampler.

### Gaps
- No VRAM or FLOPs figures were published for C-RADIOv4, DINOv3 ViT-L/H+ or RVM at 4K; these need measurement. Rough scaling: ViT FLOPs grow quadratically in token count in attention and linearly in MLP, so 4x the pixels is about 4 to 6x cost.
- No head-to-head matting ablation of DINOv3 vs C-RADIO vs SAM encoders was found. VideoMAE v2 and other 2025-26 dense video encoders (beyond V-JEPA 2.1 and RVM) were not searched in depth.
- V-JEPA 2.1 license and model sizes not verified.
- DINOv3 vs C-RADIOv4 ADE20k comparison above mixes two different cards; protocol parity unverified.

## 2. Depth and geometry (MoGe-2, Depth Anything 3, Video Depth Anything)

### Takeaway
Keep MoGe-2 ViT-B (MIT code, small and fast). Temporally consistent depth is attractive but the licensed options for it (Video Depth Anything Base/Large, Depth Anything 3 Large) are non-commercial CC-BY-NC, and only VDA-Small is Apache-2.0. Given the keyer already carries RVM temporal state, a separate video depth model is low priority.

### Cited Findings
- MoGe-2: MIT code (DINOv2 module Apache-2.0); variants vitl 326M, vitl-normal 331M, vitb-normal 104M, vits-normal 35M; all metric scale; about 60 ms/image on A100/RTX3090 FP16 (ViT-L); `num_tokens` 1200 to 2500 recommended, `resolution_level` 0 to 9; no documented video mode — [microsoft/MoGe](https://github.com/microsoft/MoGe)
- Video Depth Anything: Small 28.4M Apache-2.0; Base 113.1M and Large 381.8M CC-BY-NC-4.0; Large latency 67 ms FP32 / 14 ms FP16 and 26.7 / 23.6 GB VRAM on A100 at 1x32x518x518 (clip of 32 frames); streaming mode (2025-07) reduces ScanNet d1 from 0.926 to 0.836 — [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- VDA design: Depth Anything V2 backbone with a spatial-temporal head and a temporal-gradient consistency loss — [arXiv 2501.12375](https://arxiv.org/html/2501.12375v3)
- Depth Anything 3 Large: CC BY-NC 4.0, about 0.35 to 0.4B params, aimed at multi-view depth and pose; no temporal model described on the card — [HF DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE)

### Inferences
- MoGe-2's per-frame output may flicker; since the keyer fuses it as a feature at /64 resolution and RVM supplies temporal context, flicker is likely absorbed by the network. Test by temporally smoothing or feeding two frames.
- Depth's real value for keying is separating foreground from background by distance and flagging transparent/semi-transparent regions (glass, smoke), not edge detail, so low resolution is acceptable.
- VDA-Small (Apache-2.0, 28M) is the only license-clean temporal depth option; its quality versus MoGe-2 for this purpose is unmeasured here.

### Gaps
- No head-to-head temporal-consistency benchmark of MoGe-2 vs VDA vs DA3 found. DepthCrafter and Depth Pro were not researched (diffusion-based DepthCrafter is expected to be far too heavy; unverified).

## 3. Motion (optical flow)

### Takeaway
Add a flow signal only if ablation shows RVM state is insufficient to separate motion blur from transparency. If added, MEMFOF (BSD-3-Clause, 1080p-oriented, about 2 GB VRAM) or SEA-RAFT (BSD-3-Clause) are the practical choices. Compute flow at reduced resolution and feed it as a 2-channel plus magnitude input or use it to warp the previous alpha.

### Cited Findings
- SEA-RAFT (ECCV 2024): BSD-3-Clause; claims 2.3x faster than existing methods; Spring EPE 3.69 and 1px outlier 0.36; about 21 fps on RTX 3090 at 540x960 (the 1080p claim in secondary summaries is actually 540x960) — [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [arXiv 2404.04808](https://arxiv.org/abs/2404.04808)
- MEMFOF (ICCV 2025): BSD-3-Clause; needs about 2.09 GB GPU memory at runtime for 1080p; 28.5 GB for training; ranked first on Spring (1px outlier 3.289) at submission; checkpoints include a "TSKH" variant best for real video — [msu-video-group/memfof](https://github.com/msu-video-group/memfof), [arXiv 2506.23151](https://arxiv.org/html/2506.23151v1)
- MemFlow (CVPR 2024): leads 1080p Spring at its submission; license not found — [arXiv 2404.04808 (MemFlow listing)](https://www.catalyzex.com/paper/memflow-optical-flow-estimation-and)

### Inferences
- Flow estimated by RAFT-family models degrades exactly on the cases of interest (motion blur, smoke, glass), so flow alone cannot disambiguate transparency; it is better used for warping the prior alpha (temporal propagation) and for a "motion of boundary vs interior" cue.
- At 4K, run flow at 1080p or lower and upsample; MEMFOF's low memory makes this affordable.

### Gaps
- No 4K timings, no FlowFormer++ or 2026 flow models checked, and no evidence from a matting paper that flow input improves alpha quality.

## 4. Subject identity (SAM 2/3, MatAnyone 2)

### Takeaway
Use SAM 3 / SAM 3.1 or SAM 2.1 only as an optional hint generator (first-frame mask or click), not as a per-frame feature input. MatAnyone 2 is a strong reference but its S-Lab license is non-commercial and it requires a first-frame mask, so use it for pseudo-labels or baselines, not as a shipped dependency.

### Cited Findings
- SAM 3: 848M params; shared encoder for detector and tracker; SAM License; checkpoints gated on Hugging Face; SAM 3.1 adds multi-object tracking speedup — [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- MatAnyone 2 (CVPR 2026, arXiv 2512.11782): learned quality evaluator and VMReal dataset (28K clips, 2.4M frames); NTU S-Lab License 1.0 ("non-commercial" per secondary sources); requires a first-frame segmentation mask; VMReal dataset not yet released per the README TODO; no speed/VRAM published — [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/36883), [pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- SAM2Matting extends SAM2/SAM3 trackers with matting heads, trained on images only — [arXiv 2606.27339](https://arxiv.org/abs/2606.27339)

### Inferences
- For a green-screen keyer the "which subject" ambiguity is mostly handled by the screen colour itself; a one-click or garbage-matte hint channel is cheaper than running a tracker every frame.
- C-RADIOv4 already carries SAM3 knowledge, so subject/instance awareness is partially in the features.

### Gaps
- SAM 2 / 2.1 license and speed were not fetched (SAM 2 is believed to be Apache-2.0, unverified here). SAM 3.1 VRAM and fps not published on the repo.

## 5. Resolution strategy and feature upsamplers

### Takeaway
Do not just scale VFM input proportionally to 4K. Feed VFMs a capped, fixed-ish resolution (for C-RADIOv4 up to about 1 to 1.5 MP, within its tested 256 to 2048 range) and recover edges with a lightweight image-guided upsampler (AnyUp or JAFAR-style) conditioned on the full-resolution frame. Tile only for 4K if latency permits.

### Cited Findings
- AnyUp (arXiv 2510.12764): feature-agnostic; trained once on DINOv2 ViT-S, applied to other encoders; linear-probe mIoU on DINOv2 ViT-S features COCO / VOC / ADE20k: Bilinear 59.48 / 81.43 / 40.54, FeatUp 61.95 / 83.37 / 42.19, LoftUp 62.15 / 83.69 / 42.02, JAFAR 61.82 / 84.36 / 42.06, AnyUp 62.16 / 84.00 / 42.43; at 448 px on A100 runtime AnyUp 92.4 ms, JAFAR 113.9, LoftUp 201.2, FeatUp 73.7; peak memory fwd 3.3 GB vs JAFAR 6.9 and LoftUp 7.9; transfers to SigLIP2 (58.51 mIoU) and DINOv3 (62.96) — [arXiv HTML](https://arxiv.org/html/2510.12764)
- UPLiFT (CVPR 2026, arXiv 2601.17950): local-attender iterative upsampler claiming SOTA at lower inference cost than cross-attention upsamplers; comparison numbers not retrieved — [arXiv](https://arxiv.org/abs/2601.17950)
- Gains from upsamplers over bilinear on segmentation probes are about 2 to 3 mIoU on semantic tasks; the larger benefit is at boundaries, which linear-probe mIoU understates (inference).
- CineMatte uses an image-guided pretrained upsampler instead of a conv detail branch for a frozen DINOv3 matting model at 4K — [arXiv 2605.18328](https://arxiv.org/abs/2605.18328)
- MoGe-2 accepts a token budget (1200 to 2500 tokens), so its cost is controllable independent of image size — [microsoft/MoGe](https://github.com/microsoft/MoGe)

### Inferences
- Current setup: at 720p the VFMs see about 320x180 (about 20x11 patches for patch-16, which is very coarse). Raising C-RADIOv4 input to about 640x360 to 960x540 (1.5 to 3x per side, 4 to 9x tokens) is likely the cheapest large quality gain and stays inside the model's tested range.
- Fixed token budget (as MoGe-2 does) keeps VRAM constant across 720p to 4K, but loses detail on 4K; combine with upsampler.
- AnyUp's training on a ViT-S DINOv2 and transfer to others is not guaranteed for C-RADIO's features (not tested in the paper); a short fine-tune on C-RADIOv4 features may be needed.

### Gaps
- No published test of AnyUp/JAFAR/LoftUp on C-RADIO or RVM features, and no matting-specific edge metric (e.g. gradient/connectivity error) for upsamplers.
- FeatUp/LoftUp/JAFAR licenses not checked.

## 6. Licenses (summary)

### Takeaway
Everything in the current stack plus DINOv3 and SAM 3 permits free redistribution of derivatives with attribution and including the license text; the ones that block commercial use are Video Depth Anything Base/Large, Depth Anything 3 Large (CC-BY-NC) and MatAnyone 2 (S-Lab). A free, non-commercial-compatible release is fine with all of them.

### Cited Findings
- DINOv3 License: non-exclusive, worldwide, non-transferable, royalty-free; allows use, copying, distributing, modifying and derivatives; distribution must carry the agreement; research publications must acknowledge use; no explicit commercial prohibition; prohibits reverse engineering and military/ITAR uses; terminates if you sue Meta; indemnity clause — [LICENSE.md](https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md)
- SAM License (SAM 3 repo): essentially the same structure as the DINOv3 License, with the same acknowledgement, prohibited-uses, termination and indemnity terms — [SAM License text](https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE). Summaries were produced by a fetch model, not legal advice.
- C-RADIOv4: NVIDIA Open Model License; card says ready for commercial and non-commercial use — [HF card](https://huggingface.co/nvidia/C-RADIOv4-SO400M)
- MoGe: MIT (DINOv2 submodule Apache-2.0) — [MoGe](https://github.com/microsoft/MoGe)
- RVM: Apache-2.0 code, CC-BY 4.0 other materials — [representations4d](https://github.com/google-deepmind/representations4d)
- PE: Apache-2.0 — [perception_models](https://github.com/facebookresearch/perception_models)
- SEA-RAFT and MEMFOF: BSD-3-Clause — [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT), [MEMFOF](https://github.com/msu-video-group/memfof)
- Video Depth Anything: Small Apache-2.0; Base and Large CC-BY-NC-4.0 — [repo](https://github.com/DepthAnything/Video-Depth-Anything)
- Depth Anything 3 Large: CC BY-NC 4.0 — [HF](https://huggingface.co/depth-anything/DA3-LARGE)
- MatAnyone 2: NTU S-Lab License 1.0, non-commercial — [repo](https://github.com/pq-yang/MatAnyone2)

### Inferences
- A distributed CorridorKey checkpoint that bakes in frozen weights does not itself ship the third-party weights if the user downloads them separately; whether fine-tuned or distilled derivatives are "derivative works" of DINOv3/SAM/NVIDIA weights should be checked with counsel. A derivative of DINOv3/SAM-licensed weights must carry those licenses.
- NVIDIA Open Model License full text was not fetched (only the card statement); terms on derivative models and output ownership are unverified.

### Gaps
- NVIDIA Open Model License text, SAM 2.1 license, V-JEPA 2.1 license, S-Lab License text, and the licenses of FeatUp/LoftUp/JAFAR/AnyUp code were not read.

## Recommendation (keep / replace / add)

- KEEP C-RADIOv4-SO400M (best size/accuracy; already distills DINOv3-7B, SigLIP2, SAM3). Raise its input from 1/4 to roughly 1/2 to 1/3 of a 1080p frame when budget allows (up to its tested 2048 limit), cap the token count so 4K does not blow up memory.
- KEEP RVM small (only stateful Apache-2.0 video encoder); try RVM-B if VRAM allows. Optional experiment: V-JEPA 2.1 (license unverified).
- KEEP MoGe-2 ViT-B at a fixed token budget (1200 to 2500). No replacement justified; VDA-Small is the only licence-clean temporal alternative.
- ADD an image-guided feature upsampler (AnyUp first, fine-tuned on C-RADIOv4 features; JAFAR/UPLiFT as alternatives) fed with the full-resolution frame, to restore hair/fur edge detail. This follows the CineMatte design.
- ADD (optional, ablate first) optical flow from MEMFOF or SEA-RAFT at 1080p or below, as input channels and for alpha warping.
- ADD (optional) a hint channel from SAM 3/3.1 or user garbage matte for subject selection; avoid shipping MatAnyone 2 (non-commercial, needs first-frame mask).
- DO NOT ADD: raw DINOv3 (redundant, separate license), PE-spatial (weaker), Depth Anything 3 / VDA Base+Large (CC-BY-NC), DepthCrafter (not researched, likely heavy).
- All numbers on 4K cost need on-hardware measurement; none were published.
