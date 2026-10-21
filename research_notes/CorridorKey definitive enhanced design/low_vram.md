# Low-VRAM inference optimizations for an enhanced CorridorKey 2.0 (4-8 GB, RTX 30 series and newer, 1080p-4K), as of Oct 2026

Method note: about 20 search/fetch calls. Primary sources were fetched where possible (torchao docs, flash-attention README, RVM and BGMv2 READMEs, Accelerate docs, I-Segmenter paper). Several sub-questions returned no measured numbers; these are listed under Gaps rather than guessed. Arithmetic marked "own arithmetic" is derived by me, not sourced.

## 1. Weight quantization (INT8 / FP8 / INT4 / NVFP4) and what to keep high precision

### Takeaway
Weight-only INT8 is the only quantization path that is hardware-agnostic across RTX 30/40/50 and fits the precision constraint; FP8 needs SM89+ (Ada and newer, not Ampere), and NVFP4 compute needs Blackwell. Dense-prediction ViT quantization is documented as noticeably more fragile than classification (about 4-6 mIoU PTQ loss in the one integer-only segmentation study), so for matting keep softmax/attention logits, norms, residual stream, and the alpha head in high precision and quantize only large linear weights.

### Cited Findings
- torchao `Int8WeightOnlyConfig` is "int8 weight-only symmetric per-channel quantization" with bf16 activations; the docs list no hardware requirement. `Int4WeightOnlyConfig` also lists no hardware requirement, group sizes 256/128/64/32. | [torchao inference workflows](https://docs.pytorch.org/ao/main/workflows/inference.html)
- torchao float8 dynamic-activation config: "Requires CUDA ≥8.9, AMD MI350+, or Intel XPU" (i.e. SM89 Ada or newer; RTX 30 / Ampere excluded). `Float8WeightOnlyConfig` states the matmul is computed in original precision (so it saves memory but is not a speedup). | [torchao docs](https://docs.pytorch.org/ao/main/workflows/inference.html)
- torchao NVFP4 dynamic-activation config: "Requires NVIDIA SM100+ (Blackwell)", prototype status; on Llama-3.1-8B wikitext perplexity it is 8.4459 vs 7.3315 bf16, the largest accuracy drop in the docs' table. MXFP8 also lists SM100+ and prototype. The docs do not say whether consumer Blackwell (SM120, RTX 50) is covered; treat as unverified. | [torchao docs](https://docs.pytorch.org/ao/main/workflows/inference.html)
- Nunchaku (SVDQuant W4A4): Diffusers docs list sm_120 among supported architectures; NVFP4 checkpoints need Blackwell; older cards use INT4 variants (sm_80/86/89 reported). Self-reported: NVFP4 FLUX pipeline peak about 12 GB vs about 24 GB BF16 on an RTX 5090; SVDQuant 12B FLUX 3.6x smaller. These are diffusion-transformer results, not matting. | [Diffusers Nunchaku docs](https://huggingface.co/docs/diffusers/main/quantization/nunchaku), [HF blog](https://huggingface.co/blog/nunchaku-diffusers)
- ComfyUI: NVFP4 acceleration needs a PyTorch build with CUDA 13.0; otherwise sampling can be up to 2x slower than fp8; claimed about 2x over fp8/bf16 on RTX 50 when working. A community NVFP4 model card reports 9.4 percent relative RMS weight error vs bf16 versus 1.0 for an int8 variant, and notes dequantized-matmul fallback on non-Blackwell GPUs. Self-reported, unverified. | [Comfy blog](https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia), [MiniMax H3 NVFP4 card](https://huggingface.co/MATLOWAI/minimax-h3-nvfp4)
- FP8 PTQ study over 75 architectures (including segmentation): FP8 formats gave better accuracy and workload coverage than INT8 (92.64 percent vs 65.87 percent coverage); E3M4 slightly better on CV. Not ViT- or matting-specific. | [Efficient PTQ with FP8 Formats](https://arxiv.org/pdf/2309.14592)
- PTQ4ViT: ViT/DeiT/Swin at 8-bit lose under 0.5 percent top-1 on ImageNet classification. | [PTQ4ViT](https://github.com/hahnyuan/ptq4vit)
- Segmentation is more sensitive: I-Segmenter (integer-only ViT segmenter, ADE20K): PTQ drops 3.9-5.6 mIoU (Large: 52.29 FP32 to about 47.5 PTQ), QAT recovers to 48.74 (Large); Cityscapes PTQ drops about 5.1-6.2 mIoU. The paper states segmentation is sensitive because it needs dense pixel-level predictions. Sensitive ops: GELU (naive quantization collapsed to 0.92 mIoU in one case), softmax (computed in INT16 because small QK^T errors are amplified by exp), residuals kept INT16, biases INT32. This is integer-only (activations too), a harsher regime than weight-only. | [I-Segmenter](https://arxiv.org/html/2509.10334v1)
- DiT PTQ (closest to a "regression output from a big transformer"): PTQ4DiT and Q-DiT report near-lossless W8A8 on DiT-XL/2 and mild degradation at W4A8; difficulty traced to channel-wise outliers handled with SmoothQuant-like rebalancing. Notes that LLM methods (AWQ, LLM.int8) may not transfer directly. | [PTQ4DiT](https://arxiv.org/pdf/2405.16005), [Q-DiT](https://arxiv.org/html/2406.17343v2)
- GPTQ and AWQ are weight-only (memory traffic savings, math in fp16); SmoothQuant and FP8 quantize activations too. LLM-focused source. | [GeneralCompute overview](https://www.generalcompute.com/blog/quantization-for-inference-gptq-awq-smoothquant-fp8)

### Inferences
- Weights are about 5 GB (about 665M params at bf16 is about 1.3 GB; the rest is the three frozen foundation models and extras, per the brief). Weight-only INT8 roughly halves that and INT4 roughly quarters it (own arithmetic), which moves the problem from "does it fit" to "activations and attention", since weight-only keeps activations bf16.
- On RTX 30 the practical menu is weight-only INT8/INT4 with bf16 compute (dequantize-on-the-fly); FP8 or NVFP4 weight storage on Ampere would be dequantized and gives memory but not speed.
- Frozen foundation backbones (C-RADIOv4, MoGe-2, DINOv3) are the best quantization candidates: they are not trained against the matte loss, and if their features are precomputed to disk the weights need not be resident at all. The trainable matting head, stage-2 attention with d=1280, the final alpha projection, and the sparse edge refiner should stay bf16/fp16 or be quantization-aware trained.
- Keep in higher precision for matting (by analogy to I-Segmenter and DiT PTQ, not tested on matting): softmax and attention logits, LayerNorm/RMSNorm, residual stream, first patch-embed and last (alpha/foreground) projection layers, and any small conditioning/flow-fusion layers. Treat any W4 path as requiring a per-layer ablation measured on edge-region alpha error (SAD/grad/conn on hair), not global mIoU.

### Gaps
- No published PTQ or weight-only quantization results for any video matting model (MatAnyone, RVM, ViTMatte) were found.
- No source confirmed torchao NVFP4 weight-only on consumer SM120, or that INT4 kernels (tinygemm/Marlin-style) are fast on Ampere for vision shapes; I only found the config docs.
- First/last-layer sensitivity for ViTs is asserted in general literature but I found no ViT-specific citation in this pass.
- bitsandbytes, TensorRT-Model-Optimizer, and GPTQ/AWQ-on-vision were not fetched; no vision-specific numbers obtained.

## 2. Attention memory: FlashAttention versions, SDPA kernels, chunking, support on RTX 30/40/50

### Takeaway
Use PyTorch SDPA (flash / memory-efficient backends) or FlashAttention-2 on RTX 30/40; FA3 is Hopper-only and FA4 targets Hopper/data-center Blackwell, so neither is a dependable consumer path. RTX 50 (SM120) support for flash-attn is patchy as of the sources found, so SDPA with a fallback must be the default.

### Cited Findings
- FlashAttention-2: "Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100)"; fp16 and bf16 (bf16 needs Ampere+); head dim up to 256; head dim 256 backward works on consumer GPUs without dropout (>192 backward needs A100/H100 otherwise). | [flash-attention README](https://github.com/Dao-AILab/flash-attention)
- FlashAttention-3 (beta): "optimized for Hopper GPUs", requires H100/H800 and CUDA >= 12.3; FA4 (CuTeDSL): "optimized for Hopper and Blackwell GPUs (e.g. H100, B200)". SM120 not mentioned. | [flash-attention README](https://github.com/Dao-AILab/flash-attention)
- RTX 50 reports: building flash-attn 2.x from source for sm_120 crashes nvcc on backward kernels (CUDA 12.8/12.9, torch 2.9.1); an RTX PRO 6000 (sm_120) user hit an "invalid argument" kernel error; an unofficial gist argues FA4 cannot run on RTX 5090 (no TMEM) and recommends FA2 via Triton/flex_attention; a community repo bundles unmerged FA4 SM120 PRs. All community-sourced, may be outdated. | [FA issue 2361](https://github.com/Dao-AILab/flash-attention/issues/2361), [FA issue 1987](https://github.com/Dao-AILab/flash-attention/issues/1987), [gist](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739), [HF flash-attn-4-sm120](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120)
- Nanochat docs: FA3 on Hopper, PyTorch SDPA fallback on Ada, Blackwell, Ampere. | [nanochat flash attention](https://www.mintlify.com/karpathy/nanochat/architecture/flash-attention)
- Accelerate offload: weights offloaded to CPU are moved to GPU just before forward and freed after; "no pre-fetching (yet)"; disk offload "might be very slow" without NVMe. Relevant to attention-query chunking only insofar as it shows offload latency is not hidden by default. | [Accelerate big model inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)

### Inferences
- Flash-style kernels remove the N^2 score matrix, so memory is O(N) in tokens; the 80-frame stage-1 full space-time attention at /64 is fine on memory with a flash/mem-efficient kernel but its compute stays quadratic in tokens (own reasoning). Existing "chunk attention queries" is a manual equivalent when no fused kernel is available (e.g. odd head dims or masks).
- Recommended: `torch.nn.functional.scaled_dot_product_attention` with kernel auto-selection, FA2 optional on 30/40, and a tested fallback to the memory-efficient/math path on SM120. Do not depend on FA3/FA4.
- Swin/CSwin windowed attention in stage 2 is already memory-light; spend effort on stage 1 and any new temporal attention.

### Gaps
- The PyTorch SDPA backend documentation fetch returned only a redirect; I did not verify per-backend dtype/head-dim/mask constraints or SM120 behavior. xFormers was not researched.
- No measured peak-memory numbers for flash vs mem-efficient vs math attention at video-token counts.
- Current (Oct 2026) status of SM120 flash-attn wheels is unconfirmed; the issues found may be stale.

## 3. Recurrent / streaming designs vs windowed attention

### Takeaway
Fixed-size recurrent state or fixed-size memory banks give constant memory over arbitrary clip length; RVM (ConvGRU) demonstrates this at 4K on consumer cards, and MatAnyone/Cutie-style memory shows the quality benefit of consistent memory but adds its own cost. Direct evidence for state-space or linear attention in matting was not found.

### Cited Findings
- RVM: recurrent states carried frame to frame (`rec = [None]*4`); benchmark (batch 1, tensor throughput only): RTX 3090 FP16 172 FPS HD / 154 FPS 4K; RTX 2060 Super FP16 134 FPS HD / 108 FPS 4K; GTX 1080 Ti FP32 104 HD / 74 4K; HD uses downsample_ratio 0.25, 4K uses 0.125. The README gives no VRAM numbers. The intro text says 76 FPS 4K on the 1080 Ti while the table says 74. | [RVM README](https://github.com/PeterL1n/RobustVideoMatting)
- MatAnyone: memory-based consistent propagation; the original was trained on short (8-frame) sequences with only local-window frames stored in memory; MatAnyone 2 adds long-range reference frames "without additional memory overhead" and notes widening the training window would sharply raise memory use. MatAnyone is listed at 35.3M params. | [MatAnyone 2 paper](https://arxiv.org/pdf/2512.11782), [MatAnyone HF](https://huggingface.co/peiqingyang/matanyone)
- Fixed-size memory in related VOS: RDE-VOS and the Space-Time Recurrent Memory Network keep a constant-size memory so cost does not grow with video length. | [STRM](https://arxiv.org/pdf/2109.06474)
- SAM2 per-frame memory when offloaded is quoted as about 0.025 GB per 1080p frame saved (Det-SAM2 figure relayed by a forum write-up); the same write-up names `video_storage_device="cpu"` and `inference_state_device="cpu"` in the Transformers API. Secondary source, unverified. | [sam2_oom write-up](https://huggingface.co/datasets/John6666/forum1/blob/main/sam2_oom.md)

### Inferences
- A recurrent memory replacing 80-frame windows removes the temporal-window memory term entirely: peak VRAM becomes independent of clip length and bounded by one frame's activations plus a state of fixed size. It also removes the "1-2 frame window" quality compromise now forced by low-VRAM mode.
- Memory-bank approaches (SAM2/Cutie/MatAnyone) are only constant if the bank is capped; the cap and its resolution should be set per VRAM tier. Store state at /16 or /64 latent resolution, not full-res (own recommendation).
- State to carry at low precision is a risk: recurrent state errors accumulate over thousands of frames, so keep recurrent state in fp16/fp32 even if weights are quantized.

### Gaps
- No evidence found for linear attention / SSM (Mamba-style) matting; no measured memory for MatAnyone or Cutie at 1080p or 4K (the HF demo only says inputs are resized to 1080p because of GPU memory limits, per the [MatAnyone demo](https://peiqingyang-matanyone.hf.space/)).
- Quality of recurrent vs windowed attention for fine hair/edge detail is not established here.

## 4. High-res tiling and sparse edge refinement

### Takeaway
Run the heavy transformer at reduced resolution and spend full-resolution compute only on edge patches (BGMv2-style); this is the standard way 4K matting reaches real-time on consumer GPUs. Tile overlap/seam specifics were not sourced.

### Cited Findings
- BGMv2: base network at low resolution plus a learned refinement stage; reports "4K 30fps and HD 60fps on an Nvidia RTX 2080 TI GPU". The README does not describe the refiner internals or memory; details are in the paper. | [BackgroundMattingV2 README](https://github.com/PeterL1n/BackgroundMattingV2)
- RVM runs HD at downsample_ratio 0.25 and 4K at 0.125 (own arithmetic: 1920x0.25 and 3840x0.125 both give a 480-pixel-wide backbone input) and still produces full-resolution alpha through a deep guided filter style decoder. | [RVM README](https://github.com/PeterL1n/RobustVideoMatting)

### Inferences
- Sparse refiner design for the enhanced model: coarse alpha plus uncertainty/gradient map selects top-K patches (for example 8x8 or 16x16 patches in the unknown band), refiner runs only on those with a fixed K cap so peak memory is constant per frame regardless of resolution.
- Cap K per VRAM tier; this makes the 4K cost approximately that of 1080p.
- Tiling the dense stage-2 attention is a fallback only; seams and global-context loss (the model relies on full-frame attention at /64 in stage 1) argue for downsample-and-refine instead of tile-and-blend. Where tiling is unavoidable, use at least the receptive field in overlap and blend only in the overlap with cosine weights (own recommendation, unsourced).

### Gaps
- BGMv2 refiner patch sizes, thresholds, and memory were not retrieved (README lacks them).
- No source on tile-overlap seam handling for transformers was found.

## 5. Distillation, frozen-backbone feature caching, low-res backbones plus feature upsamplers

### Takeaway
Caching frozen-backbone features (already in CorridorKey's low-VRAM mode) removes backbone weights and activations from the GPU entirely; running backbones at lower resolution and upsampling features with a learned guided upsampler is a published alternative, but upsamplers carry their own cost and OOM limits at large token counts.

### Cited Findings
- Feature upsamplers: AnyUp (encoder-agnostic, takes low-res features plus high-res RGB guidance), JAFAR, LoftUp, FeatUp, LiFT. Wall-clock per image from one 2026 paper: AnyUp about 61.6 ms, JAFAR 59.4 ms, LoftUp 134.7 ms, FeatUp 65.1 ms, LiFT 6.8 ms. A separate paper (UPLiFT) reports LoftUp, AnyUp, and JAFAR ran out of memory at its largest token counts. | [AnyUp](https://arxiv.org/html/2510.12764v2), [JAFAR](https://arxiv.org/pdf/2506.11136v2), [alphaxiv 2605.17472](https://www.alphaxiv.org/abs/2605.17472), [UPLiFT](https://arxiv.org/pdf/2601.17950)
- DINOv3 paper: ViT-L about 300M params, about 163 GFLOPs at 256x256 and about 721 GFLOPs at 512x512. | [DINOv3](https://arxiv.org/pdf/2508.10104)
- DINOv3 ViT-B/16 bf16 peak memory about 359-388 MB on the grid sizes tested in CanViT. | [CanViT](https://arxiv.org/pdf/2603.22570)
- Depth Anything V2-Large is 335.3M params; an fp16 checkpoint is 640 MB; default inference input 518; a ComfyUI page claims fp16 is half the VRAM of fp32 with 0.02 percent quality difference (unsourced vendor-node claim). | [DA-V2 Large HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Large), [ComfyUI node](https://comfy.icu/node/TS_ImageDepth)
- A lightweight distillation of SAM 3 and DINOv3 for edge deployment exists (livestock monitoring); not matting. | [arXiv 2604.27128](https://arxiv.org/pdf/2604.27128)

### Inferences
- DINOv3-L weights are about 0.6 GB in bf16 (own arithmetic: 300M x 2 bytes). At 1080p with patch 16 there are 120x68 = 8160 tokens, at 4K 240x135 = 32,400 tokens (own arithmetic); full global attention over 32k tokens needs a fused kernel and is the activation-memory hot spot, so feed the backbone a downscaled image (about 1024-1536 long side) and upsample features.
- Feature caching: with a disk memmap, fp16 features at /16 with C channels cost H/16 x W/16 x C x 2 bytes per frame (own formula; for C=1024 at 1080p about 16.7 MB per frame per backbone, own arithmetic). Streaming these from NVMe at 24 fps is about 0.4 GB/s per backbone, feasible on NVMe.
- Distilling a teacher into a smaller student (reducing d=1280 or depth in stage 2, or a lightweight student backbone) is the only change that cuts both VRAM and speed on Ampere without precision risk, but needs training; I found no matting-specific distillation numbers.

### Gaps
- No measured peak VRAM for DINOv3-L, SAM2-L, or MatAnyone at 1080p/4K; searches only returned weights and FLOPs. These must be measured with `torch.cuda.max_memory_allocated()`.
- Which upsampler preserves alpha-edge-quality features is untested; their accuracy is reported on segmentation/depth probes, not matting.

## 6. Offload, torch.compile, CUDA graphs, TensorRT, ONNX

### Takeaway
Sequential CPU offload works but is un-prefetched and latency-bound; it is a last resort for the stage-swap design. torch.compile / CUDA graphs help batch-1 latency with fixed shapes; TensorRT supports FP16/BF16/INT8/FP8 but adds export complexity.

### Cited Findings
- Accelerate: offloaded weights are moved to GPU just before the forward pass and released after; "no pre-fetching"; `cpu_offload_with_hook` is "more performant but less memory saving" and suits pipelines that run a module in a loop; disk offload slow without NVMe; CUDA context itself takes about 1-2 GB of the card. | [Accelerate](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)
- Torch-TensorRT: precision is the largest speed lever; BF16 requires Ampere or newer; CUDA graphs need fixed shapes and help most for batch-1 and many-small-kernel models (attention, layer norm). | [Torch-TensorRT performance tuning](https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html)
- One Collabora benchmark found torch.compile easier and faster than TensorRT on LLMs (not ViTs). | [Collabora](https://collabora.com/news-and-blog/blog/2024/12/19/faster-inference-torch.compile-vs-tensorrt)

### Inferences
- On a 4 GB card, the CUDA context (1-2 GB per Accelerate) plus allocator overhead leaves roughly 2-3 GB usable, so a design targeting "4 GB" must budget for that (inference from the Accelerate note).
- Use a manual double-buffered stage swap (copy next stage's weights on a side CUDA stream with pinned memory while the current stage computes) instead of Accelerate's hook-based offload, which has no prefetch. Stage 1 then stage 2 are coarse-grained, so swap cost is amortized across the whole window or recurrent step.
- Variable window length and variable resolution fight CUDA graphs; use graphs only on fixed-shape per-frame modules (stage 2, refiner at fixed K).

### Gaps
- No ViT-specific torch.compile/TensorRT speedups found; no measured offload slowdown for this model class.
- ONNX export viability for Swin-3D / CSwin / custom attention is unverified.

## 7. Precision of the alpha output (bf16 vs fp16 vs fp8)

### Takeaway
The matte-precision concern is real and solvable: use bf16 for the body but compute the final alpha projection in fp32 (or fp16), and never quantize the last layers. I found no published study of fp8/bf16 vs fp16 on matting output.

### Cited Findings
- Torch-TensorRT notes BF16 has better numerical range than FP16 and requires Ampere+. | [Torch-TensorRT](https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html)
- RVM runs FP16 inference on Turing and newer for matting at 4K (GTX 1080 Ti ran FP32 as pre-Turing lacks FP16 inference), so fp16 matting works in production. | [RVM README](https://github.com/PeterL1n/RobustVideoMatting)

### Inferences
- bf16 has 8 bits of significand (7 stored), so values in [0.5,1) quantize to 2^-8 = 1/256, matching the brief; fp16 has 11 bits (10 stored), giving 2^-11 steps in [0.5,1) (own arithmetic from format definitions). Hence: keep the body in bf16 for range, but run the alpha head and final blend/clamp in fp32 (cheap: a few channels at output resolution), or use fp16 for the sparse refiner where activations are bounded.
- fp8 for activations/weights in the alpha path would give coarser steps (3 mantissa bits for E4M3); do not use it there.

### Gaps
- No quantitative study of bf16 vs fp16 alpha error (SAD/MSE/Grad) was found; recommend measuring on the validation set.

## Recommended low-VRAM inference design for the enhanced model

(This section synthesizes the above; items without citations are my recommendations or arithmetic.)

1. Offline/first pass: run all frozen backbones (DINOv3, C-RADIOv4, MoGe-2, RVM-small, optical flow) once, at reduced input resolution (long side about 1024-1536), store fp16 features at /16 (and /64 pooled) to NVMe memmaps. The backbones never share the GPU with the matting stages, so their weights (and the 4K token-count activations) leave the VRAM budget. Optionally INT8 weight-only these backbones (torchao Int8WeightOnly, no hardware requirement) since they only run in this pass.
2. Stage 1 (coarse, /64 space-time attention): run as a recurrent memory step with a capped fixed-size memory-token bank instead of an 80-frame window; SDPA flash/mem-efficient attention with query chunking only as fallback; weights INT8 weight-only on Ampere/Ada (dequant to bf16 for compute); FP8 weight storage only where SM89+ is detected; no NVFP4 in the default path (Blackwell-only, prototype, higher weight error).
3. Stage 2 (/16, d=1280): per-frame windowed attention (Swin/CSwin), temporal attention restricted to the recurrent state or 1-2 neighboring frames; keep attention logits/softmax, norms, residuals bf16; MLP weights INT8 with token chunking (existing).
4. Edge refiner: coarse alpha + gradient/uncertainty selects at most K patches at full resolution; refiner in fp16; K capped by VRAM tier so 4K cost is bounded.
5. Output head: final alpha projection and composite in fp32; never quantize first/last layers or the recurrent state.
6. Execution: manual two-slot pinned-memory weight swap between stage 1 and stage 2 on a side stream for the 4 GB tier; both stages resident for 8 GB. CUDA graphs only on fixed-shape per-frame modules. Reserve about 1-2 GB for the CUDA context when sizing tiers.
7. Tiers: 4 GB = swap stages + INT8 weights + cached features + K-capped refiner; 6-8 GB = both stages resident, INT8 or bf16 weights, longer memory bank; FP8 weight storage on Ada/Blackwell as an option; NVFP4 only as an opt-in experimental path.
8. Validation gate before shipping any quantization: compare against the bf16 reference on edge-band alpha error and temporal flicker, per layer group, not global metrics.

## Flagged as unverified or weakly sourced
- All NVFP4/Nunchaku/ComfyUI speed and VRAM figures are vendor or community self-reports on diffusion models.
- SM120 flash-attn status relies on GitHub issues and a gist that may be outdated.
- SAM2 per-frame offload figure (0.025 GB per 1080p frame) is relayed via a forum write-up.
- Measured VRAM for DINOv3-L, SAM2-L, MatAnyone, DA-V2-L at 1080p/4K was NOT found; do not treat any such number as established.
