# Solve the screen, learn only the ambiguity

The definitive design keeps CorridorKey 2.0's shipped 665M-parameter two-stage trunk and its three frozen foundation models (C-RADIOv4-SO400M, MoGe-2 ViT-B, DeepMind RVM small), and wraps it in a **physics layer that does the part of keying the screen makes solvable**. The physics layer has four parts. A per-shot **backing memory** rebuilds the clean screen, its exposure gain and the subject's shadow. A **full-resolution projection head** reads alpha straight from raw pixels wherever keying signal-to-noise allows and hands off to the network wherever it does not. A **calibrated confidence output** routes a sparse full-resolution refiner. A **degradation side channel** carries a screen-calibrated noise map, codec QP and native chroma planes. Around that, the design switches on the trunk's own dormant /64 recurrence and /4 refine head, gives the last two stage-2 blocks three-frame attention, and runs the whole output tail in fp32. The trunk grows only about 3% (to roughly 690M trainable parameters), so the low-VRAM story survives. The 4 GB tier is budgeted (by arithmetic, to be confirmed with a memory-capped run) to key 1080p with INT8 weight-only foundations and stage 1, a bf16 stage 2 and an fp32 tail, and to key 4K by running the trunk at half resolution while the projection head and refiner work at native 4K. Training is a roughly 30 GPU-day fine-tune on the one available A6000, from the shipped checkpoint. The data is Blender light-group renders (exact spill, unlimited backing recolours, paired green and blue versions of the same frame), composites over our stage simulator, real static triangulation captures, and our own real green-screen footage admitted only through a green-screen-specific learned critic. The strongest evidence for this direction is our own preliminary measurement: inference-only fixes to precision, decoding and foreground recovery already cut composite error 8% and "holes" 70% without retraining, while the compressed and low-light shots barely moved (-1 to -3%). The trunk's understanding is not the bottleneck. The output path and the input's trustworthiness are, and that is where this design spends its effort. Every risky decision below has a named kill test, most of which cost an afternoon and no training.

## The decided design: CorridorKey 2.0 Enhanced in one spec

The license requires derivatives to stay free, carry the same CC BY-NC-SA 4.0 variant license and keep the CorridorKey name ([CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)), so the model is **CorridorKey 2.0 Enhanced**. The diagram shows the full inference path. Stage boxes marked "kept" retain the shipped checkpoint's weights as initialization; everything marked "new" or "zero-init" is output-neutral at insertion, so training starts from exactly the released model's behaviour.

```
INPUT   ffmpeg decode, 16-bit, colour-tag aware; frames held on CPU as uint16 / half (never bf16)
        + native Y (/1) and CbCr (/2 or /1) planes when the source is 4:2:0 / 4:2:2
        + per-block QP and frame type where the decoder exports them (else a constant from bitrate)
   |
   v
[A] BACKING MEMORY  (no learned weights; per shot)                                      NEW
    first-pass alpha from stage-1 aux head (stride 8) at low res
    screen registration: identity (locked-off) or per-plane homography, up to 2 planes (wall + floor)
    robust plate P(x) = (1-alpha)^k-weighted median over revealed frames, with per-pixel count
    global gain g_t fitted on always-visible screen; noise model sigma^2(level) fitted on screen pixels
    outputs: B_t = g_t * P, confidence c(x) (observed vs pull-push filled), sigma map,
             screen-plane signed distance (RANSAC planes on MoGe-2 points)
   |
   v
[B] FROZEN FOUNDATIONS  bf16 (INT8 weight-only on the 4 GB tier)                       KEPT, input res CHANGED
    C-RADIOv4-SO400M ~431M | MoGe-2 ViT-B ~104M | DeepMind RVM-S ~34M (recurrent, chronological)
    input = 1/2 source res, long side capped per tier (1920 / 1280 / 960)
    tokens 2x2 average-pooled back to the shipped /64 grid for stage 1
    unpooled C-RADIO grid also fed to stage 2 through a zero-init 1x1 projection        NEW
   |
   v
[C] STAGE 1  463.6M  /64 full space-time attention + /16 Swin-3D (full-T windows)      KEPT
    + side channels into the stem: asinh(B_t), c, sigma, QP, screen distance (zero-init)  NEW
    + 2 FullResolutionRecurrentMemorySite blocks at /64 (already in the code, zero-init) ENABLED
      trained with random window lengths, so short windows carry long context
    --> baton (d=512, /16)
   |
   v
[D] STAGE 2  201.8M  /16 per-frame, d=1280, 8 mixed blocks + 2 Swin blocks             KEPT
    + final 2 blocks: each window also attends to the same window at t-1 and t+1
      (reused qkv projections, per-head zero-init gate, temporal offset embedding)       NEW
    --> /16 smooth heads: F_loc (3), spill a,b (2), log shadow s_t (1), fusion logits (2) NEW
    --> alpha_net, F_net through the shipped linear Depatchify x16 (now the coarse path) KEPT
   |
   v
[E] OUTPUT TAIL  all fp32, ~1.5M params                                                 NEW / ENABLED
    refine4 head ON: /4 ConvNeXt, 4 blocks, width 96, reads exact RGB pixel-unshuffle
      + Laplace log-b uncertainty per pixel
    physics projection at /1:
      alpha_phys = <I - B_t, F_loc - B_t> / (||F_loc - B_t||^2 + lambda * sigma^2)
      w = sigmoid(logit + log(||F_loc - B_t||^2 / sigma^2))
      alpha = w * alpha_phys + (1 - w) * alpha_net      (then clip to [0,1])
    sparse /1 refiner on the top-K 32x32 tiles ranked by uncertainty (K capped per tier)
    FG = SNR-weighted blend of plate-recovered F (spill subspace removed) and F_net
   |
   v
OUTPUTS  alpha (fp32, linear) | straight FG (despilled, scene-linear) | confidence | shadow pass s_t | clean plate P
OPTIONAL TTT-Key "final quality" mode: per-shot online LoRA (rank 8) on stage 2 + tail   (12+ GB tier)
```

| Component | Size | Status | Precision at inference |
|---|---|---|---|
| C-RADIOv4-SO400M | ~431M, frozen | Kept; input raised from 1/4 to 1/2 source res, capped | bf16; INT8 weight-only on 4 GB |
| MoGe-2 ViT-B | ~104M, frozen | Kept; token budget ~2,000 (inside its 1,200 to 2,500 range) | bf16; INT8 on 4 GB |
| DeepMind RVM small | ~34M, frozen | Kept, unchanged | bf16; state in fp16/fp32 |
| Stage 1 | 463.6M (274.2M is the /64 body) | Kept + side-channel stem + 2 recurrent sites | bf16; INT8 weight-only for /64 body on 4 GB |
| Stage 2 | 201.8M | Kept + 3-frame attention in final 2 blocks + /16 smooth heads | bf16 always |
| refine4 /4 head | ~1.1M | Enabled (was `refine4_blocks=0`) | fp32 |
| Physics projection + fusion | ~0 learned params | New | fp32 |
| Sparse /1 refiner | ≤0.5M | New | fp32 (fp16 permitted on 4 GB after gate) |
| Side-channel branches | ~5M | New, zero-init | bf16 input projections |
| **Total trainable** | **~690M** | +~3.5% over shipped | |

Parameter splits come from reading the shipped `corridorkey_v2.safetensors` header (stage 1 463.61M, stage 2 201.80M, depatchify 5.24M). Sizes for the frozen models come from their model cards ([C-RADIOv4](https://huggingface.co/nvidia/C-RADIOv4-SO400M), [MoGe](https://github.com/microsoft/MoGe), [representations4d](https://github.com/google-deepmind/representations4d)). Sizes for the new modules are design budgets, not measurements.

### What stays, and why it earns its place

The three frozen foundations stay because nothing on the market dominates them for this job. **C-RADIOv4-SO400M already distills DINOv3-7B, SigLIP2-g and SAM3** and scores 55.14 ADE20k mIoU, within 0.8 of the 6.7B DINOv3 at about 6% of the parameters, while C-RADIOv4-H buys only 0.06 more for 50% more weight ([C-RADIOv4 card](https://huggingface.co/nvidia/C-RADIOv4-SO400M); [DINOv3 card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)). Adding a raw DINOv3 or SAM 3 encoder would mostly duplicate knowledge C-RADIO already carries. DeepMind's RVM is the only stateful, Apache-licensed dense video encoder in the set, with linear-cost long-horizon propagation ([representations4d](https://github.com/google-deepmind/representations4d)). MoGe-2 ViT-B is small, MIT-licensed and accepts a token budget, so its cost is independent of frame size ([MoGe](https://github.com/microsoft/MoGe)). Its job in the enhanced design grows: besides depth context, its point maps now fit the screen planes used by backing registration and by screen-membership subject selection.

The trunk stays because our measurements say it is not the problem. Our preliminary benchmark numbers (internal, being re-run with corrected ground truth) show three things. Fixing decode precision, the fp32 output tail and plate-recovered foreground already moved mean composite error -8% and composite PSNR +1.0 dB. The gain was -40% on clean and HDR plates. The bf16 precision fix alone removed 70% of "holes". That pattern is what you would see if a capable trunk were being throttled by its last layer and its input path. Training from the shipped checkpoint also keeps everything Corridor learned from its in-house renders, which we cannot reproduce.

### What changes, and the single reason for each

The **output path** changes most, because the shipped one is structurally weak. Stage 2 ends in one linear PixelShuffle x16, which must encode a whole 16x16 block of alpha and colour from one 1280-d token in a single affine map. Every strong detail method instead adds a cheap high-resolution path that reads raw pixels. ViTMatte bolts "simple lightweight convolutions" onto a plain ViT, and BGMv2 refines only selected patches at full resolution ([ViTMatte](https://arxiv.org/abs/2305.15272); [BGMv2](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)). The shipped code already contains a zero-initialized /4 ConvNeXt refine head that "is an exact no-op until it learns". We switch it on, and add the analytic /1 projection and the sparse refiner beneath it.

**Precision** changes because bf16 cannot represent near-opaque alpha. bf16 carries 8 significant bits, so values in [0.5, 1) sit 1/256 apart and the largest value below 1 is 0.996. fp16 spaces them 1/2048 apart. PyTorch's own AMP guidance says reductions need float32's range ([torch.amp](https://docs.pytorch.org/docs/stable/amp.md)). The whole tail (refine4, projection, refiner, clip and composite) runs in fp32, as do all losses. Input frames are never stored in bf16 again.

**Foundation input resolution** changes because at 1080p the shipped 1/4 scaling gives C-RADIO a 480x270 image, about 30x17 patches, which is very coarse for subject decisions near thin structures. Raising it to 1/2 stays inside C-RADIOv4's tested 256 to 2048 px range ([C-RADIOv4 card](https://huggingface.co/nvidia/C-RADIOv4-SO400M)). Pooling the extra tokens back to the shipped /64 grid leaves stage-1 cost unchanged. At 4K, 1/2 resolution means 1920 px long side on big cards; the 4 GB tier caps it at 960, which equals the shipped behaviour. Training randomizes this scale among 1/4, 1/3 and 1/2 so one checkpoint serves every tier.

**Temporal processing** changes in two places, because the shipped stage 2 is explicitly per-frame ("No temporal processing here: stage 1 has done all temporal work", per its docstring) while fine-detail flicker is born in stage 2. Memory-propagation methods beat recurrent RVM-class matting on dtSSD by 10 to 35% at 1080p. On VideoMatte 1080p, MatAnyone 2 scores 1.15 against Robust Video Matting-Large's 1.78 ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)). We do not import a memory bank. The two changes are these. First, the final two stage-2 Swin blocks let each 8x8-token window also attend to the same window in the neighbouring frames. Inter-frame motion at /16 is usually under one token, so no optical flow is needed, and reusing the existing qkv projections keeps the addition at nearly zero parameters. Second, stage 1's dormant `FullResolutionRecurrentMemorySite` blocks (present in the code, empty in the shipped config, whose metadata says `pre_recurrence: true`) are enabled and trained. Small cards then get long context through short windows plus a carried state. The README admits the current Hann carry "can drift on long crowded shots" and that small windows weaken consistency ([CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)).

## Physics carries the edge; the network carries what physics cannot

The design's core idea comes from a 30-year-old result. Smith and Blinn proved single-backing matting is "unsolvable as generally practiced", but it becomes fully solvable with two backings ([Smith & Blinn 1996](https://alvyray.com/Papers/CG/blusig96.pdf)). With the backing B known, each pixel has exactly one unknown on a known colour line. Given a foreground colour estimate F, the least-squares alpha is the projection of I - B onto F - B. Its noise is sigma divided by ||F - B||, a quantity worth calling the local **keying SNR**. Illustrative linear values make the stakes concrete. Brown hair against a typical green gives ||F - B|| ≈ 0.29, so a clean camera (sigma ≈ 0.005) pins alpha to about ±0.017, while low light (sigma ≈ 0.02) loosens it to ±0.07, and green wardrobe (||F - B|| ≈ 0.10) is three times worse again. These numbers are worked examples, not benchmark measurements; the real sigma per shot will come from the backing memory's noise fit.

The same arithmetic explains our preliminary results shot by shot. The precision fix helped clean and HDR plates most because they have high keying SNR and were being limited by quantization. Compressed and low-light shots moved only 1 to 3% because they are noise-limited, so no amount of output precision helps. Plate-recovered foreground made smoke 1% worse for a reason the algebra predicts: with F = B + (I - B)/alpha, foreground noise scales as sigma/alpha and explodes in thin smoke. Glass did not move (edge error 16.7) because refraction breaks the compositing equation itself. The design follows directly. It improves B, measures sigma, reads alpha physically where SNR is high, and trusts the network elsewhere.

### The eleven techniques that are in

| Technique | Closest prior art | What is new here | Bench shots it targets | Confidence |
|---|---|---|---|---|
| Backing memory (plate, gain, shadow) | Chuang 2002 filled background from the nearest frame showing it, and found plate errors "can seriously degrade" mattes ([Chuang et al.](https://grail.cs.washington.edu/projects/digital-matting/papers/sig2002.pdf)); BGMv2 needs a captured plate | Automatic, per shot, factorised into gain x shadow x plate, used as learned conditioning | c02, c03, c04, c07, c09, subject selection | High |
| /1 physics projection + SNR fusion | FBA's projection fusion "reliably" improved results ([FBA](https://ar5iv.labs.arxiv.org/html/2003.07711)) | Known B, deliberately smooth F_loc, noise-aware weight, used as the /1 path of a /16 transformer | c05, c06, c12, c14, c15, c07 | Medium-high |
| Confidence head + sparse /1 refiner | BGMv2: refining 5 to 10% of pixels gets most of the gain ([BGMv2](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)) | Calibrated Laplace uncertainty shipped as an artist pass and reused as fusion and TTT gate | c12, c14, c15, c11 edges | Medium |
| Light-group backing algebra + paired green/blue loss | Cycles light groups recombine lights "without having to re-render" ([Blender manual](https://docs.blender.org/manual/en/latest/render/layers/passes.html)); dual-backdrop matting improved spill handling ([Hachmann & Rosenhahn](https://arxiv.org/html/2306.02954)) | Exact spill, bounce and shadow for unlimited backing colours from one render; paired consistency kills the "green means background" shortcut | c08, c10, c02, spill everywhere | High |
| Degradation side channel | Compression-aware modules beat plain training on compressed frames ([COMISR](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.pdf)); codec priors help VSR ([CIAF](https://arxiv.org/html/2210.08229), [CDA-VSR](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Compressed-Domain-Aware_Online_Video_Super-Resolution_CVPR_2026_paper.html)); noise-level maps ([FFDNet](https://arxiv.org/abs/1710.04026)) | Noise calibrated on the screen itself; native 4:2:0 chroma into the /1 path so the model knows which chroma detail is interpolated | c03, c04, c09 | Medium |
| Geometry-predicted spill subspace | Spill compensation is the "missing link" to production keying ([Hachmann & Rosenhahn](https://arxiv.org/html/2306.02954)) | Spill = a·B̂ + b·(F ⊙ B̂) with two smooth scalars, supervised exactly by light groups, replacing the single k·B̂ despill | c10, c02, c08 | Medium |
| SNR-weighted foreground blend | Derived from the projection algebra | Plate F trusted where alpha·‖F−B‖/sigma is high, network F elsewhere; fixes the smoke regression | c07, c13, all FG error | Medium-high |
| Screen membership channel | MatAnyone calls auxiliary-free matting "under-defined" ([MatAnyone](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.pdf)) | Signed distance to RANSAC-fit screen planes: markers on the screen are background whatever their colour, from frame 1 | c02 markers, subject selection | Medium (narrow) |
| Stage-1 recurrence + stage-2 3-frame attention | Memory methods win dtSSD ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)); trajectory attention beats joint space-time attention ([Motionformer](https://arxiv.org/html/2106.05392v2)) | Uses the shipped zero-init recurrent site; window-local temporal K/V without flow | c05, c12, c14, c15 shimmer | Medium |
| TTT-Key (optional final-quality mode) | Online video TTT: +45% and +66% relative on instance and panoptic segmentation; one step per frame captures most of it ([Wang et al.](https://video-ttt.github.io/)); big gains only under severe shift ([Azimi et al.](https://sslneurips21.github.io/files/CameraReady/SSL_TTA.pdf)) | Degeneracy-aware loss: reconstruction is informative only because F_loc is forced smooth | c04, c09, c03 | Medium-low |
| Failure-driven render curriculum | Automatic Domain Randomization widens each parameter range when performance passes a threshold ([OpenAI ADR](https://arxiv.org/html/1910.07113)) | Axes are keying failure modes: unevenness, marker density, smoke density, IOR and tint, thread width, CRF, ISO, blur, wardrobe hue distance | All synthetic axes | Medium |

**The backing memory** is the highest-value addition, because every error in B enters alpha with weight (1 - alpha). That makes B the dominant error source in exactly the semi-transparent band compositors care about. The mechanism has five steps. A cheap first pass gives alpha. Frames are registered to screen coordinates, using identity for locked-off shots and a per-plane homography fitted on alpha < 0.02 pixels for moving ones (tracking markers exist precisely to make this solvable). A robust plate accumulates from every revealed pixel. The current backing is factored as B_t = g_t · s_t · P, with a global gain g_t fitted on always-visible screen and a smooth multiplicative field s_t that the network predicts. B_t, the confidence that each plate pixel was observed rather than interpolated, and the sigma map then enter both stages through zero-initialized branches. Plate noise falls as sigma/√K over K revealing frames, and markers or seams behind the subject become known instead of guessed. The shadow field s_t ships as a compositor deliverable. Our Blender farm already renders the clean backing with the subject's real shadow, so s_t has exact supervision. Inference cost is one low-res pre-pass plus a float plate buffer per shot.

**The projection head** is what attacks 4K hair and the sub-pixel web. Stage 2 predicts, at /16, only what is genuinely smooth: the local foreground colour F_loc, its uncertainty and a fusion logit. The fp32 tail computes alpha_phys at /1 from the raw pixel and blends it with the network alpha using a weight anchored to log(‖F_loc − B_t‖²/sigma²). It costs a handful of elementwise operations per pixel. It is exact under motion blur and defocus against a smooth backing, because blur is a linear mix of alpha-weighted colours, so it also serves c05 and c06. It introduces no temporal instability of its own: its shimmer can only come from input noise. Real-time green-screen keyers in the literature already pair a deep prior with a per-pixel linear model ([Li et al. 2021](https://www.mdpi.com/2073-8994/13/8/1454)). What is new is a known B, a noise-aware weight, and placing the projection beneath a 665M space-time transformer.

**The degradation side channel** exists because a decoder's upsampled RGB presents interpolated 4:2:0 chroma as if it were per-pixel detail, and keying is chroma-driven. COMISR found that training on compressed frames alone "brings only modest improvement" ([COMISR](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.pdf)), which matches our -1 to -3% on the codec shots. The decided form has three parts. A per-shot noise map is fitted on confident-background screen pixels (the backing is a giant flat calibration chart). A per-block QP and frame-type map is added. Native Y at /1 and CbCr at native resolution go only into the /1 path (refine4, projection, refiner), so the trunk is untouched. All side inputs are randomly dropped in training, so ProRes, EXR and image inputs without them keep working.

### What is out, and what would bring it back

| Rejected | Why | What would reopen it |
|---|---|---|
| Video-diffusion prior in the model (GVM, VideoMaMa class) | GVM costs 0.32 to 0.90 s per frame on an A100 against RVM's 0.01 to 0.02 s ([GVM](https://arxiv.org/html/2508.07905v1)), trails MatAnyone 2 on all four metrics in that paper's tables ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)), trains at 704 to 1024 px, and carries Stability Community or academic-only licences ([VideoMaMa repo](https://github.com/cvlab-kaist/VideoMaMa); [GVM repo](https://github.com/aim-uofa/GVM)) | A distilled one-step prior beating our model on c07 at under 2x cost |
| Diffusion teacher distillation | Teacher licences taint labels; ~960 px cap is useless for 4K hair; generative hallucination | Teacher with a permissive licence and filtered labels beating our own outputs on c02/c07 |
| Optical flow input or flow-warped losses (SEA-RAFT, MEMFOF) | Flow is unreliable exactly on translucent pixels; a flow-warped training loss hurt in practice ([Zhang et al.](https://ar5iv.labs.arxiv.org/html/2105.11427)) | Post-hoc flow-guided median beats the trained model's dtSSD by >10% on hair shots |
| Passive motion triangulation | Useful only over high-contrast backing (markers); needs sub-pixel strand correspondence | Oracle-flow triangulation beats the model over markers in a 2-hour render test |
| RGB transmission and refraction head for glass | Narrow, and red/blue transmission against green is observed with several times more noise; GT definition for glass decides whether it can even score | Per-channel transmission fit to c11's GT composite clearly beats the best scalar fit |
| Extra backbones (raw DINOv3, SAM 3 per frame, PE-spatial, V-JEPA 2.1) | Redundant with C-RADIOv4's distilled teachers; PE-spatial scores lower (G/14 ADE20k 49.3) ([perception_models](https://github.com/facebookresearch/perception_models)); V-JEPA 2.1 has no dense segmentation evidence ([arXiv](https://arxiv.org/abs/2603.14482)) | A probe showing C-RADIO features fail subject selection where another encoder succeeds |
| Temporal depth (Video Depth Anything Base/Large, Depth Anything 3) | CC-BY-NC weights and no keying-specific evidence; RVM already supplies temporal state ([VDA](https://github.com/DepthAnything/Video-Depth-Anything)) | MoGe-2 flicker measurably leaking into alpha |
| Third-party feature upsamplers (AnyUp, JAFAR, LoftUp) | Never tested on C-RADIO; some run out of memory at large token counts ([UPLiFT](https://arxiv.org/pdf/2601.17950)); the projection head attacks edges directly | Projection head kill test fails on hair |
| MatAnyone-style memory bank | Recurrent site already in code, constant memory, zero-init retrofit | Recurrence kill test fails |
| Gradient loss, GAN loss | Gradient loss made every metric worse in FBA's ablation ([FBA](https://ar5iv.labs.arxiv.org/html/2003.07711v1)); no matting GAN ablation found | n/a |
| NVFP4, FlashAttention-3/4 as defaults | Blackwell- or Hopper-only, prototype status ([torchao](https://docs.pytorch.org/ao/main/workflows/inference.html); [flash-attention](https://github.com/Dao-AILab/flash-attention)) | Opt-in experimental paths only |

## Data: rendered backings with exact spill, real footage only through a critic

The field's 2025-26 lesson is unambiguous. **Data moved the needle more than architecture.** In MatAnyone 2's ablation on YouTubeMatte 1080p, the curated real dataset cut MAD 12% and reference-frame training another 9%, while the online evaluator loss gave 5% ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)). Training even Robust Video Matting on that data cut its 1080p MAD from 4.27 to 3.37. VideoMaMa's ablation is sharper still: composition-only data scored 7.58 MAD on V-HIM60 Hard, real pseudo-labelled footage 3.18, and both together 2.61 ([VideoMaMa](https://arxiv.org/html/2601.14255v1)). Corridor trained CorridorKey 2.0 only on in-house 3D renders, so real footage is the largest untapped lever. But the big real sets are unreleased or legally murky. VMReal shows no release ([MatAnyone2 repo](https://github.com/pq-yang/MatAnyone2)), MA-V "will be publicly released" ([CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/39947)), and VideoMatte240K is stock footage under an "other" licence ([hyper.ai](https://hyper.ai/datasets/21099)). The decided data plan therefore makes the license-clean renders the core and admits real footage only through our own critic.

**Blender light-group renders (license-clean core).** Each render splits the screen emitter into a subject light group plus a 2x2 grid of screen-panel groups. Light transport is linear in emitter radiance, so one render reproduces any backing colour, any coarse unevenness and per-frame flicker, with physically exact spill on the subject, bounce, and the subject's own shadow on the screen ([Blender passes](https://docs.blender.org/manual/en/latest/render/layers/passes.html)). Every training sample recolours the backing at random: green, blue, off-hue, desaturated, uneven, flickering. 25% of batches also present the same frame under green and blue backings for the paired consistency loss. Subjects come from what the farm already does: procedural hair grooms on a bust, fur, Poly Haven CC0 plants, smoke and fire volumes, glass, sheer cloth, particles, motion blur and depth of field. Tracking markers, C-stands and floor tape are labelled alpha 0 as explicit negatives. Volume target: about 5,000 eight-frame clips plus 400 forty-eight-frame clips for temporal and recurrence training, about 60,000 frames stored as compressed half-float EXR within roughly 1.2 TB of the 2.4 TB disk. The 2x2 panel grid is a deliberate disk trade-off; fine screen texture is baked into each panel's emitter. The ADR curriculum widens each failure axis as validation error on that axis falls.

**Composites over the stage simulator.** VideoMatte240K HD train, P3M-10k (FG re-estimated), AM-2k and Transparent-460 train are composited over the same simulated stage the benchmark uses: uneven screen, markers, vignette, spill. Composites go through real ffmpeg H.264 and ProRes round trips with sensor noise, as the fine-tune pipeline already does. Degradation follows Real-ESRGAN's high-order recipe at low probability ([Real-ESRGAN](https://arxiv.org/abs/2107.10833v2)). Chroma subsampling is applied after compositing, never to GT, and 25% of samples stay clean so the model never trades sharpness for robustness. Image sets get synthetic camera motion so they can serve as pseudo-clips. **AIM-500, Transparent-460 test and VideoMatte240K test are excluded from all training** because benchmark shots c11, c12, c14 and c15 are built from them. AM-2k is perceptual-hash deduplicated against AIM-500 before use.

**Real exact ground truth by triangulation.** About 200 static captures: wigs, fur swatches, plants, cloth, glass and smoke in a jar, each shot in front of a monitor that displays two or three known backings. With two known backings, alpha and foreground are solved per pixel ([Smith & Blinn](https://alvyray.com/Papers/CG/blusig96.pdf); [McGuire et al.](https://casual-effects.com/research/McGuire2006PolarizationMatte/polarization-siggraph06.pdf)). This costs nothing but time, and gives real-sensor, real-optics GT for edges and transparency. It cannot give motion blur, so it anchors detail rather than temporal behaviour. Half the captures are held out as benchmark tier A3.

**Real green-screen footage, pseudo-labelled.** About two hours of our own green-screen footage is shot across several sessions with deliberately bad conditions: uneven light, a phone camera, web-grade compression, green wardrobe, markers. It is split by session into train and qualitative test. Pseudo-labels come from a teacher ensemble limited to licence-compatible sources: the enhanced model's own outputs (including TTT-Key mode), CorridorKey 1.0 (same licence family), and the physics projection with backing memory. The CK-Critic fuses them pixel-wise as MatAnyone 2 does, with a 9x9 Gaussian-blurred fusion mask. Training happens only on critic-reliable pixels ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)). Green screen adds a reliability test MatAnyone 2 never had. A pseudo-label must also reconstruct the plate (C ≈ alpha·F + (1 − alpha)·B_t), with F smooth and no more screen-coloured than the spill budget allows. **Competitor outputs (MatAnyone 2, VideoMaMa, GVM) are never used as pseudo-labels.** Their licences forbid clean redistribution, and training on a competitor's outputs would contaminate the comparison.

| Stage | Blender light-group renders | Composites over stage simulator | Triangulation captures | Real pseudo-labelled footage |
|---|---|---|---|---|
| S2 main fine-tune | 50% | 40% (VM240K 25%, image sets 15%) | 10% | 0% |
| S3 real adaptation | 40% | 15% | 10% | 35% (critic-masked) |
| S4 high-res polish | 60% (2K to 4K crops) | 20% (Transparent-460, AM-2k at full res) | 20% | 0% |

These ratios are decided starting points, not published values. No paper reports mixture ratios for a fine-tune at this scale. They follow VideoMaMa's finding that mixing beats either source alone, RVM's warning that models overfit synthetic matting distributions ([RVM paper](https://openaccess.thecvf.com:443/content/WACV2022/papers/Lin_Robust_High-Resolution_Video_Matting_With_Temporal_Guidance_WACV_2022_paper.pdf)), and MatAnyone's practice of ending on the cleanest edge data ([MatAnyone supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Yang_MatAnyone_Stable_Video_CVPR_2025_supplemental.pdf)).

## Losses and the CK-Critic: proven terms plus three green-screen additions

The backbone of the loss set is the one the field converged on and the one controlled ablation supports. In FBA's ablation on Composition-1k, adding composition loss to L1 cut MSE 11.2 to 9.1, Laplacian cut it further to 7.4, and the gradient loss made every metric worse ([FBA](https://ar5iv.labs.arxiv.org/html/2003.07711v1)). Robust Video Matting, MatAnyone, MatteFormer, AEMatter and ViTMatte all use L1 + Laplacian ([RVM train_loss.py](https://github.com/PeterL1n/RobustVideoMatting/blob/master/train_loss.py); [MatAnyone](https://arxiv.org/html/2501.14677v2)). The frame-difference temporal loss helped "across all metrics", while a RAFT-warped version was rejected because "there are severe motion noises for translucent pixels" ([Zhang et al.](https://ar5iv.labs.arxiv.org/html/2105.11427)). HDR evidence says to compress before computing the loss: display-encoded training beat linear by 2 to 9 dB across restoration tasks ([Ke et al. 2025](https://doi.org/10.1109/cvprw67362.2025.00085)), and tanh-compressed L1 beat plain L1/L2 in HDRUNet ([HDRUNet](https://doi.org/10.48550/arxiv.2105.13084)). The table gives the decided set. All weights are relative to alpha L1 = 1, and every term is computed in fp32 on the final fused output unless noted.

| Term | Weight | Where | Basis |
|---|---|---|---|
| Alpha L1, with edge weighting 1 + 5·edge indicator | 1.0 | All pixels | Field standard; BiRefNet-style 6x edge weight ([BiRefNet loss](https://huggingface.co/spaces/multimodalart/LHM/blob/main/engine/BiRefNet/loss.py)) on A/B |
| 5-level Laplacian, level weights 2^(s−1)/5 | 1.0 | All pixels | FBA ablation; MatAnyone |
| Core hole term: L1 on (1 − alpha) | 1.0 | GT alpha ≥ 0.99 | Holes are tiny in total L1 but visible in pushed levels; our -70% holes result |
| Leak term: L1 on alpha | 1.0 | GT alpha ≤ 0.01 | Mirror of the hole term |
| Display-space composite L1, random backgrounds | 1.0 | All pixels | FBA composition loss |
| asinh scene-linear composite L1 | 0.5 | All pixels | Targets the dark and light fringes that gamma-space blending hides |
| FG L1 in asinh space, weighted by alpha (soft) | 1.0 (2.0 in edge band) | alpha > 0 | RVM FG term ([train_loss.py](https://github.com/PeterL1n/RobustVideoMatting/blob/master/train_loss.py)); HDR compression evidence |
| Spill excess: max(0, G − max(R,B)) of predicted FG above GT FG (screen-channel generic) | 0.25 | Edge band | No published spill loss exists; exact spill-free GT from light groups makes it well-posed |
| Spill subspace and shadow field L1 (log s_t) | 0.25 each | Subject / screen | Exact light-group supervision |
| Temporal: L1 on alpha frame differences vs GT differences | 2.0 | Pixels reliable in both frames | dtSSD-as-loss ([Zhang et al.](https://ar5iv.labs.arxiv.org/html/2105.11427)); RVM uses 5x MSE |
| Temporal FG differences in asinh | 1.0 | alpha > 0 | RVM |
| Paired green/blue consistency (stop-grad on the lower-loss member) | 0.5 | 25% of batches | Light-group pairs |
| Laplace NLL for the confidence head (stop-grad on alpha mean) | 0.1 | All pixels | Keeps NLL from down-weighting hard pixels |
| Auxiliary losses on alpha_net and refine4 outputs | 0.25 and 0.5 | All | FBA's 0.25 secondary weighting; MODNet multi-branch gains ([MODNet](https://doi.org/10.1609/aaai.v36i1.19999)) |
| CK-Critic penalty, plus critic masking of all terms | 0.1 | Real pseudo-labelled clips only | MatAnyone 2: L_total = L_mat + 0.1·L_eval |

Temporal and hole terms ramp up over the first 10% of each stage to avoid early over-smoothing. Flow-warped consistency is used only on opaque cores of real clips, never on edges.

### CK-Critic: a green-screen evaluator that sees the plate

MatAnyone 2's Matting Quality Evaluator is the only published learned no-reference matting evaluator. It is a DINOv3 encoder with a DPT decoder, fed (RGB, predicted alpha, segmentation mask). It is trained as binary segmentation on labels from a 7x7 patch grid with discrepancy 0.9·MAD + 0.1·Grad, using focal plus Dice loss, and used both online and to fuse pseudo-labels ([MatAnyone 2](https://arxiv.org/html/2512.11782v2); [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Yang_MatAnyone_2_Scaling_CVPR_2026_supplemental.pdf)). Its checkpoint is unreleased, it is portrait-centric, and a 7x7 grid is far too coarse for 4K hair. So we build our own with four changes.

**Input**: plate, predicted alpha, predicted FG, the backing B_t, the recomposition residual I − (alpha·F + (1 − alpha)·B_t) and the sigma map, 11 channels in all through an expanded zero-init patch embedding. Spill is only visible by comparing FG with the plate, so the plate and residual are essential. **Encoder**: DINOv3 ViT-B/16 (86M) plus a DPT decoder, fine-tuned at encoder LR 3e-6 and 10x on the decoder as MatAnyone 2 did. It is used only at training time, so DINOv3 weights never ship. **Labels**: per-pixel composite error over four backgrounds, in asinh-linear and display space, computed exactly on synthetic data. Prediction sources are released CK2, CK2 with our fixes, CorridorKey 1.0, our own checkpoints over time, and synthetic corruptions (erode/dilate, chatter, holes, spill residue, bf16 quantization). **Outputs**: a continuous log-error regression, which avoids the saturation MatAnyone 2's authors report for binary maps, plus a reliable/erroneous map.

Training takes about two GPU-days on held-out-free renders, never on benchmark shots. Three rules guard against reward hacking. The critic penalty applies only to real clips. The critic is refreshed once mid-S3 with the student's new failure cases. Evaluation uses a separate **judge critic** with a different encoder (PE-spatial B/16, Apache-2.0 ([perception_models](https://github.com/facebookresearch/perception_models))) and disjoint corruptions, calibrated against blind compositor preferences. We expect the critic's main value to be offline curation. MatAnyone 2 got roughly twice as much from data curation as from the online loss.

## Fitting 4 GB: precision where the matte lives, savings everywhere else

The low-VRAM design rests on one asymmetry. **The parameters that matter for matte precision are few and cheap; the ones that cost memory are robust.** The frozen foundations (about 569M) and stage 1's /64 body are semantic and coarse. They are the best quantization candidates, and the foundations can run in a separate pass with features cached to NVMe, as the shipped low-VRAM mode already does. Stage 2 is only 201.8M (0.40 GB in bf16), so it never needs quantizing. The fp32 tail is tiny. Dense-prediction ViTs are fragile under full integer quantization: an integer-only segmenter lost 3.9 to 5.6 ADE20K mIoU, and softmax, GELU and residuals were the sensitive operations ([I-Segmenter](https://arxiv.org/html/2509.10334v1)). Weight-only INT8, by contrast, keeps activations in bf16 and needs no particular GPU generation ([torchao](https://docs.pytorch.org/ao/main/workflows/inference.html)). FP8 requires SM89 or newer and NVFP4 requires Blackwell, so neither can be the default for an RTX 30-series floor.

| Tier | Weights resident | VFM input (long side) | Stage-1 window | Stage 2 | /1 tail | Target |
|---|---|---|---|---|---|---|
| 4 GB | Two passes. Pass A: VFMs INT8 (0.57 GB), features to NVMe. Pass B: stage 1 INT8 (~0.49 GB) + stage 2 bf16 (0.41 GB) | 960 | Shipped auto-selector windows (2/1 to 4/2) + recurrent carry | bf16, block-wise chunked over windows and stripes (exact for windowed attention); causal t−1 only | fp32 streamed in row bands; refiner K = 256 tiles | 1080p native; 4K with trunk at 1080p and projection + refiner at native 4K |
| 6-8 GB | Everything bf16 resident (~2.5 GB) | 1280 | Shipped auto-selector (8/4 to 24/12) + carry | bf16, t−1 and t+1 | fp32; K = 1,024 tiles | 1080p and 4K native trunk |
| 12+ GB | Everything bf16 | 1920 | Up to 80/40 + carry | bf16, t±1 | fp32; K uncapped | Best quality; TTT-Key on 1080p crops (4K TTT needs ~24 GB) |

The weight sizes come from parameter counts. Activation and peak figures are design budgets, not measurements. Each tier reserves about 1.2 GB for the CUDA context and allocator; Accelerate puts the context alone at 1 to 2 GB ([Accelerate](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)). Stage 2's residual stream is about 21 MB per frame at 1080p and 83 MB at 4K in bf16 (8,160 and 32,400 tokens at d = 1280), so chunking its window attention and MLP keeps stage 2 well under 1 GB even at 4K. The shipped auto-selector reports 4K as "not enough memory" on 4 GB cards ([CorridorKey2.0 README](https://github.com/nikopueringer/CorridorKey2.0)). The decided fix is a split, not tiling. The trunk keys a 1080p downscale, and the resolution-free projection head and sparse refiner restore native detail from the 4K pixels. Robust Video Matting runs 4K at downsample ratio 0.125 the same way ([RVM repo](https://github.com/PeterL1n/RobustVideoMatting)). This works because the projection reads detail from pixels, not tokens.

The remaining rules fill in the rest of the low-VRAM design. **Never quantize** the stem, the final norms, the depatchify, any head, the refiner, attention logits, softmax, the residual stream, or recurrent state, which accumulates error over thousands of frames. INT8 on stage 1 is limited to the /64 body's linear weights (274M), and both INT8 paths must pass a gate. That gate is per layer group, against the bf16 reference, on holes at three epsilons, edge-band alpha error and dtSSD, and allows no more than 1% relative regression. **Attention** uses PyTorch SDPA with automatic kernel selection, FlashAttention-2 optional on RTX 30/40, and a tested fallback on RTX 50 (SM120), where community reports show flash-attn builds failing ([flash-attention issue 2361](https://github.com/Dao-AILab/flash-attention/issues/2361)). **Compilation** uses torch.compile with CUDA graphs only on fixed-shape modules (stage 2 per frame, the refiner at fixed K), because CUDA graphs need fixed shapes ([Torch-TensorRT guide](https://docs.pytorch.org/TensorRT/user_guide/performance_tuning.html)). **Swapping**, if a future card needs it, is a manual two-slot pinned-memory prefetch on a side stream, because Accelerate offload does no prefetching. **FP8 weight storage** is offered on Ada and newer as a memory-only option; NVFP4 is not offered.

## One A6000, about 30 GPU-days: the training schedule

The schedule trains one design. It is not a roadmap of releases. Full fine-tuning of ~690M parameters with fp32 master weights and AdamW needs about 11 GB for weights, gradients and optimizer state. That leaves room on the 48 GB A6000 for activations with the gradient checkpointing the code already supports, bf16 autocast in the trunk and fp32 in the tail and losses. Frozen foundations run live in bf16 under no_grad. Augmentations change their inputs every sample, so feature caching is impossible in training. Learning rates follow MatAnyone's 1e-4 / 1e-5 / 1e-6 stage pattern ([MatAnyone](https://arxiv.org/html/2501.14677v2)) and AEMatter's finding that large patches widen context ([AEMatter](https://arxiv.org/html/2304.01171)). Step counts are set from step time measured in the first hour; the GPU-day budgets are the decision.

| Stage | GPU-days | What trains | Data and crops | LR (new / trunk) |
|---|---|---|---|---|
| S0 Kill tests + corrected-GT benchmark baseline | 2 | Nothing (inference and oracle tests) | Benchmark + 3 test renders | n/a |
| S1 Warm-up | 2 | New branches, refine4, refiner, heads, gates, recurrent sites; trunk frozen | S2 mix, 512² crops, 8 frames | 1e-4 / 0 |
| Critic | 2 | CK-Critic and judge critic | Renders + S1 predictions + corruptions | 3e-6 encoder, 3e-5 decoder |
| S2 Main fine-tune | 12 | Everything trainable | S2 mix, 512 to 768² crops, 8 frames; 30% with far reference frames and two-window truncated backprop through the recurrence; random window lengths {1,2,4,8,24,48}; random VFM scale | 1e-4 / 2e-5 |
| S3 Real adaptation | 5 | Everything | S3 mix, critic-masked real clips | 1e-5 / 5e-6 |
| S4 High-res polish | 3 | Stage 2, tail, refiner (stage 1 frozen) | S4 mix, 1024 to 2048² crops, 4K native refiner tiles | 1e-6 |
| S5 Gates and benchmark | 4 | Quantization gates, tier calibration, confidence calibration; full benchmark + competitor runs | All tiers | n/a |

Two operational constraints shape this. GPUs 0 and 1 serve production, so all rendering runs on the 128 CPU threads, in parallel with S0 to S2, never on GPU 2. And because MatAnyone 2 attributes 9% MAD to **reference-frame training at no inference cost** ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)), the 400 long renders and VideoMatte240K's long clips exist specifically to supply far frames for the recurrence. The shipped model must not regress where it is already good, so S2 keeps trunk LR 5x below new-module LR. An EMA of weights (decay 0.999) is what gets evaluated, and every stage ends with a Tier C regression check against the released model on real footage, including all three hint modes.

## Benchmark protocol: exact GT first, public protocol second, eyes last

Pixel metrics alone mislead. In Rhemann's perceptual study, SAD and MSE correlated with user rankings at Kendall tau 0.28 to 0.51, against 0.75 for purpose-built gradient and connectivity errors ([Rhemann et al. 2009](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/01/cvpr09-matting-Eval.pdf)). For video, temporal coherence matters more to viewers than accuracy, especially at slow motion ([Erofeev et al. 2015](https://videomatting.com/content/paper/videoMattingBenchmark.pdf)). And every public video set's GT comes from a keyer or roto, which biases toward keyer-like outputs ([Magenta Green Screen](https://arxiv.org/abs/2306.13702)). The suite therefore has three tiers with different jobs.

**Tier A, exact ground truth (headline).** A1 is our 15-shot benchmark (c01 clean control through c15 4K hair), re-run on the corrected GT where spill lives only in the plate. A2 is 10 new Blender shots from held-out assets and seeds: a hair groom, fur, smoke volume, glass with known IOR, sheer cloth, particles, marker-dense stage, green wardrobe, blue screen and a 48-frame whip move. A2 has exact physical spill and shadow. A3 is the 100 held-out real triangulation captures, static but exact. A1 shares its stage simulator with our training composites, so **the A1-versus-A2/A3 gap is the overfitting alarm**. A win on A1 that does not reappear on A2 and A3 is not a win.

**Tier B, public sets at the MatAnyone 2 protocol (comparability).** VideoMatte240K-test and YouTubeMatte at 512x288 and 1920x1080, scored with MAD, MSE, Grad, dtSSD, and Conn at 512x288 only, plus the 19-video CRGNN real set with GT every 10th frame ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)). The VideoMatte composites download from the Robust Video Matting evaluation archive ([training.md](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md)), and YouTubeMatte ships with first-frame masks ([MatAnyone repo](https://github.com/pq-yang/MatAnyone)). The bar to beat is MatAnyone 2 (YouTubeMatte 1080p MAD 1.61, VideoMatte 1080p 4.10, CRGNN 4.24) and the SAM2 + VideoMaMa pipelines (YouTubeMatte 1080p MAD 1.27 to 1.36) ([VideoMaMa](https://arxiv.org/html/2601.14255v1)). Two cautions apply. These composites have no green screen, so a keyer is out of its domain, and we report that honestly as row B1. As a clearly labelled supplement, B2 composites the same foregrounds and alphas over our simulated green stage for every method. **Protocol validation**: our re-run of MatAnyone 2 must land within ±5% of its published YouTubeMatte 1080p MAD before any Tier B number is trusted.

**Tier C, real footage (eyes).** This tier uses the CorridorKey 1.0 QualityBenchmark inputs ([HF dataset](https://huggingface.co/datasets/nikopueringer/CorridorKey1.0_QualityBenchmark)), our held-out real green-screen sessions, and CineMatte-4K if it is released ([arXiv](https://arxiv.org/abs/2605.18328)). CineMatte-4K is real green screen with expert alpha, so it would be promoted to a quantitative tier and kept out of training. Scoring uses judge-critic ERR/MER/BER maps and blind compositor A/B.

### Every runnable competitor and exactly what it is fed

Every method receives the same decoded frames at the highest precision it accepts, so our 16-bit decode is not a hidden advantage. Methods with resolution caps are run at the cap, and their alpha is upsampled bilinearly to GT size, with the cap stated in the table. Mask-guided methods get two rows: an oracle mask derived from GT, and a practical mask from a SAM 2.1 click on frame 1, following VideoMaMa's practice of also testing degraded masks ([VideoMaMa](https://arxiv.org/html/2601.14255v1)).

| Method | Licence | Practical input | Oracle row | Notes |
|---|---|---|---|---|
| CorridorKey 2.0 Enhanced (ours) | CC BY-NC-SA variant | Nothing (no hint) | First-frame GT mask hint | All three tiers; also 4 GB and 8 GB tier rows |
| CorridorKey 2.0 released | CC BY-NC-SA variant ([README](https://github.com/nikopueringer/CorridorKey2.0)) | Nothing | First-frame GT mask hint | Default inference |
| CorridorKey 2.0 + our inference fixes | same | Nothing | First-frame GT mask hint | The preliminary −8% composite baseline |
| CorridorKey 1.0 (green and blue weights) | CC BY-NC-SA variant ([repo](https://github.com/nikopueringer/CorridorKey)) | Per-frame hint from a rough chroma key | Eroded GT hint | Needs coarse hints by design |
| Robust Video Matting (MobileNetV3, ResNet50) | GPL-3.0 ([repo](https://github.com/PeterL1n/RobustVideoMatting)) | Nothing; downsample 0.25 HD, 0.125 4K | none | Run only, no code reused |
| BGMv2 | MIT ([repo](https://github.com/PeterL1n/BackgroundMattingV2)) | Clean plate without the subject's shadow (what a real plate pass captures) | Plate with shadow | Per-frame |
| MatAnyone | NTU S-Lab 1.0 ([repo](https://github.com/pq-yang/MatAnyone)) | SAM 2.1 first-frame mask | GT first-frame mask | |
| MatAnyone 2 | NTU S-Lab 1.0 ([repo](https://github.com/pq-yang/MatAnyone2)) | SAM 2.1 first-frame mask | GT first-frame mask | The public bar |
| SAM 2.1 + VideoMaMa | CC BY-NC code, Stability Community weights ([repo](https://github.com/cvlab-kaist/VideoMaMa)) | SAM 2.1 per-frame masks from a frame-1 click | GT per-frame masks | Reduced resolution on 48 GB, stated |
| GVM | BSD-2 academic ([repo](https://github.com/aim-uofa/GVM)) | Nothing; max_resolution 960, 1 step | none | |
| MaGGIe | CC BY-NC 4.0 ([repo](https://github.com/hmchuong/MaGGIe)) | SAM 2.1 per-frame masks | GT per-frame masks (upper bound) | |
| FTP-VM | GPL-3.0 ([repo](https://github.com/csvt32745/ftp-vm)) | Trimap from a rough chroma key | GT-band trimap | |
| BiRefNet_HR-matting | MIT ([repo](https://github.com/ZhengPeng7/BiRefNet)) | Raw frames, per frame | none | No temporal model |
| Classical: ffmpeg chromakey + despill | n/a | Fixed parameters | Parameters grid-searched against GT (labelled oracle) | The "compositor's quick key" baseline |
| pymatting closed-form | MIT ([repo](https://github.com/pymatting/pymatting)) | Trimap from rough chroma key | GT-band trimap | Per-frame |

GUI-only and paid keyers (Keylight, IBK, Primatte, Resolve Magic Mask) are excluded from quantitative rows because they cannot be driven headless or cost money. A volunteer compositor's manual key on three Tier C shots may appear as a labelled qualitative reference. GVM and VideoMaMa nominally need about 80 GB ([CorridorKey repo](https://github.com/nikopueringer/CorridorKey)). They run on the 48 GB A6000 at reduced resolution, and every such downscale is printed beside the number.

### Metrics, statistics and the ship gate

| Metric | Computed on | Why it is there |
|---|---|---|
| **comp_mae** (headline): composite error over 4 held-out backgrounds, display and asinh-linear | Tier A, banded core / edge / background | What a compositor sees; banding stops tiny edge areas hiding |
| comp_edge, comp_psnr | Tier A | Edge-focused and global composite quality |
| SAD, MSE, Grad, edge_mae | A and B | Comparability; Grad tracks humans (tau 0.75) |
| Conn on edge-band crops | A (B at 512x288) | Floating fragments, which tracks humans (tau 0.75); too expensive at full 4K ([RVM paper](https://arxiv.org/pdf/2108.11515)) |
| holes at ε = 0.01, 0.05, 0.2; leak | Tier A | Separates precision loss from semantic holes, as in a pushed-levels QA view |
| fg_mae, chroma-excess spill in edge band | Tier A | Despill balance: neither green residue nor magenta overcorrection ([Keheka tech check](https://www.keheka.com/tech-check-checklist/)) |
| Halo sign: signed composite luminance error over dark and bright backgrounds | Tier A | Separates dark fringes from light fringes |
| dtSSD, stratified by motion speed | A and B | Standard; humans punish incoherence most at slow motion |
| MESSDdt with GT flow from the GT foreground over solid green | Tier A | Motion-aware coherence, as videomatting.com computes it ([videomatting.com](https://videomatting.com/)) |
| Edge-chatter energy: alpha temporal energy where GT does not change | Tier A | Boiling edges on static subjects, the classic tech-check failure ([CompStart](https://compstart.pro/tech-check)) |
| Confidence calibration: expected calibration error of the confidence pass | Tier A | The confidence pass must be honest to be useful |
| Judge-critic ERR / MER / BER | Tier C | No-GT real footage ([MatAnyone 2](https://arxiv.org/html/2512.11782v2)) |
| fps and peak VRAM at 1080p and 4K; pass/fail under 4, 6, 8 GB caps | All methods | Cost of quality |

Fifteen shots is a small sample, so every comparison uses **per-shot paired differences with bootstrap 95% CIs over shots and frames**. Macro-averages are reported alongside every per-shot number. The **ship gate** has three conditions. First, the enhanced model beats released CK2 plus our fixes on comp_mae, with A1 and A2 both CI-excluding zero. Second, it regresses no single shot by more than 5%. Third, it does not lose on Tier C blind preference. Low-VRAM tiers are simulated on the A6000 with `torch.cuda.set_per_process_memory_fraction`, a stand-in that undercounts a real card's context overhead, which is why each tier also reserves 1.2 GB.

**Presentation** uses the existing side-by-side renderer and adds the views VFX QA actually uses. The main view is a synchronized grid (GT, ours, the three strongest competitors, released CK2) at identical crop and frame. Alongside it come fixed 4x nearest-neighbour zoom crops on hair, web, smoke and motion blur. Per-method |alpha − GT| heatmap videos share one colour scale, and per-pixel temporal-variance flicker maps share another. xt-slices stack one row through a hair edge over all frames. Composites are shown over a contrasting colour, over a checkerboard, and with levels pushed up and down to expose holes and floating greys. That is the practice of previewing "at full resolution in full motion against a bright primary colour" ([CompStart](https://compstart.pro/tech-check)). Encodes are lossless or CRF 12 with the same codec for all methods, and failure shots are always included. The blind A/B study uses at least three compositors on looped full-resolution composite clips with randomized sides, and reports win rate with CIs. Rhemann found individual-to-mean agreement of 0.87 to 0.90 ([Rhemann et al. 2009](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/01/cvpr09-matting-Eval.pdf)), which is the realistic ceiling for agreement. The same human preferences calibrate the judge critic (Kendall tau between critic score and win rate).

## Licensing: what a free NC-SA derivative can safely ship

A free, non-commercial, same-licence derivative is permissive for most inputs. The two real exposures are training data with unstated licences, and anything trained on competitor outputs. The decided policy is that the shipped weights depend only on rows marked "ship" or on gray datasets whose licence has been confirmed. **Gray licences are resolved during S0**, since reading licence texts and emailing authors costs no GPU time. Any dataset still unconfirmed when S2 starts is dropped from the mix, and its share goes to renders. A fork trained after S4 cannot launder data that S2 already used, so the gate has to come before S2.

| Item | Licence | Role | Decision |
|---|---|---|---|
| CorridorKey 2.0 code and weights | CC BY-NC-SA 4.0 variant; free, same licence, keep name ([README](https://github.com/nikopueringer/CorridorKey2.0)) | Base | Ship under the same licence as "CorridorKey 2.0 Enhanced" |
| C-RADIOv4-SO400M | NVIDIA Open Model License; card says commercial and non-commercial use OK ([card](https://huggingface.co/nvidia/C-RADIOv4-SO400M)) | Frozen, shipped dependency | Ship as a separate download; full licence text still to be read |
| MoGe-2 | MIT, DINOv2 module Apache-2.0 ([MoGe](https://github.com/microsoft/MoGe)) | Frozen | Ship |
| DeepMind RVM | Apache-2.0 code, CC-BY 4.0 materials ([representations4d](https://github.com/google-deepmind/representations4d)) | Frozen | Ship with attribution |
| DINOv3 | DINOv3 License: derivatives allowed, agreement must accompany distribution ([LICENSE](https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md)) | CK-Critic encoder, training only | Do not ship; no redistribution question arises |
| PE-spatial | Apache-2.0 ([repo](https://github.com/facebookresearch/perception_models)) | Judge critic, eval only | Not shipped |
| SAM 2.1 | Apache-2.0 ([repo](https://github.com/facebookresearch/sam2)) | Optional click-to-hint tool and benchmark masks | May ship as an optional tool |
| Blender renders, Poly Haven CC0 assets, own captures | CC0 / ours | Core training data | Ship-clean |
| VideoMatte240K | "Other", stock footage ([hyper.ai](https://hyper.ai/datasets/21099)) | Training composites | Gray: ask the BGMv2 authors; clean-only fallback |
| P3M-10k, AM-2k, Transparent-460 | Not verified in our research | Training composites | Gray: read each licence before release; clean-only fallback |
| AIM-500 | No explicit licence found ([TIB](https://service.tib.eu/ldmservice/dataset/bibtex/aim-500)) | Benchmark c14 only | Eval only |
| CorridorKey dataset (gated) | CC-BY-NC-ND per the third-party record | n/a | Excluded unless Corridor agrees in writing |
| CineMatte-4K, MA-V, VMReal | Unreleased or unknown | n/a | Eval only if released; never train on CineMatte-4K |
| MatAnyone 2, VideoMaMa, GVM, MaGGIe outputs | S-Lab / CC BY-NC / Stability / academic | Benchmark competitors | Never used as training labels; check before publishing side-by-sides containing their output |

## Risk register: every risky call has a cheap kill test

| # | Decision | How it fails | Confidence | Cheapest kill test (cost) | If killed |
|---|---|---|---|---|---|
| 1 | Backing memory as conditioning | Oracle plate barely helps; bad first-pass alpha poisons the plate | High | Swap GT clean backing for B̂ inside `plate_foreground` / `despill_backing` on A1; then build P from the model's own alpha on c02 (hours, no training) | Keep plate for FG recovery only; drop the stem conditioning |
| 2 | /1 physics projection + SNR fusion | Oracle alpha_phys does not beat the model on hair, fur or web | Medium-high | alpha_phys with GT F blurred to /16 and GT backing, then with model F and pull-push B, on c12, c14, c15 (an afternoon) | Oracle loses: drop it and lean on the refiner. Only model-based loses: fix B and F_loc first |
| 3 | Light-group algebra | Group sum ≠ combined pass; emissive meshes not assignable to light groups | High | One scene, check sum of groups vs combined (PSNR > 50 dB) and recolour vs true re-render (half a day) | Model the screen as lit card with its key lights in the group |
| 4 | Native chroma path | Codec-shot error is blocking or noise, not chroma resolution | Medium | Luma-guided chroma upsampling before the current model on c03 (an hour) | Drop chroma planes; keep noise map and QP |
| 5 | VFMs at 1/2 resolution | No measurable gain for 4x VFM tokens at 1080p | Medium | After S1, evaluate 1/4 vs 1/2 on 1080p shots (holes, edge, subject errors) | Ship 1/4 everywhere, saving VRAM |
| 6 | Stage-1 recurrence | Window 4 + carry does not approach window 24 | Medium | Mid-S2: dtSSD on c01, c05, c15 for window 4 + carry vs window 24 vs released | Under half the gap recovered: drop recurrence, keep Hann tiers |
| 7 | Stage-2 3-frame attention | Shimmer headroom is small | Medium | Post-hoc temporal median of released alpha on hair, fur, web; dtSSD change (an evening) | Under 10% dtSSD headroom: skip it |
| 8 | Spill subspace | Two-scalar model misses specular and subsurface spill | Medium | Least-squares fit on Blender passes; unexplained energy fraction (an hour) | Over 5% residual: free RGB spill head |
| 9 | Screen membership | MoGe-2 is blind on featureless screens | Medium | Fit plane on c02; marker vs subject distance (an hour) | Rely on backing memory alone |
| 10 | Sparse refiner routing | Uncertainty does not localize error | Medium | After S1, share of alpha error in the top-10% most uncertain pixels | Under 50%: dense refine4 only |
| 11 | TTT-Key | Confirmation bias; no gain | Medium-low | One LoRA step per frame on c04 and c09 with rec + temporal + anchor (one evening) | Remove the mode |
| 12 | CK-Critic | Fails to rank errors, or gets gamed | Medium | Spearman between critic score and true composite error on A2 (≥ 0.7 required); track student-vs-judge divergence | Heuristic reliability (recomposition residual + teacher agreement) |
| 13 | Synthetic overfit, shared simulator | A1 improves, A2/A3/C do not | Medium | A1 vs A2/A3 delta at every stage end | Cut composite share; more renders and captures |
| 14 | Gray datasets | A licence rules one out | Medium | Read each licence and email authors during S0 (no GPU); if its value is in doubt, a 2 GPU-day S2-style A/B of clean vs full mix from the S1 checkpoint | Drop the dataset before S2; its share goes to renders |
| 15 | INT8 on 4 GB | Holes or edges regress | Medium | Per-layer-group gate vs bf16 reference (≤1% relative) | bf16 stage 1 with two-slot swap |
| 16 | 4 GB fits 1080p and split-4K | Peak exceeds budget | Medium | Memory-fraction cap on the A6000 with real clips (an hour) | Smaller K, shorter window, trunk at 720p for 4K |
| 17 | Edge-weighted L1 | Trades composite error for edge error | Low-medium | 3k-step A/B on edge_mae and comp_mae | Plain L1 |
| 18 | Single-GPU budget | Step time too slow for the GPU-days | Medium | Measure step time in hour one | Shorter clips in S2; S4 crops to 1024 |
| 19 | Glass (c11) | No change, as now | Low expectation | Scalar vs per-channel fit to c11 GT composite (an hour) | Reopen the RGB transmission head only if the fit gap is large |
| 20 | Regression of shipped behaviour | Hint modes or real footage worsen | Medium | Tier C plus hint-mode regression at every stage end | Lower trunk LR; replay shipped-style renders |

## Conclusion

The research changes the question from "what bigger network" to "which pixels need a network at all". Keying with a known screen is one equation short. Recovering B more exactly, measuring sigma, and reading alpha where ‖F − B‖/sigma is large converts much of the problem from learned guesswork into arithmetic. The 665M trunk is then spent only on the genuinely ambiguous cases: low SNR, wardrobe that matches the screen, smoke, and deciding which object is the subject. That reframing also explains our preliminary numbers better than any architecture story, and it is why the enhanced model barely grows and still fits a 4 GB card. Even the 4K-on-4 GB problem dissolves once full-resolution detail comes from pixels instead of tokens.

The expected losers are worth saying out loud. Glass will not improve much, because refraction breaks the compositing equation this design exploits. On natural-background public sets, MatAnyone 2 and the VideoMaMa pipelines may still win, because those sets measure general matting, not keying. The decisive early evidence costs almost nothing. The oracle-plate and oracle-projection tests (risks 1 and 2) take a day together and either confirm the design's foundation or redirect the training budget before a single GPU-day is spent.

## Addendum, 2026-10-21: kill-test results and the revised design

The two "run first" oracle checks were run on the corrected benchmark (results from the shipped weights with our inference fixes; `bench/oracle.py`, `bench/decompose.py`).

**Physics alpha fails its kill test.** Even with oracle inputs (true backing, foreground colour pushed out from ground-truth solid pixels, screen-measured noise), the /1 projection alpha is 3 to 4 times worse than the network's alpha in the edge band. Edge MAE on the real-stage shot is 0.016 against the network's 0.0046, with similar ratios on motion blur, hair and fur. SNR fusion does not rescue it. The physics projection head and the TTT-Key mode (which relies on the same smooth-foreground assumption) are **removed** from the design.

**The backing memory mostly fails its kill test.** Swapping the true clean screen into plate-recovered foreground moves composite error by 1 to 3% on most shots (real stage 0.920 to 0.915). The exception is smoke, 18% better (2.67 to 2.20). The full per-shot backing memory is **removed**; the cheap pull-push estimate stays, and smoke is addressed through training data instead.

**Where the error actually is.** Splitting composite error into an alpha-only part (predicted alpha with GT foreground) and a foreground-only part (GT alpha with predicted foreground):

| Shot | Total | Alpha part | FG part |
|---|---|---|---|
| Clean control | 0.18 | 0.17 | 0.02 (equals the plate floor) |
| Real stage | 0.72 | 0.25 | 0.53 |
| Camera H.264 | 1.69 | 0.21 | 1.54 |
| Crushed web | 2.07 | 0.24 | 1.86 |
| Smoke | 1.85 | 2.17 | 2.39 |
| Glass | 1.72 | 1.74 | 0.70 |
| Spider web 4K | 0.60 | 0.60 | 0.36 |
| Fur | 2.74 | 0.36 | 2.86 |
| Hair 4K | 1.10 | 0.20 | 0.93 |

Alpha is already near its floor (about 0.2) except on smoke, glass, the web and fur. Everywhere else the error is foreground colour: residual spill, plus codec and sensor damage that lives in the plate. Our despilled plate foreground already beats a naive plate-recovered foreground by 3 to 5 times on every spill shot.

**Revised design (supersedes the physics layer above).** Keep: the 16-bit decode, fp32 IO and output tail, plate-recovered foreground, the trunk and its three frozen foundations, stage-1 recurrence, stage-2 three-frame attention, refine4 plus the sparse confidence-routed refiner, the degradation side channel (noise map, chroma planes) into stage 2 and the tail, the CK-Critic, the training data plan and the benchmark protocol. Change: the learned **spill and foreground head** becomes the central addition, supervised exactly by Blender light-group renders (spill-free and spilled passes), replacing the physics projection; the confidence head routes the sparse refiner only. Training effort concentrates on smoke, glass, sub-pixel threads and fur (render curriculum) for alpha, and on spill removal under codec damage for foreground.

**Budget constraint (from Thomas, 2026-10-21): $50 maximum cloud spend.** Training runs on the free a6000-box A6000 plus one Thunder Compute RTX A6000 ($0.35/hr, about 140 hours), with periodic weight averaging between the two, a box-side watchdog that deletes the instance before $50, and benchmark-scored checkpoints that stop the run when quality plateaus.
