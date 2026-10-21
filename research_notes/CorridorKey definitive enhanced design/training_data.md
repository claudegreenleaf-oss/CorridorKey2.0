# Training data strategy for an enhanced green/blue-screen matting model (CorridorKey 2.0 retrain), as of Oct 2026

Method note: ~15 tool calls. Many dataset license pages could not be fetched; anything not backed by a URL below is in Gaps or Inferences and is unverified. Core finding: the field (MatAnyone 2, VideoMaMa) shifted in 2025-26 toward large real-footage datasets with pseudo-labels curated by learned quality evaluators, with small amounts of exact-GT data as anchors.

## Q1. Which public datasets are worth adding (availability, license, size, alpha quality)

### Takeaway
Only a few candidates are verifiably usable. The big new real-footage sets (VMReal, MA-V) are the most valuable but VMReal has no confirmed release, and licenses on VM800/VideoMatte240K/YouTubeMatte are unstated and likely inherit restrictive stock-footage terms. For a non-commercial, redistributable-weights goal, treat all stock-footage-derived data as a legal risk to resolve before release, and rely on the in-house generator as the license-clean core.

### Cited Findings
- VideoMatte240K: license listed only as "other"; footage was purchased as stock or found as royalty-free online; original distribution via the BGMv2 page (University of Washington GRAIL). — [hyper.ai metadata](https://hyper.ai/datasets/21099), [paperswithcode mirror](https://astro.paperswithcode.com/dataset/videomatte240k) (via search summary; no explicit license text found)
- VM800 (MatAnyone training set): green-screen foregrounds downloaded from Storyblocks, Envato Elements, Motion Array; alpha extracted with After Effects Keylight plus cleanup. I found no statement that VM800 is released. — [MatAnyone paper](https://arxiv.org/pdf/2501.14677)
- YouTubeMatte: 32 foregrounds from YouTube videos, composites harmonized with Harmonizer; benchmark released March 2025 via Google Drive; no data license stated. Intended as an eval set extending VideoMatte240K-Test (5 foregrounds). MatAnyone code is under NTU S-Lab License 1.0 (non-commercial style; follow its terms); training code released 2026.03. — [MatAnyone GitHub](https://github.com/pq-yang/MatAnyone)
- VMReal (MatAnyone 2): about 28K clips, 2.4M frames, ~35x larger than VM800. 4.5K clips at 1080p from footage sites and YouTube; the rest from SA-V (human-centric subset, typically 720p). Alpha is a fused output of two branches (MatAnyone video branch from first-frame mask; MattePro image branch from per-frame SAM 2 prompts), blended by a learned quality map. The text I retrieved contains no dataset release statement. — [MatAnyone 2 HTML](https://arxiv.org/html/2512.11782)
- MA-V (VideoMaMa, CVPR 2026): 50,541 real videos with pseudo-matting from SA-V masks, ~50x larger than existing real-video matting sets. CVPR listing says models and MA-V "will be publicly released"; no license found. — [VideoMaMa HTML](https://arxiv.org/html/2601.14255v1), [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/39947)
- SynthHair (NeurIPS 2026 poster): 16k RGBA 4096x4096 portraits rendered with MetaHuman + Unreal; labels include matting, hair growth direction, depth; no license or release terms on the page. MetaHuman assets carry Epic license terms, which probably do not allow redistribution of renders as training sets without checking (inference). — [NeurIPS page](https://neurips.cc/virtual/2026/poster/139817)
- CineMatte-4K (arXiv 2605.18328, 2026): 4K HDR image+video set captured on an LED virtual-production stage by briefly swapping the wall to green with the actor held still; image subset 500 samples (100 test), video 30 clips of ~10 s at 30 fps; ground-truth alpha by human experts; no license on the abstract page. Highly relevant to green-screen use. — [arXiv](https://arxiv.org/abs/2605.18328), [search snippet](https://arxiv.org/pdf/2605.18328)
- HIM-2K: HF mirror says CC BY-NC-4.0; 1680 composite images + 320 natural images (instance/multi-person benchmark). — [HF nobg/HIM-2K](https://huggingface.co/datasets/nobg/HIM-2K)
- MicroMat-3K: conflicting license metadata on the same HF repo (cc-by-4.0 in card header vs cc-by-nc-4.0 in README at one commit); verify current. — [HF naver-iv/MicroMat-3K](https://huggingface.co/datasets/naver-iv/MicroMat-3K/blob/e857e44e2bd1d437f41b64d9207369256950a42d/README.md)
- AIM-500: images collected from free-license websites per paper; no explicit dataset license found. — [TIB record](https://service.tib.eu/ldmservice/dataset/bibtex/aim-500)
- CRGNN real-world video set: 19 real videos, labeled every 10 frames, 711 labeled frames total; repo github.com/TiantianWang/VideoMatting-CRGNN; license not checked. Eval-grade only given size. — [Adaptive Human Matting](https://arxiv.org/pdf/2304.06018), [CRGNN](https://paperswithcode.com/paper/video-matting-via-consistency-regularized)
- GFM identifies the composite-vs-real gap as resolution, sharpness, noise, illumination differences and ships BG-20k clean high-res backgrounds; reports ~60% generalization error reduction with its composition route. — [GFM / Bridging Composite and Real](https://arxiv.org/pdf/2010.16188)
- Another 2024 trimap-free paper uses image harmonization in composition and releases a 19k real-world dataset. — [arXiv 2405.17916 listing](https://papers.cool/arxiv/2405.17916)
- SAM2Matting (arXiv 2606.27339) claims strong results "trained only on images", criticizing video matting datasets as expensive and narrow. Image-only training is a data-light baseline worth noting. — [arXiv](https://arxiv.org/abs/2606.27339)
- CorridorKey (the original) third-party port lists the model as CC BY-NC-SA 4.0; I could not find the CorridorKeyDataset HF card. — [LateNite port credits](https://corridorkeybylatenite.fcp.cafe/credits/)

### Inferences
- Priority add list for a green-screen product: (1) CineMatte-4K if released (real green screen with expert alpha, same domain); (2) MA-V/VMReal subsets if released, as unlabeled-footage-with-pseudo-alpha for the P3M-style portrait domain; (3) HIM-2K / AIM-500 / Distinctions-646 / Composition-1k only as small edge-quality and eval anchors, since they are photo-derived with foreground color estimation artifacts.
- Datasets built on stock or YouTube footage (VM240K, VM800, YouTubeMatte, VMReal) are probably not redistributable; weights trained on them may be fine for research but redistribution rights are unclear. Resolve with authors before release; otherwise treat them as eval-only or fine-tune-optional ablations.
- The CorridorKeyDataset (CC-BY-NC-ND, gated) is "NoDerivatives": training weights are arguably not an adaptation, but this is a legal question I cannot answer; ask Corridor in writing.

### Gaps
- Could not verify licenses for Distinctions-646, Composition-1k, PPM-100, SIMD, RefMatte, VideoMatting108, P3M-10k (P3M is known to have a privacy-protection research-use release, unverified here), or the CRGNN repo.
- Could not confirm whether VMReal or MA-V are actually downloadable as of Oct 2026.
- CineMatte-4K release status/license unknown.

## Q2. Synthetic rendering for matting: published pipelines and which realism factors matter

### Takeaway
Evidence for synthetic matting data exists (SynthHair, eyebrow/eyelash renders with domain adaptation, VideoMaMa trained solely on synthetic data then transferred) but rigorous per-factor ablations of hair strand realism, motion blur, spill, or ISP are missing from what I found. The strongest transferable lesson is that synthetic data is best used to teach the generator/teacher, with real footage carrying the final distribution.

### Cited Findings
- SynthHair: 16k MetaHuman/Unreal RGBA portraits at 4K; reports strong domain generalization and clear gains on high-resolution images; over 80% user preference in A/B on high-res images. No numeric metrics on the page. — [NeurIPS page](https://neurips.cc/virtual/2026/poster/139817)
- Eyebrow matting: trained on rendered avatars with adversarial domain adaptation to real images (CGF 2022). EyelashNet pairs 5,400 real captures with 5,272 rendered samples (SIGGRAPH Asia 2021). — [Eurographics](https://diglib.eg.org/handle/10.1111/cgf14682), via [search summary](https://diglib.eg.org/bitstream/handle/10.1111/cgf14682/v41i7pp347-358.pdf)
- DiffLocks generates image + 3D hair pairs with Blender geometry nodes (110 parameters control geometry and material); HairWeaver uses Blender Cycles renders plus a Flux image-to-image step to push photorealism. Neither targets matting but show Blender procedural hair at scale and a generative "realism pass" idea. — [DiffLocks](https://arxiv.org/pdf/2505.06166), [HairWeaver](https://arxiv.org/pdf/2602.11117)
- VideoMaMa is trained solely on synthetic data yet generalizes zero-shot to real footage, then is used to label MA-V. — [VideoMaMa](https://arxiv.org/html/2601.14255v1)
- VideoMaMa ablation (V-HIM60 Hard MAD, lower better): composition datasets only 7.58; MA-V (real footage, pseudo-alpha) only 3.18; both 2.61. Also authors suggest synthetic composite datasets bias DAVIS tracking robustness (87.9 MA-V only vs 85.9 combined J&F). — [VideoMaMa](https://arxiv.org/html/2601.14255v1)
- RVM notes models overfit the synthetic matting distribution even when initialized from segmentation weights; joint segmentation training regularizes. — [RVM WACV 2022 via search](https://openaccess.thecvf.com:443/content/WACV2022/papers/Lin_Robust_High-Resolution_Video_Matting_With_Temporal_Guidance_WACV_2022_paper.pdf)
- Netflix Magenta Green Screen authors state their alpha could provide higher-quality training data for natural-image matting. — [arXiv 2306.13702](https://arxiv.org/abs/2306.13702)

### Inferences
- Because composition-only data lost badly to real-footage data in VideoMaMa's ablation (7.58 vs 3.18), your renders should be composited as full scenes (subject shadows, spill, backing pass), which your pipeline already does, rather than cut-and-paste. That is the right design and is more like MA-V's "foreground and background co-occur" property.
- Rank of realism factors (my judgment, not measured in sources): hair/edge translucency and subpixel coverage with motion blur > spill and bounce coupling to the matte > camera/ISP/codec chain > geometry detail. Plan an ablation on your own held-out real green-screen clips rather than trusting this.
- MPFB/MakeHuman (CC0 humans) and Infinigen were not researched; recommend as the human source only after a quick license check on skins/hair assets.

### Gaps
- No published per-factor ablation (strand count, motion blur, bounce, ISP) for matting found. No source found for MPFB, Infinigen, or Microsoft synthetic faces/hair for matting.
- SynthHair strand-level rendering details and release not available.

## Q3. Real ground-truth capture (triangulation, Magenta, multi-background, light stage)

### Takeaway
Time-multiplexed lighting and background-swap capture are now credibly used for exact alpha on real footage (Magenta, CineMatte, PolarMatte). I found no ablation quantifying how much a small real-GT set helps matting training, so the benefit is unquantified here.

### Cited Findings
- Magenta Green Screen: time-multiplexes red/blue and green/magenta lighting so it runs under nearly normal lighting; the green channel with the actor silhouetted against a bright background acts as a holdout matte; demonstrated on an LED virtual production stage; authors suggest data utility for matting. — [arXiv 2306.13702](https://arxiv.org/abs/2306.13702)
- CineMatte-4K: capture by swapping the LED wall to green with the actor still; 4K HDR; human-expert alpha; tracked camera trajectories for later re-rendered backgrounds. — [arXiv 2605.18328](https://arxiv.org/abs/2605.18328)
- PolarMatte (Adobe, CVPR 2024): ground-truth-quality alpha for dynamic scenes via a polarization camera in front of an LCD backlight. — [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Enomoto_PolarMatte_Fully_Computational_Ground-Truth-Quality_Alpha_Matte_Extraction_for_Images_and_CVPR_2024_paper.pdf)
- MatAnyone notes it fine-tunes on image matting data because video matting data is of inferior quality compared with human-annotated image matting. — [MatAnyone supplement via search](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Yang_MatAnyone_Stable_Video_CVPR_2025_supplemental.pdf)

### Inferences
- Practical in-house setup: a still subject (mannequin head, wigs, fur swatches, cloth, glass) in front of an LCD/monitor or a green screen plus a swappable second background gives triangulation (Smith-Blinn) alpha; two or more known backgrounds solve alpha and foreground per pixel. This is cheap and static-only, so it validates edge/hair/transparency, and cannot supply motion blur.
- Even ~100 to 300 exact real static captures are useful as a calibration set and as a validation set for the synthetic domain, and may help as a final low-LR fine-tune. Unquantified; flagged.

### Gaps
- No source quantifying gain from a small real-GT set. No detail from Smith-Blinn original or the light-stage literature was fetched.

## Q4. Semi/self-supervised real-data training

### Takeaway
Strongest evidence: MatAnyone 2 curates real-footage pseudo-labels with a learned pixel-wise quality evaluator and trains only where the evaluator trusts the label, yielding consistent gains. Composition/consistency losses are an established complement.

### Cited Findings
- MQE: predicts a pixel-wise reliable/erroneous map without GT; used online (loss masked to evaluation-map = 1, plus 0.1 x L_eval) and offline (curate/fuse pseudo-labels). Trained 80K iterations on P3M-10k pseudo-labels with DINOv3 encoder + DPT decoder. — [MatAnyone 2 HTML](https://arxiv.org/html/2512.11782)
- Dual-branch labeling: video branch (MatAnyone) for temporal stability plus image branch (MattePro with SAM 2 prompts) for sharp edges, fused by MQE maps with a 9x9 Gaussian blur (sigma 5). — same source.
- Ablation on YouTubeMatte 1080p (MAD / MSE / Grad / dtSSD): baseline MatAnyone 1.99/0.71/8.91/1.65; + L_eval 1.90/0.62/8.20/1.63; + VMReal 1.76/0.61/7.65/1.54; + reference-frame training 1.61/0.50/7.13/1.53. VMReal on RVM: MAD 4.27 to 3.37. — same source.
- MatAnyone 2 trains on VMReal alone (no joint segmentation data), 480x480 crops, 8-frame clips, 8 A800 GPUs, batch 16. — same source.
- VideoMaMa/MA-V: pseudo-alpha on 50K real videos; MA-V alone beat composition datasets by a wide margin and combining was best (2.61 MAD). — [VideoMaMa](https://arxiv.org/html/2601.14255v1)
- Background Matting (2020) closed the gap with a discriminator judging composite quality on unlabeled real images. — [arXiv 2004.00626](https://arxiv.org/pdf/2004.00626)

### Inferences
- A workable plan: train your synthetic-supervised model, run it plus 1-2 other teachers (MatAnyone 2, VideoMaMa/SAM2Matting if licenses allow, as teachers only) on unlabeled real green-screen footage, fuse by agreement and a small learned or heuristic quality map (green-screen clips offer an extra reliability signal: reconstruction error of C = alpha F + (1-alpha) B with estimated clean-plate B), and train only on high-confidence pixels. Note teacher licenses may taint outputs; check.
- The MatAnyone 2 gains are modest per step (about 0.1 to 0.25 MAD each) but cumulative (1.99 to 1.61, ~19%); expect similar order of gain, not a regime change.

### Gaps
- No test-time consistency evidence retrieved. No numbers for a MQE-like model trained on green-screen-specific footage.

## Q5. Degradation/augmentation modeling for real cameras

### Takeaway
Restoration literature supports high-order (repeated) degradation synthesis for blind robustness; matting papers mostly use light augmentations. No matting paper I found ablates codec/ISP simulation, so your real ffmpeg round-trips are ahead of the published norm.

### Cited Findings
- Real-ESRGAN: high-order degradation (blur, resize, noise, JPEG repeated in rounds, sinc filter for ringing/overshoot) trained with pure synthetic data for real-world blind SR; claims superior visual performance on real datasets. — [Real-ESRGAN](https://arxiv.org/abs/2107.10833v2), [ICCVW](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html)
- Matting augmentations used in prior work: re-JPEG and Gaussian blur (Context-Aware), pixel-level linear/non-linear/region-wise (RMat); mask-guided matting uses contrast, brightness, channel shuffle, Gaussian noise, plus blur and JPEG region-level. — [Matting survey](https://arxiv.org/pdf/2304.04672), [Park CVPR 2023 supp](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Park_Mask_Guided_Matting_in_CVPR_2023_supplemental.pdf)
- GFM attributes the composite gap to resolution, sharpness, noise, illumination differences. — [arXiv 2010.16188](https://arxiv.org/pdf/2010.16188)

### Inferences
- Apply degradations to the composite only (never to the GT alpha), with chroma subsampling 4:2:0/4:2:2 and H.264 at varied CRF applied after compositing, since chroma subsampling is the specific failure mode for green-screen keying (low-res chroma at edges). Randomize order and repeat rounds (high-order) at low probability.
- Also train with clean inputs part of the time so the model does not trade sharpness for robustness.

### Gaps
- No matting-specific evidence on codec, demosaic, rolling shutter, or lens degradation; RealBasicVSR results not retrieved.

## Q6. Mixture ratios, curriculum, data volume for a 665M fine-tune

### Takeaway
No published recipe for this exact scale. The available evidence suggests real-footage pseudo-labeled data plus a clean-GT anchor outperforms either alone, and that tens of thousands of clips were used at the scale of MatAnyone 2 (a much smaller model).

### Cited Findings
- MatAnyone 2: 28K clips / 2.4M frames, single dataset, short clip training at 480x480. — [MatAnyone 2](https://arxiv.org/html/2512.11782)
- VideoMaMa: best result came from mixing composition datasets with MA-V (2.61) over either alone (7.58 / 3.18). — [VideoMaMa](https://arxiv.org/html/2601.14255v1)
- RVM: joint segmentation training regularizes overfitting to synthetic matting. — [RVM](https://openaccess.thecvf.com:443/content/WACV2022/papers/Lin_Robust_High-Resolution_Video_Matting_With_Temporal_Guidance_WACV_2022_paper.pdf)
- MatAnyone ends with image matting fine-tuning for sharper edges. — [MatAnyone supp](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Yang_MatAnyone_Stable_Video_CVPR_2025_supplemental.pdf)

### Inferences (my proposal, unvalidated; test with one verifying run)
- Stage 1, mostly synthetic with exact GT: 50% own Blender (all passes), 20% CorridorKeyDataset if granted, 15% VideoMatte240K / VM800-style green-screen pulls (down-weight edge-contaminated clips, use erosion-aware loss), 15% image sets (P3M, AM-2k, Transparent-460, AIM, Distinctions) at low rate, with image sets compositing against your stage simulator rather than COCO backgrounds.
- Stage 2, shift to real: 40% pseudo-labeled real green-screen/portrait footage (MQE-style masked loss), 40% synthetic, 20% exact-GT real captures plus image sets. Final low-LR phase on cleanest GT for edges.
- Volume: a fine-tune of a 665M model likely needs 10^4 to 10^5 distinct clips/images with heavy augmentation; the existing Blender generator at 10^4 to 10^5 frames across varied scenes is plausible on 3 A6000s. Treat these as starting guesses.
- Keep eval real: hold out CineMatte-4K / YouTubeMatte / own real captures, never tune on them.

### Gaps
- No source on mixture ratios or sample counts for models of this size.
