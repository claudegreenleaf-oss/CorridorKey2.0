# Novel techniques to push an enhanced green/blue-screen video keyer (CorridorKey 2.0) past the state of the art

Scope note. These notes invent techniques and then justify them. "Cited Findings" holds only sourced facts (prior art and adjacent-field evidence). Everything that is my own design, arithmetic or prediction sits under "Inferences" and should be read as a hypothesis until the named cheapest experiment is run. Illustrative numbers in Inferences are back-of-envelope values chosen to be plausible for linear-light green screen footage; they are not measurements from the benchmark.

Code anchors checked in the local repo: `corridorkey/plate_fg.py` (`pull_push`, `estimate_backing`, `despill_backing`, `plate_foreground`), `corridorkey/stage2.py` (per-frame /16 attention, `Depatchify(d, out_channels=4, r=16)` with PixelShuffle(16), and an optional zero-init `Stage2Refine4Head` that reads an exact RGB pixel-unshuffle at /4), `corridorkey/color_utils.py` (signed asinh scene-linear encoding).

## Q0. What does the physics leave on the table? (framing used by every idea below)

### Takeaway
With the backing B known, single-backing keying has exactly one unknown scalar per pixel (alpha); every remaining error is a wrong point on a known line in colour space. The ways to beat the state of the art are therefore: get B more exactly, add independent equations (other frames, other backings), read alpha straight from full-resolution pixels where the signal-to-noise ratio allows it, and let a learned prior decide only where it does not.

### Cited Findings
- Smith and Blinn prove constant-colour (single backing) matting is "unsolvable as generally practiced" and that a full solution exists when the object is shot against two backings that differ everywhere ("triangulation matting"); they also warn triangulation is sensitive to brightness and misregistration errors and errs toward transparency ("fine line" problem). [Smith & Blinn 1996](https://alvyray.com/Papers/CG/blusig96.pdf)
- The triangulation estimate for two known backings is alpha_T = (I0 - I1)/(B0 - B1) - 1 (equivalently 1 - alpha = (I0 - I1)/(B0 - B1)), then alpha_T F_T = I0 - (1 - alpha_T) B0. [McGuire et al. 2006, polarization triangulation](https://casual-effects.com/research/McGuire2006PolarizationMatte/polarization-siggraph06.pdf)
- FBA Matting fuses predicted F, B and alpha by a maximum-likelihood step whose alpha update is alpha = (alpha_hat*la + (I - B).(F - B)) / (||F - B||^2 + la), i.e. a regularised projection of I onto the F-B line; one iteration was enough and it "reliably" improved results. [Forte & Pitie 2020](https://ar5iv.labs.arxiv.org/html/2003.07711); code form in [LearnOpenCV write-up](https://learnopencv.com/image-matting-with-state-of-the-art-method-f-b-alpha-matting/)
- Robust Video Matting (Lin et al.) argues temporal information "allows the model to learn more about the background over time" because background behind the subject is revealed by camera or subject motion. [RVM, Lin et al. 2021](https://ar5iv.labs.arxiv.org/html/2108.11515)

### Inferences
- Given B, the solution family per pixel is F(alpha) = B + (I - B)/alpha. Differentiating, dF = -(F - B) * dalpha / alpha. Foreground error is the alpha error scaled by |F - B|/alpha, so at alpha = 0.1 an alpha error of 0.02 becomes a foreground error of 0.2|F - B|. This is why plate-recovered FG on fine hair is noise-limited: the FG metric is mostly an alpha metric in disguise at low alpha.
- With a fixed F estimate, the least-squares alpha is the projection alpha = <I - B, F - B>/||F - B||^2, and with isotropic per-channel noise sigma its standard deviation is sigma/||F - B||. Illustrative: linear green backing B approx (0.05, 0.35, 0.08), brown hair F approx (0.30, 0.20, 0.12) gives ||F - B|| approx 0.29; at sigma = 0.005 (clean camera) sigma_alpha approx 0.017, at sigma = 0.02 (low light) approx 0.07. Green wardrobe F approx (0.08, 0.25, 0.08) gives ||F - B|| approx 0.10, three times worse. This single quantity, the local "keying SNR" ||F - B||/sigma, predicts which benchmark shots are hard (low light, compression, green wardrobe, smoke, glass) and is used repeatedly below.
- Errors in B enter alpha with weight (1 - alpha): every reduction in the noise or bias of B_hat translates directly to alpha in the semi-transparent band. Pull-push B_hat interpolates the screen under the subject from its rim; it cannot know shadows, hot spots or marker positions behind the subject.

### Gaps
- I did not find published measurements of real stage screen unevenness (percent luminance variation) or camera noise in linear units for the specific benchmark cameras; the illustrative numbers should be replaced with values measured from the benchmark's backing regions.

## Idea 1. Backing memory: shot-level clean-plate accumulation in screen coordinates with a multiplicative shadow/exposure factor

### Takeaway
Replace the single-frame pull-push B_hat with a per-shot "backing memory" that accumulates every pixel of screen ever revealed (in screen coordinates for moving cameras), factor the current frame's backing as gain x shadow x plate, and feed B_t plus an observed/interpolated confidence channel to both stages. Highest expected value of all ideas; prior art exists for the clean plate itself, not for this automatic, factorised, learned-conditioning form.

### Cited Findings
- Chuang et al. 2002 estimated background for video matting by masking foreground with user-drawn garbage mattes and filling each frame's missing background "with the color from the temporally nearest frame that contains a background color for that pixel"; with the estimated background, Bayesian matting gets "much improved" mattes and a 10x speedup, and errors in the background plate "can seriously degrade the quality of the alpha matte". They also showed smoke matting against a known background by intersecting colour lines to find the smoke colour. [Chuang et al. SIGGRAPH 2002](https://grail.cs.washington.edu/projects/digital-matting/papers/sig2002.pdf)
- Background Matting (Sengupta et al. 2020) requires a separately captured background photo; it notes it needs a static background and small camera motion. [Sengupta et al. CVPR 2020](https://doi.org/10.1109/cvpr42600.2020.00236)
- BGMv2 also takes a captured background frame and simulates misalignment and "artificial shadows to simulate how the presence of a subject can cast shadows" during training. [Lin et al. CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)
- Robust Video Matting learns background implicitly via recurrence rather than as an explicit plate. [RVM](https://ar5iv.labs.arxiv.org/html/2108.11515)

### Inferences
- Mechanism. (a) Run a cheap first pass (the current model at reduced resolution, or the previous window's output) to get alpha_t. (b) Register frames to screen coordinates: identity for locked-off shots; a per-frame homography of the screen plane for moving cameras, estimated on pixels with alpha < 0.02 (tracking markers exist precisely to make this solvable); MoGe-2 geometry (already a stage-1 input) to split a cyc into two or three planes. (c) Accumulate a plate P(x) as a (1 - alpha)^k weighted robust mean (or median) with a per-pixel count. (d) Model the current backing as B_t(x) = g_t * s_t(x) * P(x), where g_t is a global exposure/flicker gain fitted on always-visible screen, and s_t is a smooth multiplicative field (subject shadow < 1, subject bounce > 1) predicted by the network at /16. (e) Feed asinh(B_t), log s_t and a confidence channel (observed count versus pull-push-interpolated) into stage-1 /16 tokens and the stage-2 patchify, zero-initialised so the pretrained model is unchanged at insertion.
- Why it should work. Plate noise falls as sigma/sqrt(K) with K revealing frames, so B error in the (1 - alpha) term nearly vanishes on locked-off shots; markers, seams and hot spots behind the subject become known instead of interpolated; anything that matches the plate is background regardless of colour, which directly addresses the "keyed a tracking marker" failure. The Blender generator already outputs the clean backing including the subject's real shadow, so s_t has exact supervision (one extra render of the screen with the subject removed gives P).
- Benchmark targets: real stage (uneven screens, markers, spill), smoke (difference against a true plate, as Chuang did), low light and both codec shots (plate averages out noise and blocking), glass (refracted background differs from plate, giving a refraction cue), subject selection.
- Bonus deliverable: s_t is a usable shadow pass for compositors.
- Cost. Train: +5 input channels, negligible VRAM; data needs P per clip. Inference: one low-res pre-pass plus a plate buffer (one 4K float image per shot); for 80-frame windows it fits easily. Effectively free compared with a 665M model.
- Risks. Subject that never moves (falls back to pull-push; the confidence channel tells the model); lighting changes mid-shot (g_t absorbs global ones only); parallax on non-planar screens; a first-pass alpha that is wrong poisons the plate (use conservative thresholds and the median).
- Cheapest experiment (no training): on synthetic benchmark shots, swap the GT clean backing for B_hat inside the existing `plate_foreground` and `despill_backing` and measure FG and despill error. This is the ceiling for B quality. Then build P from the real-stage shot using the current model's own alpha and compare plate-recovered FG against the pull-push version. If the oracle ceiling is small, kill the alpha-conditioning part and keep only the FG improvement.

### Gaps
- No published learned keyer that conditions on an automatically accumulated, factorised plate was found in these searches; absence of evidence is not proof of novelty, and commercial keyers' internals (Nuke IBK, Keylight) were not checked.

## Idea 2. Full-resolution physics projection head with SNR-weighted fusion

### Takeaway
Let the network predict, at /16 or /4, only what is smooth (local foreground colour F_loc, its uncertainty, and a fusion weight); compute alpha at /1 analytically as the projection of the raw full-resolution pixel onto the F_loc to B_t line, and blend with the network alpha by expected keying SNR. FBA's fusion is the closest prior art; what is new is a known/accumulated B, a deliberately low-frequency F, the noise-aware weight, and use as the /1 output path of a /16 transformer.

### Cited Findings
- FBA fusion re-estimates alpha from F and B by a regularised projection and found one iteration "reliably" better. [Forte & Pitie 2020](https://ar5iv.labs.arxiv.org/html/2003.07711)
- Chuang's smoke matting treats smoke as one constant colour composited with varying alpha over a known background and estimates alpha per pixel by projecting onto the line between background and smoke colour. [Chuang et al. 2002](https://grail.cs.washington.edu/projects/digital-matting/papers/sig2002.pdf)
- Real-world green-screen keyers in the literature already combine a deep prior with a per-pixel linear model for real-time high-resolution keying (deeply guided linear models; attention plus deep guided filter at 2K). [Li et al. 2021](https://www.mdpi.com/2073-8994/13/8/1454); [Jin et al. 2025](https://doi.org/10.3724/sp.j.1089.2023-00335)

### Inferences
- Mechanism. Stage 2 outputs F_loc (straight colour of the nearest opaque material, constrained smooth by predicting at /16 and bilinearly upsampling), a log-variance for F_loc, and a fusion logit. At /1: alpha_phys = clamp(<I - B_t, F_loc - B_t>/(||F_loc - B_t||^2 + lambda*sigma^2)); w = sigmoid(fusion logit + log(||F_loc - B_t||^2/sigma^2)); alpha = w*alpha_phys + (1 - w)*alpha_net. sigma comes from Idea 6's noise map. Cost is a handful of elementwise ops per pixel.
- Why it should work. The current /1 path must carry exact sub-pixel strand geometry through a 1280-d token and a linear PixelShuffle(16). The projection reads it directly from the plate at zero learned cost. It is also exact under motion blur and defocus against a static smooth backing, because blurring is a linear mix: the time-integrated plate equals blurred-alpha times F plus (1 - blurred-alpha) times B, so the projection returns the correctly blurred alpha (benefits whip motion blur and focus pull). It is a deterministic function of the pixels, so shimmer in the projection term only comes from input noise, not from network instability.
- Targets: 4K hair, fur, 4K spider web, sub-pixel threads, motion blur, focus pull, smoke (F_loc becomes the smoke colour).
- Cost. Training: tiny head; inference: negligible. Composes with the existing `Stage2Refine4Head` (put F_loc and w there at /4).
- Risks. Noise passes straight into alpha_phys where ||F - B|| is small (green wardrobe, low light, crushed chroma); the SNR weight is the defence and must be trained under realistic noise. Strands whose colour differs from the bulk (blonde highlights over dark hair) violate the smooth-F assumption. Spill makes F_loc itself drift toward B, lowering apparent SNR (Idea 4's spill supervision helps).
- Cheapest experiment (no training, an afternoon): on synthetic benchmark shots compute alpha_phys with (a) oracle F_loc = GT FG blurred to /16 and B = GT backing, (b) F_loc from the current model's FG at /16 and B = pull-push B_hat. Measure alpha SAD and gradient error in the 0 < alpha < 1 band against the current model. If (a) does not beat the model on hair, fur and the spider web, kill the idea; if (a) wins but (b) does not, the bottleneck is B and F_loc quality (do Idea 1 first).

### Gaps
- No published measurement of how much of a transformer matting model's fine-detail error is attributable to the depatchify bottleneck versus semantic error; the oracle experiment above is the measurement.

## Idea 3. Passive motion triangulation: Smith-Blinn from subject or camera motion over a non-uniform backing

### Takeaway
Every frame in which a tracked subject point sits over a different backing value adds three equations and no unknowns, so natural screen unevenness, seams and especially tracking markers give a free, passive version of two-backing triangulation. Real but narrow: strong only where the backing has high contrast structure; speculative because it needs sub-pixel-accurate correspondence on semi-transparent detail.

### Cited Findings
- Triangulation requires the subject to be identical against two backings and is sensitive to misregistration; Smith and Blinn note it works only for non-moving objects (excluding rigid motions such as simple translation). [Smith & Blinn 1996](https://alvyray.com/Papers/CG/blusig96.pdf)
- Wexler, Fitzgibbon and Zisserman observe that in multiple images "a given foreground pixel will sweep over several background pixels, and so a constraint is available on the foreground's colour". [Wexler et al. 2002](https://www.robots.ox.ac.uk/~vgg/publications/2002/Wexler02/wexler02.pdf)
- Active versions exist in hardware: alternating green/blue LED backdrops at 100 fps with a dual-decoder network (Hachmann & Rosenhahn 2023), polarised dual-sensor capture (McGuire 2006). [Hachmann & Rosenhahn 2023](https://arxiv.org/html/2306.02954); [McGuire et al. 2006](https://casual-effects.com/research/McGuire2006PolarizationMatte/polarization-siggraph06.pdf)

### Inferences
- Mechanism. For each pixel, warp I and B_t from neighbour frames along flow (RAFT or codec motion vectors), form the pairwise estimate 1 - alpha = <I_a - I_b, B_a - B_b>/||B_a - B_b||^2 and its Fisher weight ||B_a - B_b||^2/(2 sigma^2), and pass the weighted mean and total information as two extra /1 or /4 channels (or as trajectory-attention features in Idea 7).
- Illustrative SNR. 10% screen unevenness on a 0.35 green gives ||delta B|| approx 0.035 and sigma_alpha approx 40 sigma per pair (0.2 at sigma = 0.005): nearly useless. A white marker over green gives ||delta B|| approx 0.7 and sigma_alpha approx 2 sigma (0.01): excellent. Shadow edges and panel seams are in between. Conclusion: the cue is concentrated exactly where the tracking-marker failure happens and where camera moves sweep markers behind hair.
- Targets: real stage with markers, camera-move shots, the marker subject-selection failure.
- Cost: flow at /1 or /2 (RAFT-class at 4K is non-trivial; codec MVs are free but coarse). Risk: flow on thin semi-transparent strands is unreliable (the strand and the backing move differently, so a single flow vector per pixel is ill-defined), and errors push alpha toward transparency exactly as Smith and Blinn warn.
- Cheapest experiment: render a hair clip moving over a screen with procedural unevenness plus markers, compute the oracle-flow triangulation estimate and its error versus GT as a function of ||delta B||. Two hours, no training. If error with oracle flow is not below the current model's error over markers, kill.

### Gaps
- No published learned keyer exploiting passive motion triangulation was found; the closest are the hardware duplex systems above.

## Idea 4. Backing algebra: light-group rendering for exact-spill backing recolouring and paired green/blue consistency

### Takeaway
Render the screen's emission as its own Cycles light group (split into a grid of panels for unevenness). Because light transport is linear in emitter radiance, one render then yields unlimited backing colours, unevenness patterns and flicker with physically exact spill, bounce and shadows, plus exact paired green/blue versions of the same frame for a backing-invariance loss. Cheap, high-confidence data win.

### Cited Findings
- Cycles Light Groups produce a "limited Combined render pass where the scene is only illuminated by certain lights"; passes can be added to reconstruct the full render, and recombining them makes it "possible to change the color and intensity of individual lights without having to re-render". The World background can also be assigned to a light group. [Blender manual, Passes](https://docs.blender.org/manual/en/latest/render/layers/passes.html); introduced in Cycles 3.2. [Blender 3.2 release notes](https://developer.blender.org/docs/release_notes/3.2/cycles/)
- A dual-backdrop (green then blue) network was shown superior to trimap input and gave better spill compensation; it also observed chromatic aberration makes alpha systematically overestimated on green and underestimated on blue backdrops. [Hachmann & Rosenhahn 2023](https://arxiv.org/html/2306.02954)
- Prior learned green-screen keyers built datasets "with green spill phenomenon and a large number of distinct green screen backgrounds", but by compositing, not by relighting. [Automatic and real-time green screen keying, 2022](https://link.springer.com/article/10.1007/s00371-022-02542-x)

### Inferences
- Mechanism. Put the subject's lights in group S and the screen (one group per panel of a coarse grid, e.g. 4x4) in groups P_k. Then I(c) = L_S + sum_k c_k (per-channel) * L_Pk reproduces the camera-visible screen, spill on the subject, bounce and the subject's shadow on the screen, all exactly for any panel colours c_k. Spill-free FG = subject contribution with screen groups zeroed; spill = sum_k c_k L_Pk restricted to the subject. Augment every training sample with random c_k (green, blue, off-hue, desaturated, uneven, flickering per frame). Additionally sample the same frame under a green and a blue c and add a consistency loss on alpha and on spill-free FG (stop-gradient on the lower-loss member).
- Why it should work. It removes the shortcut "green means background": the model must use agreement with the local backing estimate. That is what generalises to green wardrobe (wardrobe is not backing-coloured relative to a blue screen in the paired sample), blue screen, and off-hue real stages. Exact spill under recolouring also gives supervision for Idea 10's spill subspace for free.
- Targets: green wardrobe, blue screen, real stage uneven screens and spill, glass and smoke (spill-like transmission changes with backing).
- Cost. Render: extra passes cost memory and disk, not much render time; training: no extra VRAM for augmentation, 2x forward for the paired loss on a fraction of batches.
- Risks. Light groups cover emitted light; a screen modelled as a diffusely lit grey card (rather than an emissive surface) would need its key lights in the group instead. Denoiser and clamp settings must be identical across groups or the sum will not equal the combined pass. Chromatic aberration (Hachmann) is a lens effect not captured by recolouring; add it as a separate augmentation.
- Cheapest experiment: one scene, render combined plus groups; verify sum of groups equals combined (PSNR > 50 dB) and that a recoloured composite matches a true re-render with the screen recoloured. Half a day. Then fine-tune a short run with recolouring augmentation only and compare green-wardrobe and blue-screen shots.

### Gaps
- I did not verify in the manual whether emissive mesh objects (as opposed to Light objects and the World) can be assigned to light groups in the installed Blender version; check `Object > Shading > Light Group` on an emissive plane.

## Idea 5. TTT-Key: per-shot test-time training with a degeneracy-aware physics loss

### Takeaway
Adapt a small LoRA on stage 2 to each shot with self-supervised losses that are not trivially satisfiable: reconstruction against a smooth F and the backing memory, certain-background pseudo-labels, flow consistency, and an anchor to the frozen model. Evidence from video TTT is strong under severe distribution shift, which is exactly the crushed-codec and low-light weakness. Offer it as an offline "final quality" mode.

### Cited Findings
- Online test-time training on video streams improved instance and panoptic segmentation by 45% and 66% relative (more than 2.2x and 1.5x in the JMLR version) over a fixed model; online TTT with a short window beat offline TTT on the whole video; one gradient step per frame realised most of the gain; adapting normalisation layers alone did not help on real data. [Wang et al., Test-Time Training on Video Streams](https://video-ttt.github.io/); [JMLR version](https://jmlr.org/papers/volume26/24-0439/24-0439.pdf)
- Self-supervised test-time adaptation on single videos gave small gains under mild shift but large ones under severe shift (for example +18 to +20 J&F under Gaussian noise for MAST). [Azimi et al., NeurIPS 2021 SSL workshop](https://sslneurips21.github.io/files/CameraReady/SSL_TTA.pdf)
- Background Matting's real-data stage combined an adversarial composite loss with a teacher (pseudo-label) term and a reconstruction term ||I - alpha F - (1 - alpha) B'||; without the teacher the network could collapse to alpha = 1 everywhere, which trivially satisfies the composite. [Sengupta et al. 2020](https://doi.org/10.1109/cvpr42600.2020.00236)

### Inferences
- Degeneracy analysis (key point). With a free per-pixel F, the reconstruction I = alpha F + (1 - alpha) B is satisfied for every alpha along the line F(alpha), so its gradient on alpha is zero. Reconstruction is informative only if F is constrained to be smooth (F_loc at /16, as in Idea 2). Then the reconstruction residual at /1 is exactly the projection residual, and TTT becomes per-shot self-distillation of the physics estimate into the network where SNR is high, while the network prior governs where it is low.
- Loss: L = w_rec * ||I - alpha F_loc - (1 - alpha) B_t||_rho (robust, weighted by SNR) + w_bg * |alpha| on pixels where the backing memory observes background with |I - B_t| < 3 sigma + w_temp * |alpha_t - warp(alpha_{t+1})| on non-occluded pixels + w_anchor * |alpha - alpha_frozen| (decaying) + spill-budget penalty on F_loc's component along B_t.
- Plug-in: cache stage-1 latents (frozen), LoRA rank 8 to 16 on stage-2 qkv/proj and the depatchify, train on 512 to 1024 px crops, online over the shot (Wang's finding favours online with a short window).
- Targets: crushed web codec, camera H.264, low light, real stage. Little expected on clean renders (Azimi's mild-shift result).
- Cost: inference 2x to 4x slower per frame; VRAM for stage-2 backprop on crops with gradient checkpointing should fit on a 48 GB A6000 (estimate, not measured).
- Risks: confirmation bias (the anchor and pseudo-labels re-teach the model's own mistakes); overfitting noise into alpha; per-shot hyperparameters. Gate per pixel by the uncertainty from Idea 9.
- Cheapest experiment: crushed-web-codec and low-light shots only, 1 LoRA step per frame online, losses rec + temp + anchor; compare alpha SAD and the temporal-consistency metric with and without. One evening.

### Gaps
- No published TTT for matting/keying was found; the nearest is Sengupta's offline self-supervised fine-tune on real videos (not per shot).

## Idea 6. Degradation side-channel: codec-native chroma path, QP and motion vectors, and a screen-calibrated noise map

### Takeaway
Tell the network what the pixels can be trusted for. Feed luma at full resolution and Cb/Cr at native 4:2:0 resolution as separate patchify streams, add per-block QP and frame type, and a noise-level map calibrated per shot on the screen itself (the backing is a giant flat calibration chart). Codec side information is proven in video super-resolution; noise maps are proven in denoising; neither has been applied to keying.

### Cited Findings
- COMISR showed that simply training VSR models on compressed frames "brings only modest improvement" and that compression-specific modules were needed. [Li et al. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.pdf)
- Coding priors (partition maps, motion vectors) extracted from the bitstream improve compressed-video SR. [Chen et al., Compressed Domain Deep Video SR](https://pubmed.ncbi.nlm.nih.gov/34370665/)
- Motion-vector alignment matched optical-flow alignment at negligible cost, and residual-guided sparse processing saved up to 70% compute at CRF 23 without quality loss. [CIAF, 2022](https://arxiv.org/html/2210.08229)
- CDA-VSR (CVPR 2026) uses motion vectors, residual maps and frame types for MV-guided deformable alignment and residual-gated fusion. [Wang et al. CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Compressed-Domain-Aware_Online_Video_Super-Resolution_CVPR_2026_paper.html)
- QP-adaptive compressed VSR with coding priors (2025). [Zhang et al., Signal Processing 2025](https://doi.org/10.1016/j.sigpro.2024.109878)
- FFDNet takes "a tunable noise level map as the input", handling a wide range of noise levels and spatially variant noise with one network. [Zhang, Zuo, Zhang](https://arxiv.org/abs/1710.04026)

### Inferences
- Why chroma resolution matters for keying. Keying is chroma-driven, and in 4:2:0 the chroma is sampled at half resolution per axis, so a decoder's upsampled RGB presents interpolated chroma as if it were per-pixel detail. Luma alone carries a much weaker keying contrast (illustrative: Rec.709 luma difference between the hair and green examples in Q0 is about 0.05 versus a full-colour distance of about 0.29) but at full resolution. The optimal estimator fuses low-frequency alpha from chroma with high-frequency alpha from luma given the local F - B luma difference. A network that receives upsampled RGB has to learn this implicitly and cannot know which frequencies are real.
- Noise map from the screen. Fit a signal-dependent variance sigma^2(level) per channel from I - B_t over confident-background pixels (Idea 1 makes this exact); add QP-derived block noise for compressed sources. This sigma feeds Idea 2's SNR weight and Idea 9's routing.
- Plug-in: two zero-init patchify branches (Y at /1, CbCr at /2) beside the existing RGB patchify; side-info channels drop out randomly during training so ProRes/EXR inputs (no 4:2:0, no QP) still work. The fine-tune pipeline already does real codec round trips, so QP and MVs can be logged at encode time.
- Targets: camera H.264, crushed web codec, low light; some 4K hair benefit where sources are 4:2:0.
- Cost: small (a few convs); data plumbing for side info.
- Risks: side info absent or different for other codecs (ProRes RAW, AV1); network overfits to encoder-specific QP semantics.
- Cheapest experiment (no training): on the H.264 shot, replace the decoder's chroma upsampling with luma-guided chroma upsampling (joint bilateral or guided filter) before the current model. If edge alpha error drops noticeably, the model is chroma-resolution-limited and the native-luma path is worth building; if not, drop the chroma path and keep only the noise map.

### Gaps
- No measurement exists yet of how much of the codec shots' error is chroma subsampling versus blocking versus noise; the experiment above separates the first.

## Idea 7. Fisher-weighted trajectory fusion in stage 2 (anti-shimmer)

### Takeaway
Stage 2 is per-frame, so fine-detail alpha is regenerated independently each frame and shimmers. Add flow-guided cross-frame attention (or a warped recurrent state) in the last stage-2 blocks and a /1 temporal fusion of alpha along trajectories, weighted per frame by the physical information each frame carries about that point (||F - B_t||^2/sigma_t^2, motion blur, occlusion).

### Cited Findings
- Trajectory attention pools information "along implicitly determined motion paths", outperforming joint and divided space-time attention; per-frame (spatial) softmax normalisation and 1D temporal attention were each worth several points. [Patrick et al., Motionformer, NeurIPS 2021](https://arxiv.org/html/2106.05392v2)
- Chuang et al. flowed information through the video volume in both directions and combined estimates using accumulated flow-error maps as reliability. [Chuang et al. 2002](https://grail.cs.washington.edu/projects/digital-matting/papers/sig2002.pdf)
- Handheld multi-frame super-resolution aligns and merges a burst with a robustness model weighting each frame per pixel, increasing resolution and SNR from natural sub-pixel tremor. [Wronski et al. 2019](https://arxiv.org/html/1905.03277v2)

### Inferences
- Mechanism. In the final two stage-2 blocks, add keys/values from frames t-1 and t+1 sampled at flow-displaced positions (deformable sampling seeded by RAFT or codec MVs, as CDA-VSR does for VSR), with an occlusion gate. At /1, fuse alpha_phys (Idea 2) across K frames with weights proportional to ||F_loc - B_t||^2/sigma_t^2 times a flow-reliability term. Frames where a strand sits over brighter, more contrasting or less noisy backing get more say.
- Multi-frame super-resolution considered and rejected as a separate idea: at native output resolution the GT alpha of a sub-pixel thread is the box-filtered coverage, which a single frame already measures; extra frames buy noise reduction and stability (this idea), not resolution, unless output is above native resolution.
- Targets: shimmer on 4K hair, fur, spider web; low light (temporal averaging of alpha, roughly sqrt(K) noise reduction on static detail).
- Cost: roughly 3x keys/values in two blocks; flow at /2 or /4. Moderate.
- Risks: flow is ill-defined on semi-transparent strands over a moving or blurred background, causing ghosting on whip motion blur and fast hair; must gate by flow-consistency error.
- Cheapest experiment (no training): post-hoc flow-guided temporal median of the current model's alpha (RAFT on the plate) on hair, fur and spider-web shots; measure dtSSD-style temporal error and alpha SAD. If the post-hoc filter captures most of the gain, skip the architectural change and ship a fusion post-process.

### Gaps
- No published flow-guided fusion specific to matting at /1 with physics-derived weights was found.

## Idea 8. Colour transmission and refraction for glass (environment-matte lite)

### Takeaway
Scalar alpha cannot represent tinted glass or refraction. Output a three-channel transmission (per-channel alpha) and an optional /4 refraction offset, using the known, textured backing (markers, unevenness, Idea 1 plate) as the probe. Speculative and narrow (one benchmark shot), but the only route to correct glass.

### Cited Findings
- TOM-Net casts transparent-object matting as estimating an object mask, an attenuation mask and a refractive flow field, building on Zongker's environment matting C = F + (1 - alpha)B + Phi; it assumes colourless transparent objects and lists coloured, translucent and reflect-plus-refract cases as unsolved. [Chen et al. CVPR 2018](https://arxiv.org/html/1803.04636v3); [CVF version](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0844.pdf)
- Magenta Green Screen demonstrated glass bottles keyed with good transparency by spectral multiplexing on an LED stage. [Magenta Green Screen 2023](https://arxiv.org/html/2306.13702)

### Inferences
- Information limit. Against a green backing the red and blue channels of the backing are small, so transmission in R and B is observed with noise roughly sigma/B_r and sigma/B_b, several times worse than in G. Per-channel transmission therefore needs a learned prior plus Idea 4's paired green/blue renders for supervision; refraction is only observable where the backing has texture (markers, seams, the plate's unevenness).
- Plug-in: two extra output channels (alpha_R, alpha_B offsets from alpha_G) and a /4 flow head, all zero-init; compositing tools take RGB alpha natively in many pipelines.
- Cost: small. Risk: unclear benchmark definition of glass GT (scalar or per-channel), so gains may be invisible to the metric.
- Cheapest experiment: on the glass shot, fit the best scalar alpha and the best per-channel transmission to the GT composite and compare residuals; if the per-channel fit is not much better, the glass error is not about colour and this idea is dead for the benchmark.

### Gaps
- How the Blender generator defines alpha for glass (holdout, transmission pass, or luminance) is not documented in what I read; this determines whether the idea can score.

## Idea 9. Uncertainty-routed sparse /1 refiner with a calibrated confidence output

### Takeaway
Predict per-pixel aleatoric alpha uncertainty, ship it as a confidence pass for artists, use it as the routing signal for a small full-resolution refiner over the top 5 to 10% of tiles, and as the weight in Ideas 2, 5 and 7. The routing pattern is established (BGMv2); the calibrated confidence product and its reuse across the pipeline are new.

### Cited Findings
- BGMv2's base network predicts an error map and the refiner processes only selected patches at full resolution; "most improvement can be achieved by refining over only 5% to 10% of the image resolution". [Lin et al. CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)

### Inferences
- Mechanism: Laplace negative-log-likelihood head for alpha (predict log b), calibrated on the exact-GT benchmark (expected calibration error on |alpha error| versus b). Route 32x32 tiles by mean b; the refiner sees raw /1 pixels, B_t, F_loc, alpha_phys and upsampled stage-2 tokens. Generalises the existing /4 `Stage2Refine4Head` to sparse /1.
- Targets: 4K hair, spider web, fur, glass edges; artist workflow everywhere.
- Cost: refiner over about 0.4 to 0.8 megapixels at 4K; small.
- Risks: uncertainty heads trained on synthetic data are often overconfident on real footage; calibrate on the real-stage shots too.
- Cheapest experiment: add the NLL head only (no refiner), fine-tune briefly, and check on the benchmark whether the top 10% most uncertain pixels contain most of the alpha error. If they contain under half, routing will not pay.

### Gaps
- None beyond calibration transfer from synthetic to real.

## Idea 10. Geometry-predicted spill in a two-dimensional physical subspace

### Takeaway
Spill is screen light reflected by the subject, so for diffuse surfaces it lies in span{B, F_clean (elementwise) B} and its magnitude follows the subject's geometry relative to the screen (normals and distance, available from MoGe-2). Predict two spill scalars instead of free RGB, supervised exactly by Idea 4. This separates a green shirt (albedo) from spill (geometry), the core green-wardrobe confusion.

### Cited Findings
- Hachmann and Rosenhahn treat colour spill compensation as a missing link between alpha matting research and production chroma keying. [Hachmann & Rosenhahn 2023](https://arxiv.org/html/2306.02954)
- Smith and Blinn list blue spill, flare and backing shadows among the practical problems of constant-colour matting. [Smith & Blinn 1996](https://alvyray.com/Papers/CG/blusig96.pdf)

### Inferences
- Mechanism: F_spilled = F_clean + a*B_hat_dir + b*(F_clean * B_hat_dir), with a, b >= 0 predicted at /16 from features including MoGe-2 normals and depth, and a smoothness prior. Replace `despill_backing`'s single k*B_hat with this model.
- Targets: green wardrobe, real stage spill, blue screen.
- Risk: glossy (specular) spill and subsurface scattering fall outside the 2-D subspace; check the residual.
- Cheapest experiment: from Blender passes, least-squares fit spill = a*B + b*(F_clean * B) per pixel and measure the unexplained energy fraction across materials; if it is under about 5% for most pixels the subspace holds. Then count how often the current k*B_hat despill desaturates genuinely green wardrobe on the benchmark.

### Gaps
- No source found for typical spill magnitudes on real stages.

## Idea 11. Geometric screen membership for subject selection

### Takeaway
A tracking marker is coplanar with and static relative to the screen. Fit the screen surface from MoGe-2 points on confident backing pixels and feed signed distance to the screen surface as a token feature; anything on the surface is background whatever its colour. Cheap fix for the marker failure that works on frame 1, before Idea 1's plate has seen anything.

### Cited Findings
- MatAnyone argues auxiliary-free matting is "under-defined" without a clear target and uses a first-frame target mask with memory propagation. [Yang et al. CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.pdf)
- SDMatte adds point, box and mask prompts to a diffusion-based matting model. [Huang et al. ICCV 2025](https://arxiv.org/abs/2508.00443)

### Inferences
- Mechanism: RANSAC one to three planes on MoGe-2 point maps over pixels with alpha < 0.02, signed distance map (normalised by depth) as an extra /16 channel; optionally an artist prompt (box or click) as in MatAnyone/SDMatte for the residual ambiguity (C-stands, tape on the floor).
- Risk: monocular depth on a featureless green surface may be poor; markers themselves help.
- Cheapest experiment: one hour: fit the plane on the marker shot and check the marker's distance to it versus the subject's.

### Gaps
- MoGe-2 accuracy on featureless chroma screens is unmeasured.

## Idea 12. Physics-filtered distillation from a video-diffusion matting teacher

### Takeaway
Use a one-step fine-tuned video diffusion model as an offline teacher on unlabelled real stage footage, but keep pseudo-labels only where they agree with the physics (projection agreement where SNR is high, temporal consistency, F not screen-coloured beyond the spill budget). Plausible for semantics, smoke and subject selection; weak for 4K hair because current teachers run near 1K. Expensive and speculative.

### Cited Findings
- Generative Video Matting fine-tunes Stable Video Diffusion with flow matching for few-step inference, generalises from synthetic data, and claims strong real-world generalisation and temporal consistency; its demo runs with `--max_resolution 960` and `--denoise_steps 1`. [Ge et al. SIGGRAPH 2025](https://arxiv.org/html/2508.07905); [GVM repo](https://github.com/aim-uofa/GVM)
- End-to-end single-step fine-tuning of image-conditional diffusion models (and even plain Stable Diffusion) into deterministic dense predictors matches or beats multi-step pipelines while being over 200x faster. [Garcia et al. WACV 2025](https://arxiv.org/abs/2409.11355)
- Background Matting's teacher-student scheme was needed to avoid collapse when training on real unlabelled data. [Sengupta et al. 2020](https://doi.org/10.1109/cvpr42600.2020.00236)

### Inferences
- Novel part is the physics filter, which converts the 1-D ambiguity of Q0 into an acceptance test: a pseudo-label alpha must, with the plate-recovered F, give an F that is smooth, not screen-coloured, and consistent with alpha_phys where ||F - B||/sigma is large.
- Cost: large offline teacher compute; zero inference cost for CorridorKey.
- Risk: generative hallucination of hair detail; resolution cap.
- Cheapest experiment: run GVM on the real-stage and smoke shots at 960 px, apply the filter, and measure filtered pseudo-label error against benchmark GT at that resolution. If filtered labels are not better than CorridorKey's own outputs there, drop.

### Gaps
- No benchmark numbers for GVM on chroma-screen footage.

## Idea 13. Failure-driven render curriculum (automatic domain randomisation over the generator)

### Takeaway
Parameterise the Blender generator along the benchmark's failure axes (screen unevenness, marker density, smoke density, glass IOR and tint, thread width, CRF, ISO, motion blur, green-wardrobe similarity) and widen each range automatically when the model passes a threshold. Not novel as a method; novel as applied to keyer data, and cheap.

### Cited Findings
- Automatic Domain Randomization starts from a single environment and widens each parameter's range when performance passes a threshold (and shrinks it below a lower threshold), removing manual tuning and keeping the task always challenging; it was used for both policy and vision model training. [OpenAI 2019](https://arxiv.org/html/1910.07113); [OpenAI blog](https://openai.com/index/solving-rubiks-cube/)

### Inferences
- Combine with Idea 4: recolouring parameters are free axes (no re-render).
- Risk: curriculum chases synthetic difficulty that does not match real failures; anchor thresholds to benchmark-like synthetic validation per axis.
- Cheapest experiment: per-axis loss curves on a held-out synthetic set to confirm the current model's error rises monotonically along each axis before automating.

### Gaps
- None material.

## Ranking and the definitive design

### Takeaway
Include six: (1) backing memory, (2) physics projection head with SNR fusion and the confidence output, (3) light-group backing algebra with paired consistency and exact spill supervision, (4) degradation side-channel (screen-calibrated noise map, QP, native chroma path), (5) Fisher-weighted trajectory fusion in stage 2, and (6) TTT-Key as an optional offline quality mode. They form one system: the backing memory supplies B_t and the noise map, which the projection head, the temporal fusion and TTT all consume, while the light-group data trains the model to use them.

### Cited Findings
- Supporting evidence for the top picks is cited in each idea above: clean-plate value ([Chuang 2002](https://grail.cs.washington.edu/projects/digital-matting/papers/sig2002.pdf)), projection fusion ([FBA](https://ar5iv.labs.arxiv.org/html/2003.07711)), light-group linearity ([Blender manual](https://docs.blender.org/manual/en/latest/render/layers/passes.html)), codec side info ([CIAF](https://arxiv.org/html/2210.08229), [CDA-VSR](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Compressed-Domain-Aware_Online_Video_Super-Resolution_CVPR_2026_paper.html)), noise maps ([FFDNet](https://arxiv.org/abs/1710.04026)), trajectory attention ([Motionformer](https://arxiv.org/html/2106.05392v2)), video TTT ([Wang et al.](https://video-ttt.github.io/)), sparse refinement ([BGMv2](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)).

### Inferences
Ranking (expected benchmark gain x confidence / cost; my judgement):

| Rank | Idea | Main shots | Confidence | Train cost | Inference cost | In design |
|---|---|---|---|---|---|---|
| 1 | 1 Backing memory | real stage, markers, smoke, low light, codecs, subject selection | High | Low | Low | Yes |
| 2 | 2 Projection head + SNR fusion (+9 confidence) | 4K hair, fur, spider web, motion blur, focus pull, smoke | Medium-high | Low | Negligible | Yes |
| 3 | 4 Light-group backing algebra + paired loss | green wardrobe, blue screen, spill, real stage | High | Low (render passes) | None | Yes |
| 4 | 6 Degradation side-channel | H.264, crushed codec, low light | Medium | Low | Low | Yes |
| 5 | 7 Fisher-weighted trajectory fusion | shimmer on hair/fur/web, low light | Medium | Medium | Medium | Yes |
| 6 | 5 TTT-Key | crushed codec, low light, real stage | Medium | None | High (2-4x, optional) | Yes (optional mode) |
| 7 | 10 Geometry-predicted spill subspace | green wardrobe, spill | Medium | Low | Low | Fold into 3 if the subspace test passes |
| 8 | 9 Sparse /1 refiner | 4K hair, web | Medium | Medium | Low | Confidence head yes; refiner only if routing test passes |
| 9 | 11 Screen membership | marker shot | Medium (narrow) | Low | Low | Cheap add-on to 1 |
| 10 | 13 Render curriculum | all synthetic axes | Medium | Low | None | Training hygiene |
| 11 | 3 Motion triangulation | markers, camera moves | Low-medium | Medium | Medium | No (partly learnable inside 5 once B_t is an input) |
| 12 | 8 RGB transmission/refraction | glass | Low | Low | Low | No until GT definition checked |
| 13 | 12 Diffusion teacher distillation | semantics, smoke | Low | High | None | No |
| rejected | Multi-frame SR of the matte | none at native res | n/a | n/a | n/a | No (redundant with 7) |

- Build order follows the cheapest experiments, most of which need no training: (a) oracle clean plate inside `plate_foreground` (Idea 1 ceiling); (b) oracle and model-based alpha_phys on hair, fur and web (Idea 2); (c) light-group linearity check (Idea 4); (d) luma-guided chroma upsampling on the H.264 shot (Idea 6); (e) post-hoc flow-guided temporal median (Idea 7); (f) spill subspace residual (Idea 10); (g) plane distance of the marker (Idea 11); (h) one-evening TTT on the two codec/low-light shots (Idea 5). Each either promotes or kills its idea before any architecture work.
- Interaction to protect: Ideas 2 and 5 share the smooth-F assumption; if (b) fails, TTT-Key loses its main loss term and should be demoted.
- Most speculative claims in this document: the size of gains from Idea 7 on semi-transparent strands (flow reliability), any glass improvement (metric definition), and the diffusion teacher's value at 4K.

### Gaps
- No idea here has been tested on the benchmark yet; all gain estimates are qualitative. Benchmark-measured screen unevenness, noise sigma per shot and the glass GT definition are the three numbers that would most change the ranking.
- Novelty was checked with targeted searches only (learned chroma keyers, background matting, video TTT, codec-aware restoration, trajectory attention, triangulation, transparent matting, diffusion matting). A full patent and commercial-tool search (Foundry, Adobe, Blackmagic) was not done.
