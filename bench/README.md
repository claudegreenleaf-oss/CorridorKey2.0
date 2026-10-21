# CorridorKey hard benchmark

Fifteen green and blue screen shots with **exact** ground truth. Each one
composites a real foreground and its alpha over a simulated keying stage in
scene-linear light. The plate is then degraded the way cameras and delivery
codecs degrade footage. Because the plate is built from the ground truth, the
ground truth is exact on every frame, including hair, motion blur, defocus and
smoke.

Foregrounds come from held-out **test** splits only, so the train splits stay
free for fine-tuning:

- VideoMatte240K HD test: real 4K clips with real motion and hair.
- Transparent-460 test: glass, liquids and spider webs.
- AIM-500: animal fur. AIM ships photos, so the foreground is re-estimated
  with multilevel foreground estimation.

| case | what makes it hard |
|---|---|
| c01_clean_control | Nothing: lossless 16-bit, even screen, no spill or noise. This is the ceiling. |
| c02_real_stage | Uneven screen, wrinkles, tracking markers, vignette and edge spill, in 10-bit ProRes 422 HQ. |
| c03_camera_h264 | The real stage through a camera-grade 8-bit 4:2:0 H.264, with more sensor noise. |
| c04_crushed_web | Heavily compressed H.264 (CRF 31) with in-camera sharpening halos. |
| c05_motion_blur | A fast whip move with a 180-degree shutter. The blur is integrated exactly over 16 subframes. |
| c06_focus_pull | Defocus that ramps from sharp to 14 px bokeh across the shot. |
| c07_smoke_veil | Drifting semi-transparent smoke over the subject. |
| c08_blue_screen | A blue backing with blue spill. Green-only despill heuristics do nothing here. |
| c09_low_light | The screen 2.5 stops under, with heavy noise, in 8-bit H.264. |
| c10_green_wardrobe | The lower body dressed in a green close to the screen color. |
| c11_glass_transparency | A glass splash and water. |
| c12_spiderweb_4k | UHD spider web with sub-pixel threads. |
| c13_hdr_linear | A scene-linear EXR plate with highlights above 1.0, which tests the HDR path. |
| c14_animal_fur | Leopard fur and whiskers. |
| c15_hair_4k | Native UHD hair from a 4K source on the real stage. |

Sensor noise counts as part of the subject, so it stays in the ground-truth
foreground. A compositor wants the plate's own grain on the subject. Screen
spill, codec damage and sharpening are not part of the ground truth: the
ground-truth foreground is the subject without the screen's light on it, so a
keyer has to remove spill to score well.

## Metrics

| metric | meaning |
|---|---|
| `comp_mae` | **Headline.** Composite error over four new backgrounds (grey, black, bright, colour texture), in 8-bit sRGB units, over the whole frame. This is what a compositor actually sees. |
| `comp_edge` | The same error, restricted to the edge band. |
| `comp_psnr` | Composite PSNR in dB. |
| `sad`, `mse`, `grad` | Standard alpha-matting metrics. |
| `edge_mae` | Mean alpha error in the band where 0 < GT < 1, dilated by 7 px. |
| `holes` | Missing alpha inside solid regions, x1000. |
| `leak` | Garbage alpha in empty regions, x1000. |
| `dtssd` | Temporal flicker error, x100. |
| `fg_mae` | Foreground color error in 8-bit sRGB, alpha-weighted. |

## Running

```
python bench/make_bench.py --data <data dir> --out <bench dir>       # ~20 GB
python bench/run_bench.py --bench <bench dir> --out results/baseline
python bench/run_bench.py --bench <bench dir> --out results/fixed -- --decode ffmpeg --precision mixed --plate_fg
python bench/compare.py results baseline fixed
```

Anything after `--` goes straight to `infer.py`.
