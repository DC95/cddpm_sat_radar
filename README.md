# cddpm_sat_radar
A conditional DDPM, which means:  Input = satellite IR cloud maps  Target = radar reflectivity (e.g.,GPM-DPR).  The model learns to map noisy versions of reflectivity conditioned on IR inputs, and denoise them step-by-step
