# TCAD Dataset for Surrogate-Assisted Design of p-GaN-Gate HEMTs

This release contains the processed data needed to inspect and reproduce the principal data-driven results reported in the manuscript, "Surrogate-Assisted Design of p-GaN-Gate HEMTs: An Integrated Framework for Prediction, Screening, and Interpretability."

## Dataset organization

- `01_design_samples/`
  - `samples_augmented_1600.csv`: 1600 Latin-hypercube-sampled device designs. The six design variables are `t_AlGaN_nm`, `x_AlGaN`, `N_AlGaN_cm3`, `t_pGaN_nm`, `N_pGaN_cm3`, and `Lgd_um`.
  - `final_sample_status.csv`: TCAD convergence and data-availability status for every sampled design.
- `02_tcad_outputs/`
  - `foms.csv`: Device-level TCAD figures of merit. `vth_v` is in V, `idsat_a_per_mm` is in A/mm, `ron_ohm_mm` is in ohm mm, and `gm_max_s_per_mm` is in S/mm.
- `03_screening_evaluation/`
  - `candidate_table_117.csv`: The 117 independent-test candidates satisfying the TCAD-derived `1 <= V_th <= 3 V` constraint, with predicted and TCAD-verified `I_D,sat` values and ranks.
- `04_reproducibility_metadata/`
  - `vth_fixed_training_membership.csv`: The 1002 device IDs in the fixed final training set for the `V_th` task.
  - `vth_catboost_independent_test_predictions.csv`: Independent-test predictions for the final CatBoost `V_th` surrogate.

## Reproducibility notes

- All IDs are retained as six-character strings through `sample_id`.
- The device-level data split used random seed 42. The primary `I_D,sat` task used 1186 training and 297 test devices. The stable-threshold `V_th` task used 1002 training and 251 test samples; its training set was further divided into 851 internal-training and 151 validation samples.
- The `I_D,sat` curve-point dataset uses `V_GS = 5 V`; the value at `V_DS = 15 V` is reported as `I_D,sat`.
- The three `V_th` regions are low (`V_th < 0.7 V`), middle (`0.7 <= V_th <= 3.0 V`), and high (`V_th > 3.0 V`).
- The released files are processed, device-level tables. They are intended for result verification and do not expose the original simulator decks or the complete internal simulation workflow.
- The public package supports the main-text screening and $V_{\mathrm{th}}$ diagnostic results. The complete output-characteristic curve-point archive, raw TCAD decks, and internal development data are not included in this release.
- Compact code for fixed-configuration CatBoost training and the main-text evaluations is supplied separately.

## Public-release scope

This package follows a reproducibility-oriented release model. It provides the processed tables and evaluation inputs needed to verify the reported screening, ranking, and threshold-diagnostic results, while retaining simulator-specific decks, full automation, and internal development scripts. These retained materials can be shared with the corresponding author for legitimate reproduction inquiries where appropriate.

## Citation

Please cite the associated manuscript when using this dataset.
