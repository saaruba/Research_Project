# Methodology — Rebuilt Draft (for Chapter 3)

**Rebuilt from the actual, current codebase, experiment logs and project-record documents as of 25 August 2026 — not written generically. Every number, fault, and design decision below traces to a specific script, node, log, or results file in the repository (`PROJECT_RECORD_POST_REPORT1.md`, `Section_3_6_Evaluation_Metrics.md`, `V2_RETRAINING_STUDY.md`, `APPROACH_ACCURACY_RESULTS.md`, `RESULTS_TABLES.md`, `dataset/processed/ospace_labelling/ospace_validation_result.json`, and the scripts cited inline). This draft supersedes the version written on 20 August 2026: the project has since closed the loop into a full 60-trial live experiment, completed the Objective 3 O-space hand-labelling validation, run and rejected a full model-improvement ablation ("v2"), and completed a systematic 360-model hyperparameter search — none of which existed when the previous draft was written.**

**This rebuild explicitly answers two supervisor comments from the annotated proposal/report document:**
- **[AP4]**, on the original Person Perception section: *"You can briefly present some alternatives to YOLO here and mention why YOLO was selected."* — addressed in full in Section 3.2 below.
- **[AP5]**, on the original Evaluation Metrics section: *"Describe in more detail the evaluation metrics you are using."* — addressed in full in Section 3.7 below, built from the dedicated `Section_3_6_Evaluation_Metrics.md` document written specifically for this comment.

No word-count target has been applied to this draft, per instruction — trim for the final submission only if the overall dissertation word budget requires it. Every image or diagram worth adding is flagged in-line with **[FIGURE]** markers as it comes up, and the complete consolidated list appears in Section 3.10 at the end, so nothing needs to be hunted for separately.

---

## 3. Methodology

### 3.0 Research approach and design

This project adopts a **quantitative, comparative experimental methodology**. Two learned approach policies and one hand-engineered geometric baseline are evaluated against each other and against naive statistical floors, under the same simulated world, the same start conditions, the same patrol route and the same evaluation metrics, so that policy identity is the only variable that changes between trials (Objective 4). Two reasons make this the right choice rather than the default choice. First, the research question — *does a policy that has learned a stopping pose from non-expert demonstration data behave more or less socially appropriately than a fixed geometric rule?* — is a comparison-of-behaviour question with a numeric, falsifiable answer (position/orientation error thresholds, O-space intrusion rate, task success rate), which quantitative methods are built to answer, whereas a qualitative approach (interviewing bystanders about how an approach *felt*) would require running the physical robot in the presence of real human participants, which sits outside both the ethical approval this project operates under and the time available (Section 3.8). Second, the PLUS-HRI dataset is a fixed, already-collected corpus of 24 non-expert teleoperation sessions rather than something this project could redesign as a live study — this fixes the appropriate design to *retrospective analysis and offline learning from existing behavioural traces*, followed by a *live simulation comparison* against ground truth, rather than new data collection.

The corresponding limitation is stated plainly rather than left implicit: every metric this project reports — approach-position/orientation error, O-space intrusion, minimum distance to person, cut-through rate, task success — is a **geometric proxy** for social acceptability, not a direct measurement of how a bystander would feel about being approached. This is an accepted simplification throughout the reviewed literature (Kruse *et al.*, 2013; Ríos-Martínez, Spalanzani and Laugier, 2015), but it is a real and unclosed gap between what is measured and what ultimately matters, held throughout this project as an explicit, stated limitation rather than something resolved.

The system pipeline the rest of this chapter describes, in order, is:

```
recorded PLUS-HRI demonstrations
        |
        v
extraction & labelling (offline)
        |
        v
offline rule / Behavioural-Cloning training & evaluation  --------→  [rejected "v2" ablation, §3.5]
        |
        v
Gazebo simulation (TIAGo + hand-built world with ground truth)
        |
        v
live perception (YOLOv8n | LocateAnything-3B)
        |
        v
group clustering & O-space estimate
        |
        v
APPROACH POLICY  (rule | Behavioural-Cloning Random Forest | Behavioural-Cloning MLP)
        |
        v
Nav2 (global planner + DWB local controller)
        |
        v
metrics recorder — offline error, 7 simulation metrics, F-formation slot accuracy
```

**[FIGURE 3.0-A]** A redrawn version of this pipeline as a proper labelled block diagram (boxes and arrows, not ASCII) belongs here — see the consolidated figure list in Section 3.10.

Both the offline half (Sections 3.1–3.5) and the live-simulation half (Sections 3.6–3.7) of this pipeline are described below in full, since Objective 4's central comparison is only meaningful once both are in place and their results are read against each other (which is done explicitly in Section 3.9).

Since the previous draft of this chapter, the project's own account of what it is claiming has sharpened, and that sharpening is itself a methodological decision worth stating up front rather than discovering three chapters later. The learned models — evidenced directly in Section 3.4 and reinforced by the entire rejected "v2" study in Section 3.5 — were shown to have learned **the final metres of an approach already under way**, not the whole act of finding and walking toward a group. The dissertation's central claim is therefore deliberately narrower and sharper than the original framing in the proposal:

> *Can the stopping pose for a group approach be learned from non-expert demonstrations?*

rather than *can a robot learn to find and approach groups end-to-end*, which would require demonstrations of the whole behaviour rather than its terminal phase. This reframing is not a retreat; it is what the evidence in Sections 3.4 and 3.5 actually supports, and claiming anything broader would not be defensible against the project's own data.

**A note on scope, to avoid duplicating Chapter 4.** This chapter's job is to justify *what* was done and *why* — the research design, the choice between candidate methods, the metrics chosen to judge success, and the empirical findings that forced a design to change. Line-by-line code structure, the ROS 2 package/node breakdown as software engineering, exact class and function names, launch-file configuration, and repository statistics (lines of code, package counts) are deliberately left to Chapter 4 (Implementation), which documents *how* the design decided here was actually built. Where a script or file is named below, it is named to evidence a methodological claim (this number came from this measurement, this fault was found and fixed this way), not as a substitute for Chapter 4's own account of the system's construction — some unavoidable overlap exists at the boundary (for instance, the rule-based policy's iterative refinement in Section 3.6.4 and the fault log in Section 3.6.5 are as much engineering history as method), and that overlap is kept here rather than in Chapter 4 specifically because each entry represents a *measured finding that changed the methodology*, not merely a code change.

### 3.1 Dataset

The PLUS-HRI dataset consists of 24 recorded non-expert teleoperation sessions. Direct inspection — rather than assuming a uniform format across all 24 — established that the dataset spans **two distinct recording generations**. Sessions 1 and 3 alone carry full per-frame facial-landmark annotations (`facial_landmarks_uniface.csv`) recorded under an earlier pipeline; the remaining 22 sessions were recorded later in a segmented-clip format carrying only `cmd_vel.csv`, the operator's own raw eye-gaze coordinates (not a human subject's), the robot's head pan/tilt joint state, and joystick input, with **no person-position ground truth of any kind**. `gaze_uniface.csv` was confirmed, by direct inspection, to be entirely empty — 100% null across every column, in every session — so no per-person facing-direction ground truth exists anywhere in this corpus, a fact that shapes the O-space estimation methodology in Section 3.3. This generational asymmetry is the single fact that shapes the extraction methodology below: sessions 1 and 3 are the project's only source of ground truth for validating a person detector (Section 3.2) and its own O-space estimate (Section 3.3), while all 24 sessions contribute rows to the Behavioural Cloning training data once a detector's own live output — rather than the recording itself — supplies the missing human-position signal for the other 22.

Extraction was implemented in `extract_training_table.py`, reading each session's ROS1 bag files via the `rosbags` library's ROS1 reader and typestore, and merging pose (`/robot_pose`, `/mobile_base_controller/odom`, `/dlo_node/odom`), LiDAR (`/scan`), and person-detection topics onto a common timestamp index using `pandas.merge_asof`. Across all 24 sessions this produced **229,678 synchronised rows at approximately 40 Hz** — the raw table from which every downstream dataset (approach-pose labels, offline evaluation splits) is derived.

Three dataset-specific defects were identified and corrected during this process, and are reported methodologically rather than as debugging footnotes, because each would silently corrupt results if left unstated:

1. `cv2.CAP_PROP_POS_FRAMES`-based frame seeking on these session videos was found to decode corrupt, solid-green frames at certain offsets. `extract_person_detections.py` was rewritten to read frames strictly sequentially instead of seeking, which eliminated the corruption.
2. An initial 0.1 s merge tolerance between `cmd_vel` and detection timestamps was found to drop 80–88% of rows to missing data, because the 22 segmented sessions' detections are sampled at roughly 1 Hz rather than per-frame. The tolerance was widened to 0.6 s — covering half the sampling gap on either side, without changing which detection `merge_asof` selects as nearest — to recover genuine coverage.
3. A separate, unrelated pitfall: `np.isclose`'s default relative tolerance (`rtol=1e-5`) evaluates to roughly 17,000 seconds on a ~1.76×10⁹ Unix timestamp, causing every detection in a session to spuriously match every frame. This was corrected by passing `rtol=0` explicitly wherever timestamps are compared for equality.

Session-level train/validation/test splitting (`split_dataset.py`) allocates **whole sessions** to each split (70% train / 15% validation / remainder test) using a fixed random seed (42), so that no single session's rows appear in more than one split. This is a deliberate anti-leakage choice: adjacent rows within one session are highly autocorrelated (near-identical robot pose and LiDAR readings a few tenths of a second apart) and would otherwise let a model "memorise" a session rather than generalise across them. Sessions 1 and 3 are deliberately forced into the training split rather than randomly assigned, because they are the only two sessions carrying real human-position ground truth and are too valuable, and too few, to risk holding one entirely out of training. This is stated as a limitation here rather than discovered independently: **the person-perception validation (Section 3.2) and the O-space validation (Section 3.3) both therefore validate against sessions the Behavioural Cloning model was also trained on**, which is acceptable for validating the *perception and clustering pipeline* (a fixed, non-learned geometric procedure) but would not be acceptable for validating the *learned model itself* — which is why the Behavioural Cloning evaluation in Section 3.4 uses entirely separate, fully held-out test sessions (5, 9 and 59) that were never used for any training or tuning decision at any point in the project, including the later v2 ablation (Section 3.5).

### 3.2 Person perception — detector choice, alternatives, and validation *(responds to supervisor comment [AP4])*

#### 3.2.1 The landscape of person-detection approaches considered

The proposal originally named NVIDIA's LocateAnything-3B (Wang *et al.*, 2026), a vision-language grounding model, as the intended perception component. Before defending the eventual substitution to YOLOv8n, it is worth setting out the actual space of alternatives that exist for this sub-problem, since a single-sentence justification ("it was faster") understates the range of options genuinely available and considered.

Object/person detectors fall into three broad architectural families relevant here:

**Two-stage detectors.** Faster R-CNN (Ren *et al.*, 2015) established the now-standard two-stage pipeline: a Region Proposal Network first proposes candidate bounding boxes, and a second network classifies and refines each proposal. This family typically achieves strong accuracy because classification and localisation are separated into dedicated stages, but at a direct computational cost — every proposed region requires its own forward pass through the classification head, which historically placed two-stage detectors well below real-time frame rates on commodity hardware without a high-end GPU. For a closed-loop robot control task, where a detection is needed at the perception loop's own operating rate (this project's perception loop runs at roughly 2 Hz) rather than as a single offline query, that architectural cost is directly relevant.

**Single-stage detectors.** YOLO ("You Only Look Once", Redmon *et al.*, 2016) collapses detection into a single forward pass over the full image, predicting bounding boxes and class probabilities simultaneously from one dense grid of predictions rather than a separate proposal stage. This trades a small amount of accuracy, particularly on small or heavily overlapping objects, for an order-of-magnitude increase in throughput. YOLOv8n — the specific variant used in this project — is the smallest ("nano") configuration in the Ultralytics YOLOv8 family (Jocher, Chaurasia and Qiu, 2023; architecturally documented independently by Yaseen, 2024, since Ultralytics did not publish a formal peer-reviewed paper alongside the release), chosen specifically for its combination of speed and a small memory/compute footprint suited to a robot's onboard or workstation-tethered compute budget.

**Transformer-based detectors.** DETR (Carion *et al.*, 2020) reformulates detection as a direct set-prediction problem using a transformer encoder-decoder, removing the hand-designed anchor boxes and non-maximum-suppression post-processing that both of the above families rely on. This is architecturally elegant and removes several hand-tuned hyperparameters, but transformer-based detectors are known for slow training convergence and, at inference time, carry the same fundamentally heavier per-frame compute cost as other transformer architectures — a cost of the same *character* (not necessarily the same magnitude) as the vision-language model actually evaluated below.

**Vision-language grounding models.** LocateAnything-3B (Wang *et al.*, 2026) sits in a fourth, more recent category: rather than a fixed closed vocabulary of object classes, it grounds an arbitrary natural-language description ("person") to a bounding box using a 3-billion-parameter vision-language backbone with parallel box decoding. This buys open-vocabulary flexibility — the model can in principle be asked to locate categories a YOLO model was never trained on — at the cost of an autoregressive, much heavier decoding process per image. This is the model this project's own measurements characterise directly, below.

**Why this project's problem constrains the choice more than the choice constrains the problem.** The task here is not "detect a person in an image" in isolation; it is "supply a real-time, continuously-updated person position to a closed-loop navigation policy that decides, moment to moment, where to stand." That framing rules out any detector whose inference latency is comparable to or greater than the control loop's own update interval, regardless of that detector's accuracy in isolation — a highly accurate detection that arrives once every eight seconds cannot drive a policy that needs to react to a person stepping into or out of a group. This is precisely the axis on which the two candidates actually available to this project — YOLOv8n and the originally-proposed LocateAnything-3B — were measured, and it is why the deciding factor below is throughput rather than raw detection accuracy, which the results show to already be adequate at either end.

#### 3.2.2 The documented substitution, and its offline justification

The substitution of YOLOv8n for the proposal's originally named LocateAnything-3B is a **documented substitution**, not an undisclosed deviation: the proposal's own risk table names "keep a simpler person-detection baseline" as an approved mitigation for exactly this contingency.

The substitution was validated, not merely asserted. `validate_detector_recall.py` measures recall directly against the facial-landmark ground truth in sessions 1 and 3 — defined precisely as: of every moment where the ground truth confirms at least one face was genuinely present, what fraction did YOLOv8n also report at least one detected person in — and found **99.7% recall against the proposal's own 80% target**, comfortably satisfying Objective 2 on its own terms.

To additionally test the substitution against the *named* model rather than only an arbitrary threshold, an early, direct head-to-head comparison (`run_locateanything_comparison.py`) was run on 30 ground-truth-positive frames from sessions 1 and 3: LocateAnything-3B reached 100.0% recall (30/30) at a measured **25.63 s/frame** on the development machine (NVIDIA RTX 3050 Ti Laptop GPU, 4.3 GB of VRAM actually used, with CUDA falling back to system memory because the model did not fit comfortably in the available VRAM — recorded in `locateanything_service.py`'s own measurement log), versus YOLOv8n's 96.7% (29/30) at approximately 0.005 s/frame (~200 FPS). The recall gap is a single frame and the two detectors' Wilson 95% confidence intervals overlap heavily (88.6–100% vs. 83.3–99.4% at n=30) — not a statistically meaningful accuracy difference at this sample size — while the roughly 5,000× speed difference on that hardware was unambiguous. This early result was the project's first, offline justification for the substitution: not that LocateAnything-3B was difficult to install, but that a direct, measured comparison found no meaningful accuracy advantage to offset an enormous speed cost for a task requiring continuous, real-time detection.

`locateanything_service.py` documents a further, purely practical reason the model runs as a separate process at all: LocateAnything-3B's dependencies require `numpy==1.25.0`, while the rest of this project's ROS 2 / scikit-learn stack requires `numpy` 2.x — the two cannot coexist in one Python environment. LocateAnything-3B is therefore deployed as an isolated HTTP microservice, called by `group_perception_node` when the LocateAnything condition is selected, rather than imported directly — an engineering detail worth reporting because it is itself a small piece of evidence about the model's practical deployability: even before its latency is considered, it does not integrate cleanly into a robotics software stack built on the current numerical Python ecosystem.

#### 3.2.3 The final, decisive measurement: a full-scale live benchmark

The early 30-frame comparison above was subsequently superseded by a far more rigorous measurement, run as part of the final 60-trial live experiment (Section 3.6) on the project's dedicated lab workstation (NVIDIA RTX 4070, 12 GB VRAM, CUDA 13.0, torch 2.5.1+cu121) — different hardware from the earlier development-machine test, and reported separately here because the two numbers describe different things (an early offline sanity check on a laptop GPU, versus a benchmark taken from 31 live inferences actually made during real experimental trials on the machine the reported results come from):

| | YOLOv8n | LocateAnything-3B |
|---|---|---|
| Mean inference | ~0.005 s/frame | **8.40 s/frame** |
| Median inference | — | 9.64 s |
| Range | — | 0.35 – 11.37 s |
| Effective rate | ~200 Hz | **0.12 Hz** |
| Relative cost | 1× | **~1,700× slower** |
| Inferences over the 30-trial batch | continuous, 2 Hz throttled | **31 (≈ 1 per trial)** |

At 8.40 s/frame the perception node necessarily falls back to a single-shot mode: it takes essentially one detection at the start of a trial and then relies on Nav2's own obstacle avoidance for the remainder, since a fresh group position is not available again for several seconds. This is not a hypothetical concern; it produced a directly measured behavioural consequence in the full 60-trial comparison (Section 3.6.6): all three approach policies became statistically indistinguishable from one another under LocateAnything-3B, and O-space intrusion rose to 30/30 (100%) against 20/30 (67%) under YOLOv8n (two-sided Fisher exact test, p = 0.001), because the robot's patrol route carries it past groups it can no longer perceive.

**This is now reported as the project's final, decisive justification for [AP4]**, in place of the earlier offline-only comparison: YOLOv8n was not selected merely because it was "simpler" or more convenient to install, and not only because an early 30-frame test found no meaningful accuracy advantage to the alternative. It was selected because it is the only one of the candidate detectors evaluated that operates at the frame rate a *closed-loop* social navigation policy actually requires, and the full-scale live experiment converts that architectural expectation into a measured engineering finding: **LocateAnything-3B, as evaluated in this project, is unsuitable for closed-loop social navigation**, not merely slower. Two-stage detectors (Ren *et al.*, 2015) and transformer-based detectors (Carion *et al.*, 2020) were not separately implemented and benchmarked in this project — a scope limitation stated plainly — but both share the vision-language model's structural disadvantage of heavier per-frame inference than a single-stage detector, and neither was named in the original proposal as the intended perception component, so neither displaces YOLOv8n as the candidate actually requiring a documented, measured substitution decision.

The detector remains selectable at runtime in the deployed system (`group_perception_node` accepts a `--detector` argument): YOLOv8n is the default for continuous, closed-loop perception, and LocateAnything-3B remains available specifically so that the proposal's originally-named model could be run through the identical evaluation pipeline and measured on equal terms — which is precisely what produced the benchmark above, rather than a claim resting on the model never having been tried.

**[FIGURE 3.2-A]** A bar or box-plot figure showing the inference-time distribution for both detectors (YOLOv8n's ~0.005s versus LocateAnything-3B's 0.35–11.37s range, log-scaled y-axis) would make the ~1,700× gap visually immediate — see Section 3.10.

**[FIGURE 3.2-B]** A short annotated example frame showing YOLOv8n's live bounding-box overlay (as seen in RViz) is a natural inclusion here to make "person perception" concrete before the group-detection stage is described.

### 3.3 Group detection and O-space estimation

Group detection (`cluster_groups.py`) treats individually-detected people, per frame, as nodes in a similarity graph, linking two people into the same group when their pixel distance falls below a threshold expressed in units of their own average bounding-box width, rather than raw pixels — connected components of this graph become candidate groups. This normalisation is a deliberate, stated approximation for the absence of camera calibration or depth in the recorded offline video: bounding-box width shrinks with distance from the camera in roughly the same way true separation does, so measuring distance in "person-widths" partially cancels perspective distortion that raw pixel distance would not. In the live system (Section 3.6), the equivalent clustering step (`group_perception_node`) is performed in true metres using depth-back-projected 3D coordinates, with a fixed 1.5 m single-linkage threshold — a cleaner formulation available only once the pixel-only constraint of the recorded dataset no longer applies.

O-space estimation follows Kendon's (1990) F-formation framework — the shared, jointly-oriented space at a conversational group's centre — but could not use the framework's usual orientation-based construction, because `gaze_uniface.csv` is entirely empty across the whole dataset (Section 3.1): there is no per-person facing-direction ground truth anywhere in PLUS-HRI. The adopted fallback is the **mutual-facing assumption**: each group's O-space centre is approximated directly as the group centroid already produced by clustering, and (in the live system) the O-space radius as the mean distance from that centroid to its members. This is not an ad-hoc workaround; it is independently justified by Vascon *et al.*'s (2016) game-theoretic F-formation detection method, which demonstrates that group structure can be recovered from relative *position* alone, without requiring orientation estimates at all — a result this project leans on directly to defend the mutual-facing fallback as a legitimate simplification given the data available, not merely a compromise made out of necessity.

#### 3.3.1 Validating the O-space estimate: a completed hand-labelling study

Validating the O-space pipeline required a hand-labelled ground truth that does not exist natively in the dataset, so one was constructed and, since the previous draft of this chapter, **fully completed**. The workflow proceeded in four stages, each implemented as its own script:

1. **`extract_clean_ospace_frames.py`** selected 30 frames spanning 18 sessions, each containing a genuine group of two or more people, avoiding motion-blurred or corrupted frames.
2. **`label_ospace_frames.py`**, paired with a purpose-built browser tool (`label_ospace.html`), presented each frame to a human labeller, who clicked the point they judged to be the group's true O-space centre.
3. **`merge_ospace_clicks.py`** merged the HTML tool's click coordinates back into `labels.csv`, matching records by `frame_file` rather than row order, so that a labelling session could be paused and resumed without misaligning rows.
4. **`validate_ospace_estimate.py`** compared each manual label against the pipeline's own automatic centroid estimate for the same frame and computed the summary statistics reported below.

**A labelling-instruction correction, reported as part of the methodology.** An earlier labelling attempt asked labellers to click the group's centre at floor level (the natural interpretation of "where the group is standing"), and produced a **161-pixel systematic bias** with a standard deviation of 48 px between the manual labels and the pipeline's own bounding-box-centroid estimate. Investigating this discrepancy revealed a definitional mismatch rather than a labelling error: the pipeline's O-space estimate is built from detection bounding-box centres, which sit at roughly chest/body height because that is where a person's bounding box is centred, while a floor-level click is, geometrically, a different point entirely — offset by the perspective foreshortening between a standing person's feet and their torso as seen from the camera's elevated, angled viewpoint. The labelling instruction was corrected to ask labellers to click the group's centre at **body height**, consistent with what the pipeline actually estimates, and the study was re-run under the corrected instruction. `labels_floor_definition.csv` is retained in the dataset alongside the corrected `labels.csv` specifically so this correction remains auditable rather than silently overwritten.

**The final, completed result.** Under the corrected labelling instruction, across all 30 frames:

| Quantity | Value |
|---|---|
| Frames prepared / scored | 30 / 30 (none excluded for missing a group) |
| Tolerance | 0.5 × mean detected person bounding-box width |
| Mean error | 92.36 px (0.760 person-widths) |
| Median error | 75.11 px |
| Frames within tolerance | **11 / 30 (36.7%)** |
| Target (Objective 3, as re-specified) | ≥ 70% |
| **Verdict** | **FAIL** |

The proposal's original criterion — an estimate within 0.3 m of the manual label for at least 70% of frames — could not be evaluated honestly, since 0.3 m is a metric tolerance and the underlying video carries no depth or calibration from which real metres could be recovered; reporting a metre figure here would be fabricated. The criterion was therefore re-specified before use as: within 0.5× the mean detected person bounding-box width of the manual label, for at least 70% of labelled frames — chosen because an adult's shoulder width is itself roughly 0.45–0.50 m, so 0.5× bounding-box width corresponds to approximately 0.22–0.25 m of real-world tolerance, preserving the original criterion's order of magnitude while expressing it in a unit this dataset can actually measure, and one that self-corrects for perspective (a person further from the camera receives a proportionally tighter pixel tolerance, which is the physically correct behaviour). This re-specification is reported here as a deliberate, disclosed methodological adaptation to a genuine data constraint, made and documented before the validation itself was run — not a criterion chosen after seeing the result.

**This result is reported as an honest failure against the project's own re-specified criterion**, and is treated as a substantive finding rather than something to be minimised. `validate_ospace_estimate.py` itself is written with an explicit instruction to this effect: *"A fail here is still a reportable result — it quantifies how well a centroid-based O-space estimate matches human judgement, which is exactly what Objective 3 asked you to measure. Report it honestly."* The 36.7% pass rate should be read alongside two mitigating facts established elsewhere in this chapter: first, the mutual-facing, centroid-based estimate is independently justified in the literature (Vascon *et al.*, 2016) as a legitimate simplification given the absence of orientation data, not an arbitrary guess; second, the live simulation results (Section 3.6) that actually depend on O-space accuracy — O-space intrusion rate, task success — are computed against the world's true ground-truth group positions, not against this offline pixel-based estimate, so this specific failure does not propagate into and invalidate the live experimental results. It is, however, a direct and honest answer to Objective 3 on its own terms, and is reported as such in Chapter 5 rather than omitted.

**[FIGURE 3.3-A]** A montage of a small number of the 30 labelled frames — showing the human-clicked point, the pipeline's automatic centroid, and the resulting pixel error — would let a reader see directly what a "pass" and a "fail" frame each look like. Two or three examples, one clearly within tolerance and one clearly outside it, would be the most informative choice.

**[FIGURE 3.3-B]** A simple bar chart or histogram of the 30 per-frame errors (in person-widths), with the 0.5-person-width tolerance line marked, visualises the 11/30 pass rate more legibly than the summary table alone.

### 3.4 Behavioural Cloning model

The approach-pose model trained here sits within the broader robot learning-from-demonstration paradigm surveyed by Argall *et al.* (2009): rather than hand-engineering a stopping rule from first principles, the policy is fit directly to examples of a human operator's own choices, on the premise that non-expert demonstrations encode useful, if imperfect, judgement about where it is appropriate to stand. This framing matters for how the rest of this section, and the whole rejected v2 ablation in Section 3.5, should be read: every limitation found below is a limitation of *what the demonstrations themselves contain*, not of the learning-from-demonstration paradigm in the abstract.

**Label construction.** No stage of the raw teleoperation recording labels any moment as "this was a demonstrated group approach" — this had to be inferred. `build_approach_pose_dataset.py` identifies genuine approach events by finding transitions from sustained movement to a genuine stop, with a detected group present nearby at the stop; that stop's pose is then treated as the human operator's implicit judgement of "a good place to be near this group," and every row in the lead-up to that stop is labelled with the eventual stop pose as its prediction target. Raw `cmd_vel` in this dataset is extremely spiky — isolated single-sample non-zero values rather than sustained speed changes — so a naive sample-by-sample moving/stopped test found almost no usable segments; speed was instead averaged into 0.5-second bins before the moving/stopped test was applied, which recovered genuine, usable movement segments. This inference process identified **462 independent demonstrated approach events across all 24 sessions, expanding to 70,555 labelled rows** in `approach_pose_dataset.csv` — a dataset now referred to throughout this chapter as "v1", to distinguish it from the "v2" re-segmentation described fully in Section 3.5.

**Target and feature design.** The prediction target is the demonstrated stop pose expressed **relative to the robot's own position and heading at prediction time** (Δx, Δy in the robot's forward/left frame, plus Δyaw), rather than an absolute map coordinate — a direct consequence of an earlier finding: absolute robot (x, y) as a *feature* in the first baseline model measured *worse* than omitting it, because raw position does not generalise across sessions recorded in different rooms; the same reasoning was judged to apply, if anything more strongly, to using absolute position as a *target*. The final feature vector — seven features, in a fixed order the deployed policy node must reproduce exactly — is: `lidar_min_range`, `lidar_mean_range`, `linear_x_prev`, `angular_z_prev`, `num_people`, `group_bearing_rad`, and `group_scale_norm`.

**Model architectures.** Two model families were trained and compared, per the proposal's committed architecture decision: a multi-layer perceptron (two hidden layers, originally specified as 128/64 units with ReLU activations) as the primary architecture, and a Random Forest regressor (Breiman, 2001) as a lower-variance, documented alternative — a sensible hedge given how few independent demonstration events (462) were available relative to the 70,555 expanded rows, since tree ensembles are typically more robust than neural networks to a small number of truly independent examples. Both were tuned via grid search (`grid_search_approach_pose.py`) exactly as the proposal's Research Methods section committed to: every candidate configuration was scored **only on the validation split**, with the test split held out entirely until a single winning configuration per model family was selected and finally evaluated once.

Tuning materially changed both models' results — Random Forest improved from 0.401 m / 29.3° to 0.365 m / 25.8°, and MLP from 0.465 m / 42.2° to 0.395 m / 31.0° — and, notably, the winning MLP configuration (32/16 hidden units, L2 regularisation α = 0.1) is substantially smaller and more heavily regularised than the proposal's originally specified 128/64 architecture: **the proposed architecture was too large for the amount of independent demonstration data actually available**, not merely under-tuned. This observation is the first thread in a pattern that recurs, and is confirmed decisively, throughout Section 3.5.

**Evaluation.** `evaluate_approach_pose.py` compares four policies — a naive mean-prediction floor, the rule-based baseline applied offline, the tuned Random Forest, and the tuned MLP — on the fully held-out test sessions (5, 9, 59; 11,921 rows), reporting mean and median error for both position and orientation, plus the percentage of rows meeting each of Objective 4's thresholds (position < 0.4 m, orientation < 20°) independently and jointly:

| Policy | Mean pos. error (m) | Median pos. error (m) | Mean orient. error (°) | Median orient. error (°) | Within position | Within orientation | Within both |
|---|---|---|---|---|---|---|---|
| Naive (predict mean) | 0.410 | 0.365 | 28.97 | 12.53 | 68.4% | 63.2% | 47.2% |
| **Rule-based (geometric)** | **0.305** | **0.164** | 29.13 | 16.60 | **70.1%** | 56.8% | 43.1% |
| Random Forest (untuned) | 0.401 | 0.294 | 29.27 | 16.72 | 61.4% | 55.8% | 38.2% |
| MLP (untuned) | 0.466 | 0.353 | 42.22 | 28.78 | 55.3% | 37.5% | 21.8% |
| **Random Forest (tuned)** | 0.365 | 0.267 | **25.78** | **13.15** | 66.8% | 61.3% | **43.3%** |
| MLP (tuned) | 0.395 | 0.305 | 30.99 | 21.36 | 61.9% | 47.0% | 31.8% |

**No policy met the proposal's 20° orientation threshold**, and the naive mean predictor's 47.2% "within both" figure exceeds every learned model — the first, offline indication that the dataset itself, rather than model choice, is the binding constraint on accuracy. That suspicion is investigated directly and confirmed in Section 3.5.

A supplementary, separately-run **retraining sensitivity check** (a different random session split from the table above, so the two are not directly numerically comparable, only internally consistent) tested six configurations across both model families specifically to check whether the ranking above was an artefact of one particular tuning run:

| Configuration | Test position error (m) | Test orientation error (°) |
|---|---|---|
| **Naive (predict training mean)** | **0.349** | **25.1** |
| Random Forest, unregularised | 0.351 | 30.4 |
| Random Forest, leaf = 5, depth = 20 | 0.341 | 28.9 |
| Random Forest, leaf = 20, depth = 15 | **0.331** | 27.1 |
| MLP 128–64 | 0.507 | 38.0 |
| MLP 64–32, α = 1e-2 | 0.401 | 34.6 |

**A model that ignores every input and predicts the training-set average matches or beats every learned model tested, and beats all of them on orientation.** This is not a tuning problem confined to one grid search; six configurations across two model families, on an independent split, reproduce the same pattern. At the time this check was run, further hyperparameter search along these lines was judged exhausted, with the true limiting factor suspected to be demonstration volume rather than model capacity — a suspicion later tested far more rigorously and confirmed in Section 3.5.4.

**[FIGURE 3.4-A]** A grouped bar chart of Table 3.4's six policies × two metrics (position, orientation) would make the "naive beats every learned model" finding visually immediate in a way the table alone understates.

### 3.5 Model-improvement study: the "v2" retraining investigation (a rejected ablation)

Having observed in Section 3.4 that the naive mean-predictor rivals or beats every tuned learned model, this project undertook a dedicated, systematic investigation of *why*, and of whether the models could be substantively improved. This work is reported here as its own section, in full, because — per the project's established practice of disclosing negative and split results rather than only positive ones (already applied to the O-space validation, Section 3.3.1) — a rejected but rigorously executed improvement attempt is stronger evidence for the dissertation's central claims than simply not attempting one, and because the diagnosis it produced materially sharpens the account given in Section 3.4.

#### 3.5.1 Seven candidate improvements, assessed

Before building anything, seven possible causes of the models' weak offline performance were listed and individually assessed for feasibility:

| # | Proposed improvement | Verdict |
|---|---|---|
| 1 | Collect more independent approach demonstrations | **Not possible within this project.** PLUS-HRI is a fixed corpus of 24 sessions; new collection requires fresh ethics approval and participants, outside this project's scope and timeframe. |
| 2 | Train on the whole approach, not only the final metres | **Implemented** — this is the "v2" re-segmentation described below. |
| 3 | Add stronger social/geometric features (metric distance to group, O-space radius, nearest-person distance) | **Partly implemented.** These metric quantities are *not derivable* from the recordings, since person positions exist only as pixel boxes from an uncalibrated camera. Angular equivalents were added instead. |
| 4 | Reduce the train-to-live feature mismatch | **Implemented** — every new v2 feature is an angle or a distance ratio, never a raw pixel value, so the same quantity is computable identically offline and in simulation. |
| 5 | Sample by approach event, not by row | **Implemented** — `event_id` tracking, inverse-frequency row weighting, and 10 Hz decimation within each event. |
| 6 | Keep the MLP small and regularised | **Already the case.** The tuned 32/16, α = 0.1 network (Section 3.4) already outperformed the original 128/64 specification. |
| 7 | Try a stronger tabular regressor | **Implemented** — `HistGradientBoostingRegressor` (Ke *et al.*, 2017's LightGBM histogram-binning strategy, as implemented in scikit-learn, itself grounded in Friedman's (2001) general gradient-boosting formulation). |

#### 3.5.2 The diagnosis: v1 labelled adjustments, not approaches

Reconstructing the properties of all 462 v1 events revealed the underlying problem directly:

| Property of a v1 "approach event" | Value |
|---|---|
| Median distance from event start to the stop pose | **0.17 m** |
| Median event duration | **2.5 s** |
| Events starting more than 1.0 m from the stop | 53 / 462 (11%) |
| Events starting more than 2.0 m from the stop | 13 / 462 (3%) |
| Rows within 0.5 m of the stop | **72%** |
| Rows beyond 2.0 m from the stop | 3.6% |

The typical labelled "demonstration" was a 2.5-second shuffle ending 17 cm from where it began. The root cause lies in `find_approach_events()`: it pairs each *moving segment* with the stop that follows, where "moving" is any 1.0 s bin above a 0.05 speed threshold. A human's walk toward a group naturally contains pauses, turns and hesitations, so it is fragmented into many short segments, and only the *final* fragment before the eventual stop is labelled. **This is the single best explanation for the behaviour observed live**, both in the original 20-August draft of this chapter and confirmed again in the final 60-trial experiment (Section 3.6): the models produce sensible predictions within roughly a metre of a group and unreliable ones from further out, because they were trained almost exclusively on the terminal phase of an approach.

#### 3.5.3 What v2 changed, and what it found

`build_approach_pose_dataset_v2.py` anchors on each sustained stop near a group and walks *backwards*, absorbing brief pauses, until it has accumulated at least 1.0 m of travel (or meets a stop longer than 3 s, or exhausts a 20 s lookback). Events with less than 1.0 m of total travel are discarded.

| | v1 | v2 |
|---|---|---|
| Events | 462 | **182** |
| Rows | 70,555 | 128,506 |
| Median event start distance | 0.17 m | **1.37 m** |
| Median event travel | — | 1.98 m |
| Median distance-to-go | 0.183 m | **0.784 m** |
| Rows within 0.5 m of stop | 72% | **35.9%** |

Six new angular/ratio features were added (`group_span_rad`, `nearest_person_span_rad`, `gap_bearing_rad`, `gap_width_rad`, `person_spacing_rad`, `people_visible`), and the original seven v1 features were retained unchanged, so that v1-trained models could also be scored on v2's more genuine approach rows. All models were compared on the **same** held-out test sessions (5, 9, 59):

| Panel | Models scored | Best pos. (m) | Best orient. (°) | Best "both" |
|---|---|---|---|---|
| A — v1 models, v1 test rows | *(reproduces Table 3.4 above, to 4 d.p.)* | 0.305 (rule) | 25.78 (RF) | 43.3% (RF) |
| B — v1 models, v2 (genuine-approach) test rows | v1 Random Forest, v1 MLP, naive, rule | **0.652** (v1 MLP) | 34.91 (v1 RF) | **23.5%** (v1 RF) |
| C — v2 models (re-segmented data), v1 features only | RF, MLP, Gradient Boosting, all v2-trained | 0.722 (RF) | 34.93 (RF) | 15.5% |
| D — v2 models, v2 features (re-segmentation + new geometry) | as C, plus six new angular features | 0.722 (RF) | 36.55 | 15.5% |

**Reading this table is the central methodological finding of the study.** Moving from v1's easy terminal-adjustment test rows (Panel A) to v2's genuine-approach test rows (Panel B) roughly **doubles every policy's position error** — the naive baseline from 0.410 m to 0.724 m, the tuned Random Forest from 0.365 m to 0.656 m — and the proportion of rows meeting both Objective 4 thresholds falls from 43.3% to 23.5%. **The published offline evaluation in Section 3.4 was, in effect, being scored on the easiest part of the task.** More strikingly, **the ranking between the rule-based and learned models reverses**: on terminal adjustments the rule baseline *wins* on position (0.305 m vs 0.365 m); on genuine approaches it becomes the **worst** policy tested, beaten even by predicting the training-set mean (0.765 m vs 0.724 m), while the Random Forest becomes the best (0.656 m, 23.5% within both thresholds). This directly supports the dissertation's central claim: a fixed geometric rule — face the group, stop a fixed distance short — is adequate for the last half-metre and degrades as soon as there is a real approach to plan, which is consistent with the live simulation result (Section 3.6) where the Random Forest matched the rule's task success using less than half the path length.

Re-segmentation alone (Panel C) and re-segmentation plus new geometric features (Panel D) both perform **worse** than simply keeping the original v1-trained model and testing it on genuine approaches (Panel B: 0.652 m). **The attempted improvement did not improve anything.**

#### 3.5.4 Why re-segmentation failed: three follow-up studies

The v2 regression prompted three further, narrower investigations, each holding as much constant as possible so that a single variable could be isolated.

**Does more training data recover the loss? (Yes, but only partially.)** Reducing the minimum-travel threshold for an event to count (thereby admitting more, shorter events) was swept with the test set and learner held fixed:

| Configuration | Train events | Position (m) | Orientation (°) | Within both |
|---|---|---|---|---|
| `min_travel = 1.0 m` (v2 default) | 120 | 0.722 | 34.93 | 15.5% |
| `min_travel = 0.5 m` | 178 | 0.657 | 34.69 | 19.6% |
| `min_travel = 0.25 m` | 216 | 0.648 | 35.22 | 21.9% |
| `0.25 m` + mirror augmentation | 432 | 0.652 | 34.01 | 22.1% |
| `0.25 m` + mirror + validation sessions folded in | **578** | **0.642** | **33.56** | **24.2%** |
| *reference: shipped v1 Random Forest* | — | 0.656 | 34.91 | 23.5% |

Position error falls **monotonically** with training-event count, confirming that the earlier §3.5.2 diagnosis was correct: the re-segmentation itself was sound, and the corpus was simply too small to support it. Three qualifications matter for how this is reported, in the interest of not overstating a positive-sounding result: the gain over the shipped model is marginal (0.642 m vs 0.656 m, a 2.1% improvement); mirror augmentation is nearly inert (doubling 216 to 432 events moved position error by 0.004 m in the *wrong* direction, showing the limitation is genuine behavioural variety rather than raw row count); and the 578-event figure is mildly optimistic, since folding the validation sessions into training is legitimate only because hyperparameters were fixed in advance, but those hyperparameters were originally selected using a search that itself saw the validation split.

**Extrapolating this trend**: a 4.8× increase in training events (120 → 578) bought an 11% reduction in position error. On that trend, reaching the 0.4 m Objective 4 threshold would require substantially more independent demonstration data than the 24-session corpus contains — converting "collect more data" from a generic suggestion for future work into a quantified projection.

**Does a higher training frame rate help? (No — it makes things monotonically worse.)** Training rows are decimated to 10 Hz within each event by default; the recordings run at ~33 Hz. Sweeping the retained frame rate, with everything else held fixed:

| Training frame rate | Train rows | Position (m) | Orientation (°) | Within both |
|---|---|---|---|---|
| 2 Hz | 5,883 | **0.640** | **34.25** | **22.7%** |
| 5 Hz | 14,016 | 0.643 | 34.62 | 22.0% |
| 10 Hz (default) | 26,194 | 0.648 | 35.22 | 21.9% |
| 20 Hz | 43,967 | 0.653 | 35.22 | 21.3% |
| all rows (~33 Hz) | 117,857 | 0.667 | 35.90 | 20.1% |

Twenty times the rows costs 4.3% in position error. At 33 Hz the robot moves about a centimetre between consecutive samples and the group has not moved at all, so each additional row is a near-duplicate of its neighbour that dilutes, rather than strengthens, the trees' bootstrap statistics. **Read together, these two studies are the key methodological result of this whole investigation:** what matters is the number of independent demonstrated approach *events* (120 → 578 improved position error by 11%), not the number of *rows* (5,883 → 117,857 rows made it 4% *worse*). This retrospectively justifies a concern about how the original dataset should be described: its "70,555 rows" (Section 3.4) overstated its evidential content — the true sample size is 462 approach events (182 of them genuine approaches of more than a metre), and reporting row counts as though they were independent samples materially overstates the evidence available to the models.

**Does the model underfit, or does the label itself set a floor? (The label sets the floor.)** The tuned Random Forest was scored on the data it was *trained* on, to check whether it was failing to fit the training set (underfitting) or fitting it but failing to generalise (overfitting to label noise):

| | Position error |
|---|---|
| Predict the training-set mean (no learning at all) | 0.746 m |
| Random Forest, on its **own training rows** | **0.597 m** |
| Random Forest, on held-out test rows | 0.648 m |

An unconstrained forest (`max_depth=None, min_samples_leaf=1`) reaches 0.210 m on its training data — the model plainly *can* fit the data — but every increase in capacity tested made the **test** error worse (0.648 m at the shipped configuration, rising to 0.739 m unconstrained). The shipped configuration sits at, or very near, the optimum of this capacity/generalisation trade-off, measured rather than assumed.

The reason extra capacity does not generalise was measured directly: for each training row, its ten nearest neighbours in standardised feature space disagree on their labelled stop pose by a **mean of 0.505 m** — rows the model effectively cannot distinguish specify stop poses roughly half a metre apart. This **label-ambiguity floor** is an approximate lower bound on achievable test error, and it is a direct consequence of the target definition: `target_dx, target_dy` is the displacement from the robot's *current* pose to the eventual stop, and two moments with an identical group bearing, apparent group size and LiDAR range can sit at very different points along an approach — one two metres out, one thirty centimetres out — while carrying completely different displacement labels. The shipped model's 0.648 m test error sits only about 0.15 m above this floor, meaning roughly 19% of the theoretically available headroom was later shown (Section 3.5.5) to be reachable by hyperparameter search alone, and the remainder is not.

**A group-frame reformulation was attempted, and failed for an instructive reason.** If the target is ill-posed because it is measured from the *robot*, the obvious remedy is to measure it from the *group* instead — predicting standoff distance and approach bearing relative to the group centre, properties of the group's own configuration rather than of wherever the robot happens to be standing. This was implemented and tested:

| Target formulation | Train pos. | Test pos. | Test "both" |
|---|---|---|---|
| Robot-frame displacement (shipped) | 0.597 m | **0.648 m** | 21.9% |
| Group-frame standoff + bearing | 0.651 m | 0.747 m | 15.2% |

It performed **worse**. Reconstructing a group-frame goal requires knowing how far away the group actually is, and the only available estimate in this dataset is `lidar_min_range` — the same documented proxy already used as a feature — so that proxy's own error now enters the calculation twice: once when the training label is built, and again when a prediction is converted back to a usable robot-frame goal. **The root cause is a property of the corpus, not of the method**: the PLUS-HRI recordings provide person positions only as pixel boxes from an uncalibrated monocular camera, so no metric group distance is available to either the labels or the features, and this cannot be repaired by any choice of target formulation or model architecture.

#### 3.5.5 A systematic, 360-model hyperparameter search

Rather than rely on the earlier, smaller six-configuration sensitivity check (Section 3.4) to conclude that hyperparameter search was exhausted, `finetune_on_lab_pc.py` was written to search properly: **120 randomly sampled configurations for each of three model families** (Random Forest, Gradient Boosting, MLP — 360 models total), selected on the validation sessions (10, 14, 15, 54) and scored once, finally, on the held-out test sessions, with search ranges deliberately bracketing the shipped values in both directions so the search remained free to select a *smaller* model as well as a larger one.

| Model family | Position (m) | Orientation (°) | Within both |
|---|---|---|---|
| Random Forest | 0.636 | 34.05 | 24.8% |
| Gradient Boosting | 0.631 | 33.66 | 24.9% |
| **MLP** | **0.627** | **33.55** | **26.1%** |
| *shipped v1 Random Forest* | 0.656 | 34.91 | 23.5% |
| *label-ambiguity floor (§3.5.4)* | 0.505 | — | — |

The systematic search beats the shipped model by 0.029 m (4.4%) and 2.6 percentage points on the joint threshold — a real, measured improvement obtained with no new data. Three observations from this search matter more than the improvement itself. First, **all three families finish within 0.009 m of each other** — a random forest, a boosted ensemble and a neural network share no architectural assumptions, and their convergence on the same number is independent confirmation that the limitation lies in the data, not the learner. Second, **only 19% of the available headroom was captured**: the gap between the shipped model and the 0.505 m label-ambiguity floor is 0.151 m, and exhaustive search over 360 models recovered only 0.029 m of it. Third, **every family independently selected heavy regularisation** — the winning Random Forest used `max_depth=4, min_samples_leaf=40`, substantially more constrained than the shipped `depth=8, leaf=20`; Gradient Boosting chose `max_depth=2`; the MLP chose `α=10.0` — three independent searches all concluding smaller is better, consistent with a small, noisy training set. Notably the MLP, the weakest model in the original evaluation (Section 3.4), is the best of the three here once properly tuned; its earlier underperformance was a tuning artefact rather than an architectural limitation, though its winning configuration did emit a `ConvergenceWarning` at `max_iter=400`, so this figure carries a small residual uncertainty.

**These tuned models were not deployed.** The improvement (0.029 m offline) is real but small, and Section 3.6's own results (specifically Table 3.6.9, the offline/live ranking reversal) already establish that offline position error does not predict live social behaviour — deploying the tuned models would require thirty fresh live trials and a full re-run of the results tables in exchange for a quantity with no demonstrated relationship to task success or O-space intrusion. The value of this study is therefore evidential rather than operational: it confirms, with a properly powered systematic search rather than a six-configuration spot check, that the shipped configuration is already within 4.4% of the best findable, and that the residual error is a property of the dataset rather than of any specific model family.

#### 3.5.6 What this study changes about the reported project

Nothing reported elsewhere in this chapter or in Chapter 5 is invalidated by this study: `approach_pose_dataset.csv` and the tuned v1 models retain their original 8 August 2026 timestamps and were never overwritten, and no simulation trial needed to be re-run. What this study adds is threefold: a precise, quantified diagnosis of *why* the offline models underperform (label ambiguity, §3.5.4); direct evidence that the rule/learned ranking on genuine approaches is the *reverse* of the ranking on the originally-reported terminal-adjustment test set (§3.5.3), strengthening rather than weakening the case for the learned models; and a converted, quantified argument for future work — 182 usable approach events across 24 sessions is demonstrably too few, and this study shows precisely what re-segmentation costs when the corpus is not correspondingly enlarged.

**[FIGURE 3.5-A]** A single figure showing the Panel A → Panel B position-error jump for all four policies (naive, rule, RF, MLP), i.e. "terminal-adjustment test rows" vs "genuine-approach test rows" side by side, would communicate the ranking reversal in Section 3.5.3 more immediately than the two tables read separately.

**[FIGURE 3.5-B]** A simple two-line plot of position error against training-event count (120 → 578, from the §3.5.4 data-scaling sweep) directly visualises the "more demonstrations would help, and by how much" argument used in Future Work.

### 3.6 Simulation environment and system integration

#### 3.6.1 Platform and navigation stack

The live system runs on the TIAGo mobile-manipulator platform (Pagès, Marchionni and Ferro, 2016) in Gazebo Classic 11, using ROS 2 Humble as middleware and Nav2 (Macenski *et al.*, 2020) — specifically its global planner, DWB local controller, layered costmaps (Lu, Hershberger and Smart, 2014) and behaviour-tree executive — for path planning and execution. Both are cited here by their own system papers rather than only vendor documentation, since both are the actual software components this project's contribution is built on top of.

| Layer | Technology |
|---|---|
| Middleware | ROS 2 Humble |
| Simulator | Gazebo Classic 11 |
| Robot | PAL TIAGo (`tiago_gazebo`, `tiago_description`, `tiago_2dnav`) |
| Navigation | Nav2 (planner, DWB controller, `costmap_2d`, behaviour tree) |
| Transforms | TF2 |
| Detection | Ultralytics YOLOv8n; NVIDIA LocateAnything-3B (comparison condition) |
| Machine learning | scikit-learn, joblib |
| Data handling | NumPy, pandas, `rosbags` |

The package layout separates perception, the two policy families, mission control, localisation, and metrics into independent ROS 2 nodes:

```
src/tiago_group_approach/tiago_group_approach/
  group_perception_node.py          detection, depth back-projection, clustering
  group_approach_baseline_node.py   rule-based policy (geometric)
  bc_policy_node.py                 learned policy (Random Forest or MLP)
  mission_node.py                   scripted patrol tour and reporting
  gt_localisation_node.py           ground-truth map->odom at 30 Hz
  metrics_recorder_node.py          trajectory sampling and social scoring
```

**[FIGURE 3.6-A]** The package/node diagram above, redrawn as labelled boxes with ROS 2 topic names on the connecting arrows (e.g. `/group_centroid`, `/approach/start`, `/cmd_vel`), would give a much clearer system-architecture figure than the ASCII block above.

#### 3.6.2 World construction

The final evaluation world, `restaurant_testing.world`, superseded the earlier `restaurant_humans.world` used during initial development. It is a 20 × 15 m room (walls at x = ±9.9 m, y = ±7.4 m) containing five round dining tables (at (-4,-1), (-1,4), (2,-2), (6,1), (5,-4)), a buffet at (-5.5, -6.0), five plants, a stage at (7.45, 5.20), and kitchen partitions on the west side. Fifteen human actors are arranged into six targets:

| Target | People | Centre | O-space radius |
|---|---|---|---|
| Group A | 4 | (-3.50, -2.50) | 0.71 m |
| Group B | 3 | (4.67, 2.67) | 0.65 m |
| Group C | 5 | (5.60, -1.80) | 0.87 m |
| Solo 1 | 1 | (-6.0, -5.0) | — (no O-space) |
| Solo 2 | 1 | (-6.0, -7.0) | — (no O-space) |
| Walker | 1 (moving) | (4.0, 2.0) | — (no O-space) |

Only Groups A, B and C — those with two or more members — are treated as valid approach targets (`min_group_size = 2`), since a lone individual has no F-formation and no O-space, and approaching one therefore cannot demonstrate the group-approach behaviour under study; the solo actors and the walker remain in the scene purely as obstacles and distractors. A hand-built world was used rather than a procedurally generated one for the primary evaluation scenes specifically so that every group's true position, membership and O-space radius could be authored precisely and exported as a `.groundtruth.json` file — this ground truth is what makes the social metrics in Section 3.7 computable at all, since the recorded PLUS-HRI video never carried metric position data to evaluate against. A separate parametric generator (`generate_social_world.py`) exists for producing geometric variants (a 3-group/9-person "unseen" layout, and an "adjacent" layout for stress-testing whether clustering correctly separates two nearby groups) but the hand-built world above is the one used for all reported trials. The occupancy map Nav2 plans against is generated directly from the world's wall and furniture geometry (`world_to_map.py`) rather than by running SLAM — a deliberate choice producing an exact, drift-free map independent of a successful mapping run, at the cost of needing regeneration whenever the world geometry changes.

**[FIGURE 3.6-B]** A top-down floor-plan figure of `restaurant_testing.world` — walls, furniture, the three group centres with their O-space circles drawn to scale, and the patrol route overlaid — would be one of the single most useful figures in the whole chapter, since almost every subsequent result (task success band, O-space intrusion, cut-through, the P-space slot method in Section 3.7.6) refers back to this exact geometry.

#### 3.6.3 Live perception and localisation

`group_perception_node` closes the pixel-to-metres gap that constrained every offline stage: it subscribes to TIAGo's RGB and depth image streams and camera intrinsics, runs the selected detector (Section 3.2) at class 0 (person), confidence threshold 0.45, back-projects each detected person to real 3D coordinates using the pinhole relation `x=(u−cx)d/fx, y=(v−cy)d/fy, z=d`, transforms those coordinates into the map frame via the TF chain `camera_optical → base_link → odom → map`, and clusters them into groups in genuine world coordinates — publishing a real-metres `/group_centroid` that both policy nodes consume identically. This is the point in the pipeline where, in the project's own framing, offline data analysis becomes a robot system.

Rather than relying on AMCL — found, in this container, never to converge reliably — localisation is provided by a dedicated node, `gt_localisation_node`, which recomputes and republishes the map→odom transform continuously at 30 Hz directly from Gazebo's own simulator state (`T_map_odom = T_map_base · T_odom_base⁻¹`). An earlier, simpler approach — publishing a single *static* map→odom correction once at startup — was found to be correct only for an instant and to diverge progressively afterward, especially after wheel slip (which collisions reliably cause). This was not a cosmetic issue: it produced a cascade of downstream symptoms, including a detection initially misdiagnosed as a YOLO false positive at (4.4, 2.8) that was, in fact, a **correct** detection of a real person at (-3.0, 0.0), corrupted entirely by a stale transform. Replacing the static transform with the continuously-recomputed `gt_localisation_node` brought detections to within **10 cm** of ground truth. This is reported as a deliberate methodological choice with an explicit justification: **localisation was provided from simulator ground truth specifically so that navigation/localisation error could not confound the comparison between policies** — the policies are compared under identical, artefact-free localisation, isolating policy quality as the variable under test, while laser scanning, costmaps and planning all operate normally through Nav2 exactly as they would with any localisation source.

#### 3.6.4 The rule-based baseline: iterative, empirically-driven refinement

The deployed rule-based policy (`group_approach_baseline_node`) implements the geometric rule established in Section 3.3 — approach along the robot-to-centroid line, stop short by a standoff distance, face the centroid — but its live deployment required six rounds of empirically-driven correction beyond the initial design, each reported here as a measured methodological finding rather than routine debugging:

1. **Standoff distance**, defaulted to 1.2 m, is deliberately set at Hall's (1966) proxemics boundary between personal and social space, a principle also formalised as a repulsive force in Helbing and Molnár's (1995) social force model.
2. **Goal throttling** was added after early runs showed that publishing a fresh navigation goal on every perception frame (~2 Hz) caused Nav2 to continuously pre-empt and restart planning, so the robot barely moved. A new goal is now issued only if the target has moved by more than 0.40 m or the previous goal has finished.
3. **Per-person clearance, not just centroid standoff**, was added after a live run measured an O-space intrusion at 0.43 m from the nearest person — inside Hall's intimate distance — despite satisfying the centroid-based standoff, because the centroid itself shifts when only part of a group is visible to the camera. The chosen pose is now checked against every individually detected person and pushed back along the approach line until clear of all of them, with the target set to 0.7 m clearance to the *nearest person* rather than to the group centre.
4. **Body radii, not centre-to-centre distance**, were added after a further failure: treating clearance as a distance between point centres, rather than between TIAGo's physical footprint (~0.30 m radius) and a person's body (~0.25 m radius), left only 0.15 m of true physical gap at a nominal 0.7 m setting. The robot wedged itself into a group, came within **0.062 m** of a person, and remained stationary for 84% of a ten-minute run. Both radii are now added explicitly to the requested clearance, so a requested 0.7 m is a *delivered* 0.7 m.
5. **Gap-based approach selection** (`approach_mode: gap`, the current default) evaluates candidate standoff angles around the group and selects one with at least a 60° clear arc, rather than assuming the direct robot-to-centroid line is itself unobstructed.
6. **Stuck detection and an "unwedge" reflex**: if the robot has not moved more than 0.10 m in 25 seconds, and 30 seconds have elapsed since the trial began, it reverses at −0.15 m/s for 3 seconds before retrying — a reactive recovery behaviour for cases where Nav2's own planner cannot find a way out unassisted.

#### 3.6.5 Faults found and corrected — full record, grouped thematically

Beyond the rule-based policy's own refinement above, a substantial number of engineering and measurement faults were found and corrected across the wider system between the first report and the final experiment. **Every one of these materially affected either robot behaviour or the correctness of a reported measurement, and several had been silently invalidating results before discovery.** They are grouped thematically below for readability, but each is individually numbered exactly as it appears in the project's own fault log, so nothing is collapsed or omitted.

**(a) World and simulator setup.**
- *Stale world file.* Gazebo loads worlds from a copy held by `pal_gazebo_worlds`; edits to the project's own world file had no effect until that copy was refreshed. The bring-up script now syncs it on every run and reports when the installed copy was stale.
- *Missing ground truth.* `restaurant_testing.groundtruth.json` did not exist initially, and the pipeline refused to start without it. It is now generated from the world file and regenerated after every world change.
- *`map_server` never configured.* This is a lifecycle node whose `configure` step loads the map, and PAL's launch files set no `yaml_filename` for a custom world, so `configure` failed silently and the node stayed `unconfigured` — no `/map`, no map frame, no planning possible. The parameter is now set explicitly before configuring.
- *CameraInfo QoS mismatch.* `camera_info` was subscribed with the default **RELIABLE** QoS profile while Gazebo publishes sensor data as **BEST_EFFORT**. ROS 2 QoS compatibility is one-way: a reliable subscriber silently receives nothing from a best-effort publisher. The topic listed correctly and `count_publishers` returned 1, yet no message ever arrived — this was the cause of every `Waiting for CameraInfo...` hang. Fixed by subscribing with the sensor QoS profile.
- *Furniture absent from the occupancy map.* `world_to_map.py` originally rasterised only `<model>` elements using box geometry; the five dining tables are `<include>` blocks referencing a mesh, so **all five were entirely missing from the occupancy map** and Nav2 planned straight through them. Include elements are now rasterised as 1.2 m squares, taking the map from 9 to 20 obstacles.

**(b) Localisation.** Covered in full in Section 3.6.3 above (the static-transform-to-`gt_localisation_node` replacement) — the single most consequential fault in the whole record, since it corrupted the map-frame position of every detection made under the earlier scheme.

**(c) Goal arbitration between the mission and the policy.**
- *Policy and mission fighting over Nav2.* Both the mission node and the active policy node send `NavigateToPose` goals to the same action server, where a new goal pre-empts the old one. Logs showed approach goals accepted and reported "finished" **six milliseconds** later — the robot never executed a single approach. Resolved with explicit arbitration: the policy publishes `/approach/start` before it begins driving; the mission cancels its own goal and stands down until it receives `/approach/complete` or a timeout expires.
- *No goal throttling in the learned policies.* Throttling (item 2 in Section 3.6.4) had been added to the rule policy but never to `bc_policy_node`, so both learned policies issued a fresh goal on every perception frame (~2 Hz), each pre-empting the last. Fixed with the same 2 s minimum interval and 0.4 m re-issue threshold as the rule policy.
- *Approach never completed, and never released the mission.* The policy originally published completion the instant Nav2 reported arrival, so the mission resumed immediately and the robot rolled on without pausing. A **6 s dwell** was added, and arrival is now verified by position (within 0.75 m of the intended pose) rather than trusting Nav2's own status flag, which reports "finished" for aborted goals too.
- *No memory of attempted groups.* The policy would retry the same unreachable group indefinitely — one run lasted 30 minutes, drove 208 m, and was 65% stationary. Group positions are now remembered in 1.5 m cells and retired after three failed attempts or one success.
- *Coverage dominated by policy convergence.* The mission originally yielded control to a single approach attempt for up to 45 s with no overall limit, so a policy whose predictions rarely converged could hold the robot indefinitely — measured coverage was 34 map cells for the rule policy versus only 14–15 for the learned policies, which never crossed the centre of the room. A **total approach budget of 120 s per run** was introduced, so the patrol tour completes regardless of which policy is driving.

**(d) Measurement and scoring integrity.**
- *Collision metric measuring the robot's own body.* Every trial in the first full experiment reported a collision, with `min_obstacle_range_m` between 0.200 and 0.267 m across all nineteen early runs, twelve of them at exactly 0.200 m. This was not nineteen genuine collisions but a single constant: the base laser returning the robot's own chassis. With a 0.30 m collision threshold this flagged a collision in every trial, and because task success is gated on collision-free operation, **success was measured as 0% for all three policies for reasons entirely unrelated to their actual behaviour.** Fixed by calibrating the self-hit radius during a startup grace period and discarding returns below it, after which the collision-free rate rose to 100% under normal operation.
- *Task success scored against a phantom.* `task_success` was originally evaluated against `goal_centroid`, a variable overwritten by every incoming `/group_centroid` message and therefore holding whatever perception happened to see *last* — frequently a false positive near a wall. One MLP trial came within 0.51 m of a real person and registered a genuine O-space intrusion, yet was scored as never having held a valid pose, because it had approached a real group correctly and was then judged against a phantom target. Success is now scored against **ground-truth groups**, with distance and heading required to hold in the same trajectory sample. Re-scoring the existing trials under this correction raised MLP task success from 20% to 60% and BC-Random-Forest from 60% to 90%.

**(e) Policy behaviour.**
- *Gap selection chose the far side of the group.* Candidate gaps were originally sorted by width alone, so the robot repeatedly selected an opening 175–179° around the group — walking around, and therefore effectively through, the people to reach it (logged directly: *"177 deg gap ... 179 deg off the robot's current side"*). Any gap of at least 60° is now considered adequate, and the **nearest** adequate gap is chosen instead of the widest. On a test case with a five-person group and the robot approaching from the west, the old rule sent it to (7.31, -1.41), 149° around the group; the new rule sends it to (3.64, -1.19), 0° off.
- *Clearance measured centre-to-centre* (fault 5.10, described in full in Section 3.6.4 item 4 above).
- *Recovery that made things worse.* An earlier stall-recovery reflex reversed the robot's base blindly; since TIAGo's laser covers only the forward arc, the robot repeatedly backed into furniture it could not see. It now checks the rear beams and rotates in place instead if the space behind is not confirmed clear. A second, related fault disabled the reflex precisely when the robot had been stuck longest, because it keyed off "a goal was recently *sent*" rather than "a goal is currently *in flight*" — corrected to track the latter.
- *`num_people` frozen at inference.* This feature was hard-coded to the constant `3.0`, with a comment promising it would be "refined below if perception reports it" — a refinement that never happened. One of the seven features the models were trained on was therefore a constant at inference time, while training data spanned values from 1 to 6. It is now taken from the live detection count (feature importance for `num_people` is 0.055, so the effect was real but modest, not catastrophic).
- *Predictions rejected rather than clamped.* An implausible model prediction previously caused the policy node to send **no goal at all**, so the robot simply stood still — behaviour easily misdiagnosed as "the model isn't working" rather than "a single prediction was out of range." Over-long predictions are now scaled back onto a plausible range instead of being discarded outright.

**(f) Diagnostics and operator feedback.**
- *Detection overlay only drawn on processed frames.* Perception runs at 2 Hz while the camera itself runs at ~15 Hz, so 13 of every 15 camera frames published no overlay, and the RViz panel appeared frozen or blank — easily read as "detection is not working" when it was, in fact, working correctly at its own intended rate. The overlay now redraws its most recently cached boxes on every skipped frame and publishes a status banner in every state.

**(g) Robot control verification.**
- *Arm tuck reported success it had not verified.* `play_motion2 home` was originally used to stow the arm before each trial; `home`, however, **extends** the arm rather than tucking it, and the verification check treated "the command returned exit code 0" as proof the arm had actually moved to the intended pose. The correct `tuck_arm` motion is now used, the arm controller is also commanded directly, and `/joint_states` is read back afterward to verify the arm actually reached the tucked configuration rather than trusting the command's return status.

**(h) Trial protocol.**
- *Trials starting from the previous trial's end pose.* Because the simulation stays running across trials for efficiency, each trial originally began wherever the previous one had ended — one trial's very first trajectory sample was already at (-2.46, -2.48), standing among people before the policy had acted at all. The robot is now explicitly teleported back to (0, 0) at the start of every trial.

**(i) Attempted but not adopted.**
- *Camera-based obstacle avoidance.* TIAGo's base laser scans at 0.2 m height and cannot perceive a tabletop at 0.75 m. An attempt was made to add the depth point cloud as a second Nav2 observation source via `pointcloud_to_laserscan`, which would in principle let the robot avoid furniture its laser cannot see. It destabilised the navigation stack within the time available for testing — in one configuration the robot did not move at all — and **was not used for the reported experiments**. The implementation is retained behind a `CAMERA_OBSTACLES=1` flag and is named explicitly as future work rather than silently dropped.

**[FIGURE 3.6-C]** A single before/after diagram or annotated screenshot pair for the single most consequential fault — the static-vs-dynamic map→odom transform (Section 3.6.3) — showing the robot's Gazebo position diverging from its RViz-estimated position over time, would communicate why this particular fix mattered more clearly than prose alone.

#### 3.6.6 The multi-group patrol mission and final experimental protocol

For the live experiment, TIAGo executes a fixed, repeatable patrol route (`mission_node`) rather than free exploration, so that every trial — under every policy and every detector — covers identical ground in identical order, meaning any measured behavioural difference reflects the policy under test rather than which part of the room the robot happened to visit:

```
(-5, 5)  →  (3, 6)  →  (8, 1)  →  (8, -6)  →  (-8, -4)  →  back to (-5, 5)
```

The mission yields control to the active approach policy at each detected group, bounded by a **120-second total approach budget per run** (Section 3.6.5c) rather than an unbounded convergence wait; dwells for **6 seconds** once an approach succeeds before resuming (Section 3.6.5c); retries a blocked waypoint for up to three attempts, to tolerate a temporarily obstructing person rather than treating them as a permanent obstacle; and retires a group after three failed approach attempts or one success, so a single unreachable group cannot stall the whole mission (Section 3.6.5c). The robot is teleported to (0, 0) at the start of every trial (Section 3.6.5h).

**The final experimental protocol**, as actually executed for the results reported in Chapter 5:

| Parameter | Value |
|---|---|
| Environment | `restaurant_testing.world`, 20 × 15 m (Section 3.6.2) |
| People | 15 actors: 3 conversational groups (4, 3, 5 people), 2 lone individuals, 1 walker |
| Valid approach targets | Groups of ≥ 2 people only (`min_group_size = 2`) |
| Trials | 10 per policy per detector; **60 trials in total** (3 policies × 2 detectors × 10 repeats) |
| Localisation | Simulator ground truth, republished at 30 Hz (Section 3.6.3) |
| Trajectory sampling | 10 Hz |
| Success criterion | Distance ∈ [0.5, 2.0] m of a real group centre **and** heading within 45° of it, in the same sample (full derivation in Section 3.7.3) |
| Trial termination | Route completed, 60 s of no motion (after a 90 s startup grace period), or a 30-minute hard timeout |
| Re-scoring | Every trial's full 10 Hz trajectory is retained, permitting all metrics to be recomputed offline (`rescore_sim_results.py`) against corrected definitions without re-running the simulation |

The experimental infrastructure supporting this protocol is itself a small pipeline of purpose-built scripts:

| Script | Purpose |
|---|---|
| `run_everything.sh` | Full bring-up: process cleanup, world sync, Gazebo + TIAGo + Nav2 launch, arm tuck, map activation, ground-truth localisation, RViz |
| `run_pipeline.sh` | One trial: pose reset, target selection, node launch, rosbag recording, automatic exit |
| `run_trials.sh` | A batch of trials across policies |
| `run_overnight.sh` | The full unattended experiment: both detector conditions, separate result folders, wall-clock budgets |
| `rescore_sim_results.py` | Re-scores recorded trials against ground truth without re-running the simulation |
| `drive_test.sh` | Diagnostic: drives the base directly, bypassing Nav2, to isolate hardware/simulation issues from navigation-stack issues |

All pipeline nodes are force-killed between trials so that no process state contaminates the next run, and a stalled trial still writes a valid, partial results file rather than being killed mid-write and losing its data.

An important methodological correction to the target-selection protocol itself is worth reporting explicitly. Before `min_group_size` was fixed at 2, an earlier batch of 61 trials scored approaches to targets of any size, including lone individuals:

| Policy | Approached a real (2+) group | Approached only lone individuals |
|---|---|---|
| Rule-based | 12/12 (100%) | 0 |
| BC – Random Forest | 21/26 (81%) | 4 |
| BC – MLP | 4/23 (17%) | 11 |

**The MLP was not avoiding groups; it was being handed lone individuals by perception**, since targets of size 1 were admissible at that time, and its apparently excellent O-space score in that earlier batch was partly an artefact of rarely approaching a real group at all. This motivated restricting valid approach targets to groups of two or more for every result reported in the final experiment.

**[FIGURE 3.6-D]** The patrol route figure belongs together with the world floor-plan already flagged as [FIGURE 3.6-B] — plotting the five waypoints and the loop back to start over the same floor plan, rather than as a separate figure, keeps the world geometry and the mission geometry visually unified.

### 3.7 Evaluation metrics *(responds to supervisor comment [AP5])*

*This section is built directly from `Section_3_6_Evaluation_Metrics.md`, a document written specifically to answer this comment, and is reproduced here near-verbatim with cross-references updated to this chapter's own numbering.*

The evaluation uses two families of metrics, separated by what each one requires in order to be computed honestly. **Offline metrics** are calculated directly from the recorded human demonstrations and measure how closely a policy reproduces the pose a human demonstrator chose. **Simulation metrics** require an executed trajectory and metric person positions, and measure how the robot actually behaves around people. This separation is not cosmetic: the PLUS-HRI recordings provide person positions only as two-dimensional pixel coordinates from uncalibrated video, so any spatial metric expressed in metres — O-space intrusion, minimum distance to a person, group cut-through — cannot be derived from the dataset without fabricating a scale factor, and is therefore measured exclusively in simulation, where ground-truth person positions are known in metres from the world definition (Section 3.6.2). Metrics that are properties of a *path* rather than a *prediction* — path length, navigation time, collision-free rate, task success — likewise require Nav2 to have actually driven the robot, and so are also simulation-only.

#### 3.7.1 Offline metrics

Both offline metrics are computed on the three held-out test sessions (5, 9, 59), excluded from training and validation (Section 3.1, Section 3.4). Each test row records a moment at which a human demonstrator completed an approach, so the demonstrated stop pose is known in real metres and radians from the robot's own odometry, and each policy predicts a pose in the same robot-relative frame, making the comparison direct.

**Approach-position error** is the Euclidean distance between the predicted stop position and the demonstrated stop position:

```
e_pos = sqrt( (x_pred − x_demo)² + (y_pred − y_demo)² )      [metres]
```

**Approach-orientation error** is the absolute difference between predicted and demonstrated heading, wrapped to the interval (−π, π] so that a 350° error is correctly reported as 10°:

```
e_yaw = | atan2( sin(θ_pred − θ_demo), cos(θ_pred − θ_demo) ) |    [radians → degrees]
```

The acceptance thresholds are inherited unchanged from Objective 4 of the project proposal: **position error below 0.4 m** and **orientation error below 20°**. For each policy, four figures are reported — mean and median position error, mean and median orientation error — plus the percentage of test rows meeting each threshold independently and jointly, since a policy may have an acceptable mean while satisfying the threshold on only a minority of individual cases. Two reference policies are evaluated alongside the learned models specifically to make the numbers interpretable: a **naive** policy that always predicts the training-set mean pose, establishing the floor below which a model has learned nothing useful, and the **rule-based** policy applied offline, the comparison point named directly in Objective 4.

#### 3.7.2 Simulation metrics

Seven metrics are recorded per trial by `metrics_recorder_node`, which samples the robot pose at **10 Hz** from the TF transform `map → base_footprint` and writes a JSON record containing both summary values and the full trajectory. All social metrics are scored against the world's **true, ground-truth person positions**, never against the robot's own detections — a deliberate choice avoiding a circularity that would otherwise flatter a failing perception system: a policy that never detected a group would report no intrusion on that group, and would be rewarded for its own blindness rather than penalised for missing it.

| Metric | Type | Definition |
|---|---|---|
| Task success | binary | A socially valid pose was achieved at some point in the trial |
| Collision-free | binary | No obstacle within 0.30 m, after self-hit calibration |
| O-space intrusion | binary + count | The robot's footprint overlapped a group's O-space |
| Minimum distance to any person | metres | Closest approach to any group member across the whole trial |
| Group cut-through events | count | Path segments passing between two members of the same group |
| Path length | metres | Total distance travelled |
| Navigation time | seconds | Trial duration |

**Task success.** A trial succeeds if, **at any sample**, the robot simultaneously satisfies a distance and a heading condition with respect to **any** ground-truth group, and no collision occurred during the trial:

```
0.5 m ≤ d(robot, group_centre) ≤ 2.0 m        AND
heading error to group centre ≤ 45°           (same sample)
```

Three design decisions in this definition are worth explaining, since each one corrects a specific, previously-observed measurement failure. **Scored at any point, not the final pose**, because the robot executes a patrol mission and returns to its start point by design, so scoring only the final sample marked every trial a failure regardless of actual behaviour — one trial reached 1.101 m from a person, a textbook social distance, and was recorded as failed purely because it subsequently drove home. **Distance and heading must hold in the same sample**, because scoring the single closest sample was also unsound: the nearest point on a path is frequently a tangential fly-past in which the robot is at the correct distance while travelling across the group's face, and such passes scored 88–90° heading error and failed three trials in which the robot had, in fact, settled at 1.50 m facing the group within one degree. **Scored against every ground-truth group**, because an earlier implementation scored against the last perceived centroid, which at trial end held whatever perception happened to see most recently — often a false positive against a wall — and this is the same phantom-scoring fault documented in Section 3.6.5(d). The lower bound of 0.5 m (rather than 0.8 m) reflects the approach policy's own geometry directly: it targets 0.7 m clearance to the nearest *person* while standing in a formation gap, and for a tight group that legitimately places the robot well inside 0.8 m of the group's *centre*.

**O-space intrusion.** A group's O-space is the shared transactional space enclosed by its F-formation (Kendon, 1990). Intrusion is recorded when the robot's **footprint**, not merely its centre point, overlaps that circle:

```
d(robot_centre, group_centre) < r_ospace + r_robot
```

where `r_robot = 0.27 m` is the PMB2 mobile base's actual radius (distinct from the ~0.30 m approximate radius used inside the rule-based policy's own clearance calculation in Section 3.6.4 — the two serve different purposes: one is a conservative planning margin, the other is the precise value used to score intrusion after the fact) and `r_ospace` is taken directly from the world's ground truth, derived from the members' spatial extent with a floor of 0.4 m for very tight pairs and a default of 0.7 m. A robot half inside a conversation has intruded, and the footprint test captures that where a point test would not. Intrusion is tracked per group and reported two ways: a per-trial boolean (did the robot intrude on *any* group) and a count of distinct groups intruded upon.

**Collision-free rate.** A collision is recorded when any lidar return falls below **0.30 m**. This metric required a calibration step: during an initial 5-second grace period, the recorder observes the minimum range the stationary robot reports (its own chassis, as diagnosed in Section 3.6.5(d)), adds a 0.05 m margin, and discards all subsequent returns below that calibrated floor, as well as any return below the sensor's own `range_min`. Without this correction the fixed self-return (0.200–0.267 m across nineteen early runs, twelve exactly at 0.200 m) fell inside the collision threshold and flagged a collision in every trial before the policy had even acted — and because task success is gated on collision-free operation, this forced 0% measured success for all three policies for reasons unrelated to their actual behaviour.

**Minimum distance to any person.** The smallest Euclidean distance between the robot and any group member, taken across all samples and all members — the metric most directly comparable to Hall's (1966) proxemic zones, whose personal zone spans 0.45–1.2 m.

**Group cut-through rate.** A count of path segments that pass **between two members of the same group**, implemented as a line-segment intersection test between each travelled segment and the line joining each pair of members. This is deliberately distinct from O-space intrusion: a robot can clip the edge of an O-space without ever walking through the middle of a conversation, and walking through the middle is judged the more socially disruptive of the two failures.

**Path length and navigation time.** Path length is the sum of Euclidean distances between consecutive 10 Hz samples; navigation time is the trial's wall-clock duration. Both are efficiency measures rather than social measures, reported to establish that socially appropriate behaviour is not being purchased at an unreasonable cost in travel.

#### 3.7.3 Scoring procedure and reproducibility

Every trial writes its full 10 Hz trajectory to disk alongside its summary values, which permits **offline re-scoring**: `rescore_sim_results.py` recomputes all social metrics from the stored trajectory against ground truth, so a correction to a metric definition (such as the task-success and collision fixes in Section 3.6.5(d)) can be applied uniformly to trials already recorded, without re-running the simulation. Both detector conditions are re-scored with identical criteria and an identical minimum group size before being summarised, ensuring the two batches are judged the same way. Only groups of two or more people are treated as approach targets throughout, per the correction described in Section 3.6.6.

#### 3.7.4 Statistical treatment

Each policy is run for ten trials per detector condition. Binary metrics — task success, collision-free rate, O-space intrusion — are compared between conditions using the **two-sided Fisher exact test** (Fisher, 1922), appropriate for small samples and free of the asymptotic assumptions a chi-squared test would require. Continuous metrics are reported as means with ranges. At n = 10 per condition the study is powered to detect large effects only, following the small-sample experimental guidance for human-robot interaction studies set out by Hoffman and Zhao (2020); where a difference is substantial but does not reach p < 0.05, this is stated explicitly and the effect size is reported rather than the difference being described as statistically significant. This discipline is applied consistently throughout Chapter 5: only three comparisons in the entire 60-trial experiment reach p < 0.05 (O-space intrusion, rule vs BC-MLP, p = 0.003; task success, BC-Random-Forest vs BC-MLP and rule vs BC-MLP, both p = 0.011; O-space intrusion by detector, p = 0.001), and several visually large differences (e.g. path length, 72.56 m vs 33.22 m) are reported as effect sizes on a continuous measure rather than dressed up with a significance test the sample size cannot support for a binary outcome.

#### 3.7.5 Limitations of the metric set

Three limitations should be borne in mind when this metric set's results are read in Chapter 5. First, **task success as defined is permissive**: it requires that a valid pose was achieved at some instant, not that it was held — a separate dwell requirement (Section 3.6.6, the 6-second dwell) is enforced by the mission logic but is not itself part of the success criterion. Second, the **45° heading tolerance is generous** relative to the 20° threshold applied offline (Section 3.7.1); it was chosen so that a robot correctly positioned in a formation gap, and therefore not facing the geometric group centre exactly, is not unfairly penalised. Third, **O-space radii are estimated** from the spatial extent of the group members rather than observed from participants' true orientations, since Gazebo's actor models do not encode gaze; the 0.4 m floor prevents degenerate radii for closely seated pairs but remains an approximation.

#### 3.7.6 A complementary, non-saturating metric: F-formation slot-based approach accuracy

The task-success metric above is a single per-trial boolean, and in the final 60-trial batch it **saturated**: every policy scored 100% under YOLOv8n on task success alone, at which point the metric can no longer distinguish between policies at all — a metric sitting at its ceiling measures nothing. A complementary, continuous metric was therefore developed (`scripts/approach_accuracy.py`) to ask the sharper question the project is actually about: for each group in the room, **how close** did the robot get to a socially correct place to stand, not merely *whether* it eventually got close enough somewhere.

**Deriving the reference positions from theory, not by hand.** For each group of two or more people, the bearing of every member as seen from the group centre is computed and sorted, and the angular gap between each pair of neighbouring members is measured. These gaps are the genuine openings in the formation — the P-space slots a person would step into to join the conversation. Every gap wider than 45° is kept (a narrower gap is two people standing shoulder to shoulder, not a way in), and a slot is placed on the bisector of each kept gap, at a radius of `ospace_radius + 0.6 m` beyond the group's O-space, facing back toward the centre. A four- or five-person group therefore yields several slots rather than one, matching the intuition that there are a handful of correct places to stand and many incorrect ones; standing at *any* slot counts as correct, since the robot is not required to guess a particular one. Because this construction is a fixed geometric rule applied identically to every group, nothing is hand-tuned to flatter a particular result, and the method uses the same formation-gap theory (Kendon, 1990) already underlying the O-space and P-space vocabulary used throughout this chapter.

**What is reported, and why the search logic itself went through three iterations.** For each (trial, group) pair, the whole 10 Hz trajectory is searched for a sample that best satisfies both a distance and a heading condition — and getting this search logic right was itself a small methodological investigation, documented directly in the script's own commit history, worth reporting because each iteration corrects a specific, diagnosed measurement artefact of the kind already seen twice elsewhere in this chapter (Sections 3.6.5(d) and 3.7.2's task-success redesign):

1. The first version took the single sample **nearest** to a slot and then tested only *that* sample's heading. This produced 0/30 "reached" outcomes on a batch whose worst distance error was only 0.446 m — every trial was inside tolerance on distance, and every one was rejected purely because heading was measured at the wrong instant, exactly the tangential fly-past problem already diagnosed for task success.
2. The second version instead considered every sample already within distance tolerance and took the **best heading among them**. This produced a striking but misleading result: 100% of group-visits came within 0.5 m of a slot, yet the best heading found while there was uniformly 48–75° off, identically across all six policy/detector conditions — a pattern too uniform to be genuine behaviour. The cause was that the slots (placed at `ospace_radius + 0.6 m` from the centre) and the policy's own actual target (1.35 m from the nearest *person*) are not the same location, so the robot was clipping the slot region while driving *past* it toward its own goal, and was scored mid-transit, facing its direction of travel rather than the group.
3. The final version restricts the heading judgement to **near-stationary samples only** (speed ≤ 0.10 m/s). Movement between consecutive samples gives a speed estimate; any sample above this threshold is the robot in transit and is excluded from the heading judgement, so the metric now asks specifically where the robot stood **when it stopped to engage the group**, not where it happened to be pointing while passing through.

The final reported fields, per (trial, group) pair, are: `nearest_slot_error_m` (the closest the robot ever came to a slot, over the whole trajectory), `within_distance` (whether it was inside a 0.5 m tolerance of a slot at some point), `heading_error_deg` (the best heading error among near-stationary, in-tolerance samples), and `reached` (`within_distance` **and** heading within a 45° tolerance). Reporting `within_distance` separately from `reached` matters methodologically: "got to the right place but never turned to face the group" and "never got to the right place at all" are different failures and are not collapsed into a single number.

**Results.** Applied to the same 60-trial batch (three conversational groups per trial × 10 trials = 30 group-visits per policy per detector):

| Detector | Policy | Position error (mean) | Position error (median) | Orientation error (mean) |
|---|---|---|---|---|
| YOLOv8n | rule | 0.250 m | 0.261 m | **64.6°** |
| YOLOv8n | bc_ft (Random Forest) | 0.236 m | 0.241 m | 74.7° |
| YOLOv8n | mlp_ft (MLP) | **0.158 m** | **0.085 m** | 79.1° |
| LocateAnything-3B | rule | 0.479 m | 0.254 m | **56.8°** |
| LocateAnything-3B | bc_ft (Random Forest) | 0.348 m | 0.191 m | 66.4° |
| LocateAnything-3B | mlp_ft (MLP) | **0.255 m** | 0.178 m | 79.4° |

**Both learned policies beat the rule baseline on positional accuracy, under both detectors, on both the mean and the median** — the ordering mlp_ft < bc_ft < rule is identical across two independent detector conditions and survives the switch from mean to median, which is what makes it a genuine result rather than an artefact of a small number of unlucky trials. **The orientation ranking inverts**: the rule baseline is *best* on orientation under both detectors (its facing is hard-coded, so it turns to the group centre by construction), while the MLP — the most positionally accurate — is *worst*. Read together with Section 3.4's offline finding that no policy met the 20° orientation threshold, this gives a coherent account of the system's actual behaviour: **behavioural cloning from non-expert demonstrations learns where to stand better than a hand-coded geometric rule does, but does not learn which way to face as reliably as simply pointing at the group** — a hybrid combining a learned standoff position with a geometric final-orientation correction is directly motivated by this pair of findings, and is exactly the disclosed `approach_guard` intervention already used in the deployed policies (Section 3.9).

**[FIGURE 3.7-A]** A grouped bar chart of the six position-error figures in the table above (three policies × two detectors), with error bars or a box-plot overlay to show the spread, would communicate the "learned beats rule on position, loses on orientation" finding at a glance.

**[FIGURE 3.7-B]** A schematic diagram of one group's F-formation slots (bearings, the ≥45° gaps kept, the resulting slot positions at `ospace_radius + 0.6 m`) drawn over that group's actual layout would make the slot-derivation method in this subsection far easier to follow than the prose description alone. Group C (five people, the "loose ring" case) is the most illustrative choice, since it has the largest O-space and the least well-defined single "correct" opening.

### 3.8 Ethical considerations

This project uses the pre-existing PLUS-HRI dataset of previously recorded non-expert teleoperation sessions, rather than recruiting or recording any new human participants; live evaluation is conducted entirely against simulated actor models in Gazebo, not real people. *[Saaru: state plainly here whatever your actual LEAS/ethics position is — e.g. that no new ethics application was required because no new human data was collected, and reference whatever approval covered the original PLUS-HRI recording if you have that documentation. This is a direct requirement of Learning Outcome LO2 and needs an explicit, honest answer here rather than being left implicit.]*

Beyond formal ethical approval, two further considerations are worth a critical paragraph each. First, a perception model trained predominantly on certain body types, clothing, or lighting conditions may recall people unevenly across a real deployment population — a limitation inherited from YOLOv8n's own training data rather than introduced by this project (Section 3.2), but worth naming as a professional consideration for any future real-world deployment. Second, a robot that autonomously approaches groups of people in a public or semi-public space raises a legitimate social-acceptability question independent of navigation accuracy: a robot can satisfy every metric in Section 3.7 and still be experienced as intrusive by the people it approaches, which is precisely the gap between this project's geometric proxy metrics and genuine social comfort, acknowledged explicitly in Section 3.0 and returned to below.

### 3.9 Critical discussion of the chosen methodology

Several methodological choices in this project trade rigour in one dimension for feasibility in another, and a Masters-level treatment requires naming the trade rather than only the outcome.

The decision to evaluate exclusively via quantitative geometric metrics, with no qualitative user study, sacrifices any direct measurement of perceived comfort or naturalness in exchange for a comparison that is safe, repeatable and achievable within the project's timeframe — defensible given the ethical and time constraints already discussed (Section 3.0, 3.8), but one that limits every conclusion in Chapter 5 to "geometrically closer to the human demonstration" or "geometrically less intrusive" rather than "more socially acceptable to a bystander," and this project does not claim these are proven to be the same thing.

The decision to train via plain offline Behavioural Cloning, with no DAgger-style (Ross, Gordon and Bagnell, 2011) online correction against the learner's own visited states, accepts a known, named risk of compounding error under distributional shift, in exchange for a training procedure that does not require an interactive expert available during training — appropriate given that the "expert" here is 24 already-recorded, non-expert demonstration sessions rather than a supervisor who could be queried live. Torabi, Warnell and Stone (2018) and Codevilla *et al.* (2019) independently document the specific failure mode this project's own v2 study (Section 3.5) diagnosed directly: naive Behavioural Cloning degrades under sparse state-space coverage, precisely the flaw found in the original v1 approach-event labelling, which sampled almost exclusively from the terminal phase of each demonstrated approach.

The re-specification of the O-space validation criterion from an unmeasurable metric distance to a measurable proportion of person-width (Section 3.3.1) preserves the original criterion's intent and order of magnitude, but is a genuine adaptation to a data limitation discovered during the project rather than a pre-registered method, and is disclosed as such. The validation itself was subsequently **completed and failed** against this re-specified criterion (11/30, 36.7%, against a 70% target) — a result reported honestly here rather than omitted, consistent with the project's stated practice of disclosing negative findings, and one that should be read alongside the fact that the live simulation results depending on group geometry are scored against true ground truth rather than this offline estimate, so the failure is bounded to what it actually measures (Section 3.3.1).

The `group_scale_norm` feature used by the Behavioural Cloning model was defined during offline training as a pixel-based apparent-size proxy for distance (mean bounding-box width over image width), and the live policy node must *reconstruct* an equivalent value from true simulated depth at deployment time, since no such pixel proxy exists once real metric distance is available. This is a genuine train/deployment domain shift, reported as a limitation rather than resolved, since retraining the model directly on metric features (attempted in a different form during the v2 study's group-frame reformulation, Section 3.5.4, and found to perform worse for an independently diagnosed reason) was judged out of scope for a second full retraining pass given the project's remaining time.

A further, previously undisclosed intervention is named explicitly here for completeness, since it materially affects how the learned policies' results should be interpreted. **`approach_guard`** is an active runtime safeguard in the deployed BC and MLP policy nodes: when a raw model prediction would move the robot *away* from the detected group entirely — an implausible output given the situation — the prediction's direction is re-projected onto the robot-group axis while preserving the model's own predicted standoff distance. **The learned policies actually evaluated in Chapter 5 are therefore hybrid systems: a learned standoff distance, combined with a geometrically-corrected direction of travel**, not a purely end-to-end learned policy. This was applied identically to both learned policies, so the comparison between BC-Random-Forest and BC-MLP in Chapter 5 remains fair between the two, but the comparison of either learned policy against the rule-based baseline is, to this extent, a comparison of a partially-hybridised learned system against a fully hand-engineered one rather than a "pure learning versus pure rules" comparison. This is disclosed as a limitation because Section 3.7.6's own finding — that the learned models are strong on position but weak on orientation — is exactly the kind of result `approach_guard`'s direction correction could, in principle, be partly responsible for, and a reader should be able to weigh that when interpreting Chapter 5's headline claims.

Finally, the detector-substitution finding of Section 3.2 converts what began as a pragmatic engineering choice into a measured, generalisable methodological result: substituting a lightweight single-stage detector for a large vision-language grounding model, when the task is closed-loop robot control rather than one-shot image description, is not merely "faster" but potentially the difference between a policy that can act on live information and one that cannot — a distinction this project is, to its knowledge, one of relatively few closed-loop robotics studies to have measured directly for a vision-language grounding model of this kind, rather than assumed from published single-image benchmarks alone.

### 3.10 Consolidated list of figures and diagrams to add

The following is every figure or diagram flagged in-line above, gathered here in one place as requested, in the order they occur through the chapter:

1. **[FIGURE 3.0-A]** — Redrawn system pipeline as a proper labelled block diagram (recorded demonstrations → extraction/labelling → offline training/evaluation → Gazebo simulation → live perception → group/O-space estimate → policy → Nav2 → metrics), replacing the ASCII version in Section 3.0.
2. **[FIGURE 3.2-A]** — Inference-time distribution for YOLOv8n vs LocateAnything-3B (bar or box-plot, log-scaled y-axis), illustrating the ~1,700× throughput gap from Section 3.2.3.
3. **[FIGURE 3.2-B]** — An annotated example frame showing YOLOv8n's live bounding-box detection overlay as seen in RViz, for Section 3.2.
4. **[FIGURE 3.3-A]** — A small montage of 2–3 of the 30 O-space-labelling frames, each showing the human-clicked point, the pipeline's automatic centroid, and the resulting pixel error — one clearly within tolerance, one clearly outside it.
5. **[FIGURE 3.3-B]** — A histogram or bar chart of the 30 per-frame O-space errors (in person-widths), with the 0.5-person-width tolerance line marked, for Section 3.3.1.
6. **[FIGURE 3.4-A]** — A grouped bar chart of Table 3.4's six offline policies × two metrics (position, orientation), for Section 3.4.
7. **[FIGURE 3.5-A]** — Position error for all four policies (naive, rule, RF, MLP) shown side by side for "terminal-adjustment test rows" (v1, Panel A) vs "genuine-approach test rows" (v2, Panel B), for Section 3.5.3.
8. **[FIGURE 3.5-B]** — A two-line plot of position error against training-event count (120 → 578) from the data-scaling sweep in Section 3.5.4.
9. **[FIGURE 3.6-A]** — The ROS 2 node/package diagram, redrawn as labelled boxes with topic names on the connecting arrows (`/group_centroid`, `/approach/start`, `/approach/complete`, `/cmd_vel`), for Section 3.6.1.
10. **[FIGURE 3.6-B]** — A top-down, to-scale floor plan of `restaurant_testing.world`: walls, furniture, the three group centres with their O-space circles, **and the patrol route overlaid** (this absorbs [FIGURE 3.6-D] below into one combined figure) — arguably the single most useful figure in the chapter, for Section 3.6.2 / 3.6.6.
11. **[FIGURE 3.6-C]** — A before/after diagram or annotated screenshot pair showing the robot's Gazebo position diverging from its RViz-estimated position under the old static map→odom transform, for Section 3.6.5(b)/3.6.3.
12. **[FIGURE 3.6-D]** — *(merged into 3.6-B above — the patrol route waypoints plotted on the same floor-plan figure rather than as a separate diagram.)*
13. **[FIGURE 3.7-A]** — A grouped bar chart (with spread shown) of the six F-formation slot-accuracy position-error figures (three policies × two detectors), for Section 3.7.6.
14. **[FIGURE 3.7-B]** — A schematic of one group's F-formation slots (bearings, kept gaps ≥45°, resulting slot positions), using Group C (five people, the "loose ring") as the illustrative case, for Section 3.7.6.

**Fourteen figures/diagrams in total** (thirteen distinct, since 3.6-D merges into 3.6-B). Recommended priority if not all fourteen can be produced before submission: 3.6-B (world + patrol route) and 3.0-A (pipeline) first, since almost every other section refers back to the geometry or the pipeline stage they establish; 3.2-A and 3.7-A next, since both carry a headline quantitative finding; the remainder as time permits.

---

## References

*Every source cited in-text anywhere in this chapter (Sections 3.0–3.9) is listed below, alphabetically, in full Harvard format, exactly as a dissertation-wide reference list would require it — not merely named in a sentence. Twenty-two of these twenty-four sources are shared with, and were independently verified for, the rebuilt `Literature_Review_Draft.md`; the two new ones required specifically for the [AP4] detector-alternatives discussion (Carion et al., 2020 and Ren et al., 2015) are marked **NEW** and were verified live this session. Each entry carries a short bracketed note on why it was chosen and exactly where in this chapter it is used, in the same style as the Literature Review's own reference list, so the two lists can be merged into one dissertation-wide list later without losing this annotation trail. This list was built by grep-checking every citation actually appears in-text in this file, rather than assembled from memory — no source below is listed unless it is truly cited above, and nothing cited above is missing from this list.*

Argall, B.D., Chernova, S., Veloso, M. and Browning, B. (2009) 'A survey of robot learning from demonstration', *Robotics and Autonomous Systems*, 57(5), pp. 469–483.
*[Chosen to frame the Behavioural Cloning model within the broader robot learning-from-demonstration paradigm it belongs to, rather than presenting it as a bespoke method with no lineage. Used in Section 3.4, in the opening framing paragraph before label construction is described.]*

Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.
*[Chosen because this project trains a `RandomForestRegressor` as its lower-variance alternative model family, and needed the actual algorithm's own origin cited rather than left implicit. Used in Section 3.4 (model architectures) and Section 3.5 (the 360-model hyperparameter search, where Random Forest is one of three families compared).]*

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. and Zagoruyko, S. (2020) 'End-to-end object detection with transformers', in *Computer Vision – ECCV 2020* (Lecture Notes in Computer Science). Cham: Springer, pp. 213–229. **[NEW]**
*[Chosen to represent the transformer-based detector family in the [AP4] survey of alternatives to YOLO — DETR removes the hand-designed anchors and non-maximum suppression both YOLO and two-stage detectors rely on, at the cost of the same heavier per-frame inference character this project measured directly for LocateAnything-3B. Used in Section 3.2.1 and again in the closing justification in 3.2.3, as one of three alternative detector families surveyed before defending the eventual choice of YOLOv8n.]*

Codevilla, F., Santana, E., López, A.M. and Gaidon, A. (2019) 'Exploring the limitations of behavior cloning for autonomous driving', in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019)*. Seoul: IEEE, pp. 9329–9338.
*[Chosen because it is the closest external analogue to this project's own v2-study finding: an independent group, in a different domain, showing that Behavioural Cloning's generalisation failures are usually a dataset problem rather than a model-architecture problem. Used in Section 3.9, corroborating the label-ambiguity-floor diagnosis from Section 3.5.4 from outside the project.]*

Fisher, R.A. (1922) 'On the interpretation of χ² from contingency tables, and the calculation of P', *Journal of the Royal Statistical Society*, 85(1), pp. 87–94.
*[Chosen as the origin of the two-sided Fisher exact test used for every binary-metric comparison in the 60-trial experiment. Used in Section 3.7.4 (statistical treatment), grounding the significance tests reported throughout that section and referenced again in Chapter 5.]*

Friedman, J.H. (2001) 'Greedy function approximation: a gradient boosting machine', *Annals of Statistics*, 29(5), pp. 1189–1232.
*[Chosen as the foundational paper behind gradient-boosted tree ensembles, one of the three model families compared in the rejected "v2" ablation and the 360-model search. Used in Section 3.5.1 and 3.5.5, supporting the argument that independently-designed algorithms converging on the same error ceiling is meaningful evidence the limitation lies in the data.]*

Hall, E.T. (1966) *The Hidden Dimension*. Garden City, NY: Doubleday.
*[Chosen as the primary source for proxemic zones, cited at source rather than only through a secondary survey. Used in Section 3.6.4 (justifying the rule-based policy's 1.2 m standoff and 0.7 m per-person clearance) and Section 3.7.2 (the personal-zone framing of the minimum-distance-to-person metric).]*

Helbing, D. and Molnár, P. (1995) 'Social force model for pedestrian dynamics', *Physical Review E*, 51(5), pp. 4282–4286.
*[Chosen as the physics-based formalisation of the same "keep a comfortable distance" principle Hall (1966) states qualitatively, giving the rule-based baseline's standoff distance an additional, independent grounding. Used in Section 3.6.4.]*

Hoffman, G. and Zhao, X. (2020) 'A primer for conducting experiments in human–robot interaction', *ACM Transactions on Human-Robot Interaction*, 10(1), pp. 1–31.
*[Chosen because this project runs a genuinely small-sample (n=10 per condition) controlled comparison and needed its statistical discipline — reporting effect sizes and raw counts rather than over-claiming significance — grounded in the field's own standard reference. Used in Section 3.7.4, justifying why several visually large differences are reported as effect sizes rather than claimed as significant.]*

Jocher, G., Chaurasia, A. and Qiu, J. (2023) *Ultralytics YOLOv8* (Version 8.0.0) [Software]. GitHub. Available at: https://github.com/ultralytics/ultralytics (Accessed: 25 August 2026).
*[Chosen as the correct primary citation for the actual `ultralytics` package this project imports and pins, rather than relying solely on a secondary architectural analysis for software with no formal paper of its own. Used in Section 3.2.1 and 3.2.2, alongside Yaseen (2024), to separate "the source" from "an analysis of the source."]*

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017) 'LightGBM: a highly efficient gradient boosting decision tree', in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. Long Beach: NeurIPS, pp. 3146–3154.
*[Chosen because scikit-learn's `HistGradientBoostingRegressor`, used in the v2 ablation study, directly implements LightGBM's histogram-binning strategy. Used in Section 3.5.1, as the specific lineage between Friedman's (2001) general principle and the actual class this project instantiates.]*

Kendon, A. (1990) *Conducting Interaction: Patterns of Behavior in Focused Encounters*. Cambridge: Cambridge University Press.
*[Chosen as the single most load-bearing citation in this chapter: the O-space definition, the P-space vocabulary, and the F-formation slot-derivation method used to score all 60 live trials are Kendon's theory operationalised directly, not merely referenced as background. Used in Section 3.3 (O-space estimation), Section 3.6.4 (gap-based approach selection), Section 3.7.2 (the O-space intrusion metric definition) and Section 3.7.6 (the P-space slot accuracy method).]*

Kruse, T., Pandey, A.K., Alami, R. and Kirsch, A. (2013) 'Human-aware robot navigation: A survey', *Robotics and Autonomous Systems*, 61(12), pp. 1726–1743.
*[Chosen as the field-defining survey used to frame the "every metric here is a geometric proxy" limitation as an accepted, field-wide simplification rather than a flaw unique to this project. Used in Section 3.0.]*

Lu, D.V., Hershberger, D. and Smart, W.D. (2014) 'Layered costmaps for context-sensitive navigation', in *2014 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. Chicago: IEEE, pp. 709–715.
*[Chosen because this is the literal costmap-layer architecture underneath Nav2 (Macenski et al., 2020), the stack every trial in this project runs through, not background reading. Used in Section 3.6.1.]*

Macenski, S., Martín, F., White, R. and Clavero, J.G. (2020) 'The Marathon 2: A navigation system', in *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. Las Vegas: IEEE. Available at: https://arxiv.org/abs/2003.00368 (Accessed: 25 August 2026).
*[Chosen as the actual system paper behind Nav2, cited as primary platform documentation rather than only a vendor page, since Nav2 executed every one of the 60 live trials reported in this chapter. Used in Section 3.6.1.]*

Pagès, J., Marchionni, L. and Ferro, F. (2016) 'TIAGo: the modular robot that adapts to different research needs', in *International Workshop on Robot Modularity, IROS 2016*. Daejeon, South Korea: CLAWAR Association, 10 October.
*[Chosen as the primary system paper for TIAGo, the platform this project's entire simulated system is built around. Used in Section 3.6.1.]*

Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. (2016) 'You only look once: Unified, real-time object detection', in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. Las Vegas: IEEE, pp. 779–788.
*[Chosen as the architectural origin of the single-pass detection paradigm YOLOv8n (this project's primary detector) descends from, cited to complete the [AP4] survey of alternatives alongside the two-stage and transformer families. Used in Section 3.2.1.]*

Ren, S., He, K., Girshick, R. and Sun, J. (2015) 'Faster R-CNN: towards real-time object detection with region proposal networks', in *Advances in Neural Information Processing Systems 28 (NeurIPS 2015)*. Montreal: NeurIPS, pp. 91–99. **[NEW]**
*[Chosen to represent the two-stage detector family in the [AP4] survey of alternatives to YOLO — the region-proposal-then-classify architecture that YOLO's single-stage design was created to be a faster alternative to. Used in Section 3.2.1 and again in the closing justification in 3.2.3.]*

Ríos-Martínez, J., Spalanzani, A. and Laugier, C. (2015) 'From proxemics theory to socially-aware navigation: A survey', *International Journal of Social Robotics*, 7(2), pp. 137–153.
*[Chosen for its explicit, unresolved flagging of the human-to-robot proxemics transfer assumption, which this project also makes without independently testing. Used in Section 3.0, alongside Kruse et al. (2013), to frame the geometric-proxy limitation.]*

Ross, S., Gordon, G. and Bagnell, D. (2011) 'A reduction of imitation learning and structured prediction to no-regret online learning', in *Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS)*. PMLR 15, pp. 627–635.
*[Chosen as the paper that formalises Behavioural Cloning's compounding-error problem and proposes DAgger as a fix this project deliberately does not implement. Used in Section 3.9, as a stated, named limitation of the project's offline-only training procedure.]*

Torabi, F., Warnell, G. and Stone, P. (2018) 'Behavioral cloning from observation', in *Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI 2018)*. Stockholm: IJCAI, pp. 4950–4957.
*[Chosen for its account of why naive Behavioural Cloning struggles under sparse state-space coverage — precisely the flaw the v2 study diagnosed in the original v1 approach-event labelling, which sampled almost exclusively from the terminal phase of each demonstrated approach. Used in Section 3.9.]*

Vascon, S., Mequanint, E.Z., Cristani, M., Hung, H., Pelillo, M. and Murino, V. (2016) 'Detecting conversational groups in images and sequences: A robust game-theoretic approach', *Computer Vision and Image Understanding*, 143, pp. 11–24.
*[Chosen as the single most methodologically important citation for this project's O-space estimation: it independently establishes that F-formation structure can be recovered from position alone, without orientation — the exact justification relied on for the mutual-facing fallback, forced by PLUS-HRI's complete lack of orientation ground truth. Used in Section 3.3.]*

Wang, S., Liu, S., Kuang, Y., Wei, X., Liu, Y., Li, Z., Man, Y., Chen, G., Tao, A., Liu, G., Kautz, J., Zhang, L. and Yu, Z. (2026) 'LocateAnything: Fast and high-quality vision-language grounding with parallel box decoding'. Available at: https://arxiv.org/abs/2605.27365 (Accessed: 25 August 2026).
*[Chosen because this project performs a direct, primary head-to-head empirical comparison against this exact model — both the early 30-frame offline comparison and the final 60-trial live benchmark. Used throughout Section 3.2, as the subject of the [AP4] substitution decision.]*

Yaseen, M. (2024) 'What is YOLOv8: An in-depth exploration of the internal features of the next-generation object detector'. Available at: https://arxiv.org/abs/2408.15857 (Accessed: 25 August 2026).
*[Chosen as the closest available substitute for a peer-reviewed YOLOv8 methods paper, since Ultralytics never published one. Used in Section 3.2.1 and 3.2.2, alongside the official software citation (Jocher, Chaurasia and Qiu, 2023).]*

---

**Citation consistency check against the Literature Review.** `Repiso, Garrell and Sanfeliu (2020)` — the one citation whose year was corrected during the Literature Review rebuild (from an incorrect 2019 to the confirmed 2020 print year) — **is not cited anywhere in this Methodology chapter**, so no separate correction is needed here; it belongs to Section 2.4 of the Literature Review only, contrasting a continuously-updated approach against this project's own static approach-pose formulation. All 24 sources above are dated identically to their entries in `Literature_Review_Draft.md` where they overlap (22 of the 24), so the two chapters' reference lists can be merged directly into one dissertation-wide list with no year or detail conflicts.
