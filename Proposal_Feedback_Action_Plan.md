# Proposal Feedback — Action Plan

Drafted responses to each point of supervisor feedback, ready to adapt into your dissertation write-up (Introduction, Literature Review, Methodology chapters). Nothing here changes your project's direction — it's all refinement.

---

## 1. Introduction — missing citation

Add a citation after: *"Many social navigation methods use fixed rules, personal-distance values, or social cost maps."*

Two good, verified options:

- **Simplest fix:** cite Kruse et al. (2013) again here — it's already in your reference list, and its survey explicitly covers rule-based and cost-map approaches, so reusing it is legitimate.
- **Stronger, more specific fix (recommended):** add two focused citations:
  - Lu, D.V., Hershberger, D. and Smart, W.D. (2014) 'Layered costmaps for context-sensitive navigation', *2014 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 709–715. — the actual paper behind ROS's social costmap layers.
  - Helbing, D. and Molnár, P. (1995) 'Social force model for pedestrian dynamics', *Physical Review E*, 51, pp. 4282–4286. — the foundational social-force / fixed-rule model this sentence is really describing.

Revised sentence: *"Many social navigation methods use fixed rules, personal-distance values, or social cost maps (Helbing and Molnár, 1995; Lu, Hershberger and Smart, 2014)."*

---

## 2. Aims and Objectives — add quantitative success criteria

Rewrite each objective to state a measurable pass/fail condition. Proposed versions (treat the numeric thresholds as a first working target — sanity-check them once you have pilot results from sessions 1 and 3, and adjust with justification if needed rather than treating them as fixed):

1. **Literature review:** *"Review at least 18–20 peer-reviewed sources across social navigation, F-formations, Learning from Demonstration, and Behavioural Cloning, producing a literature matrix that directly informs the system requirements and evaluation metrics used in Objectives 2–4."*
2. **People localisation:** *"Implement a people-localisation module and validate it against the ground-truth face annotations available for sessions 1 and 3 of the dataset, targeting a person-detection recall of at least 80% on those sessions before applying it to the remaining sessions."* (This is genuinely measurable — you have real ground truth for exactly this check.)
3. **Group/O-space detection:** *"Detect small groups and estimate the O-space, validated against a manually hand-labelled subset of at least 30 frames, targeting an O-space-centre estimate within 0.3m of the manual annotation for at least 70% of labelled frames."* (No ground truth exists in the dataset itself, so you create a small labelled validation set yourself — state this plainly as your validation method.)
4. **Behavioural Cloning vs baseline:** *"Train and evaluate a Behavioural Cloning model that achieves a mean approach-position error below 0.4m and mean approach-orientation error below 20° on unseen test-session group configurations, and compare this against a rule-based baseline on the same metrics."*

---

## 3. Literature Survey — add a limitation per paper + closing gap statement

Add one sentence per paper identifying a concrete limitation your project addresses (the professor's own example, for reference: *"Gao et al. (2019) require a manually designed reward function, while your use of Behavioural Cloning avoids this."*). Drafted sentences for the rest:

- **Kruse et al. (2013):** provides general human-aware navigation principles but no specific method for selecting a robot's stopping position near a conversational group — this project addresses that gap directly via F-formation-based approach-pose selection.
- **Setti et al. (2015):** F-formation detection depends on reliable position/orientation estimates, which in real robot footage can be unreliable due to occlusion or missed detections — this project directly confronts that limitation by validating its own perception pipeline against ground-truth annotations (sessions 1 and 3) before trusting it elsewhere, rather than assuming clean input.
- **Argall et al. (2009):** as a foundational survey, it does not address the correspondence problem for a specific task like group approach, and notes that LfD performance depends heavily on demonstration quality and coverage — this project directly tests that dependency by evaluating predicted vs. demonstrated approach poses on held-out sessions rather than assuming generalisation.
- **Faris et al. (2025):** learns navigation-recovery behaviour from non-expert demonstrations using only LiDAR and velocity data, without addressing group detection, conversational structure, or person perception — this project extends the same non-expert-demonstration learning philosophy to the group-approach problem, adding the human-perception and social-structure components entirely absent from that work.
- **Yang et al. (2021):** provides a valuable group-approach dataset, but a different one from the dataset used here — the PLUS-HRI dataset in this project is richer in modality (video, gaze, audio) but, unlike Yang et al.'s dataset, lacks explicit group/O-space/approach-pose annotations, which is precisely why this project must derive those labels via its own perception and clustering pipeline (Phase B/C) rather than relying on provided ground truth.
- **Wang et al. (2026), LocateAnything-3B:** trained for general-purpose open-vocabulary grounding, not specifically validated on cluttered, dimly lit, robot-mounted-camera indoor HRI footage — this project's validation step (comparing detections against the two sessions with ground-truth face annotations) directly tests this domain gap before trusting the model on the rest of the dataset.

**Closing gap statement** (add as the final paragraph of the literature review):

*"Taken together, the reviewed literature establishes the general principles of human-aware navigation (Kruse et al., 2013), formal tools for representing group structure (Setti et al., 2015), and multiple ways to learn robot behaviour either from reward signals (Gao et al., 2019) or from demonstrations (Argall et al., 2009; Faris et al., 2025), alongside an existing group-approach dataset (Yang et al., 2021) and a candidate perception model (Wang et al., 2026). However, no existing work combines person perception, data-driven group/O-space estimation, and Behavioural Cloning into a single pipeline trained on non-expert demonstrations for the specific problem of group-approach behaviour, nor evaluates such a pipeline against a rule-based baseline using the proxemic and navigation metrics proposed here. This is the specific gap this project addresses."*

---

## 4. Research Methods — BC model architecture paragraph

Add this paragraph (this also reflects real decisions already validated during early pipeline testing — see note at the end):

*"The Behavioural Cloning model takes as input a structured feature vector consisting of: the relative position and bearing of each detected group member with respect to the robot, the estimated group centroid and O-space boundary, the number of people detected, and locally available free-space/LiDAR features describing nearby obstacles. The primary architecture is a multi-layer perceptron with two hidden layers (128 and 64 units, ReLU activations), mapping this feature vector to a predicted approach pose (x, y, yaw) relative to the robot's current frame. Given the limited number of fully human-annotated demonstration sessions, a Random Forest regressor is also evaluated as a lower-variance alternative, and the two are compared on validation-set error before the final model is selected. Hyper-parameters (hidden layer sizes, learning rate, L2 regularisation for the MLP; tree depth and count for the Random Forest) are tuned via grid search on the validation split, with the test split held out entirely until final evaluation to avoid information leakage."*

*Note: this isn't hypothetical — a Random Forest baseline (LiDAR + previous action → velocity command) has already been built and validated on your real data, beating a naive baseline by ~25–30% on held-out sessions. That gives you a genuine, already-tested justification for including it as a documented alternative, not just a theoretical mention.*

---

## 5. Project Plan and Risk Analysis

**Gantt chart:** your underlying task table already includes Deliverable/Milestone and Success Measure columns per row — good, this substantively satisfies the request. Just double-check the visual Gantt chart image itself (not only the table) shows the M1–M6 milestone points clearly marked (e.g., diamond markers at the relevant weeks), since that's likely what looked like "just activities" at a glance.

**Risk-priority justification** (for distinction level) — add this paragraph:

*"Given the time available, mitigation effort is prioritised as follows. Highest priority: the risk that the dataset lacks required information, and the risk of scope creep — both directly threaten Objectives 2–4. The dataset risk has already partly materialised: direct inspection confirmed that only 2 of 24 sessions contain the face-landmark annotations originally expected to support group/O-space estimation, which is mitigated by building a custom person-detection step to recover human-position data for the remaining sessions. Scope creep is mitigated by strictly prioritising the core dataset-to-BC-to-simulation pipeline over optional gaze/language extensions. Second priority: perception accuracy (LocateAnything-3B/detector performance) and group/F-formation estimation accuracy, both mitigated by validating any detector's output against the two sessions with ground-truth annotations before trusting it elsewhere, and by starting with simple group configurations. Lower priority: BC generalisation to unseen layouts and predicted-pose validity, addressed through session-level train/validation/test splitting and pose-reachability checks against the ROS planner, since these are only meaningfully assessable once the higher-priority perception and scope risks are resolved. The evaluation-validity risk (metrics as proxies for real social comfort) is treated as an accepted, explicitly stated limitation rather than something mitigated away, since resolving it fully would require real-world testing outside this project's ethical scope."*

---

## 6. References & Ethics

No action needed beyond continued discipline: keep Harvard style consistent, include page numbers for any direct quotations in the dissertation. Ethics section is already correctly handled — only revisit if scope changes to include real robot testing or participant feedback.

---

## Decisions locked in from this feedback (affects the technical build, not just the writing)

- **BC model:** MLP (2 hidden layers, 128/64 units) as primary, Random Forest as documented comparison — matches what's already been tested.
- **Feature set for the richer model:** group-relative position/bearing, group centroid, O-space boundary, person count, LiDAR free-space features, previous action.
- **Target thresholds to validate against:** person-detection recall ≥80% (sessions 1/3 ground truth), O-space estimate within 0.3m for ≥70% of a hand-labelled validation set, approach-pose error <0.4m position / <20° orientation on held-out sessions.
