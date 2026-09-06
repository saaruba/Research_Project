# Cuttable Content Checklist

**Built 27 August 2026, against your current document** (the one you uploaded as `CMP9140_Dissertation_1st_Draft.docx` — it's actually your current ~22,000-word merged draft, same content as the `CMP9140_Dissertation_2nd_Draft.pdf` you sent earlier, just re-paginated to 105 pages after I filled in the real Table of Contents / List of Figures / List of Tables).

**About the highlighting**: I did also mark these same 9 passages in yellow directly inside `CMP9140_Dissertation_Highlighted_and_Indexed.docx` (the Word file, not a PDF) — the highlight only shows up if you open that `.docx` in Word or LibreOffice; it won't appear if you're viewing a PDF export of it. This file gives you the same 9 spots as plain text instead, organised by heading, in case that's easier to work from or the highlighting didn't come through on your end.

**Scope, and why**: every item below comes from Chapter 3 (Methodology, 15% of your CRG mark) or Chapter 4 (Implementation, not separately weighted). I deliberately did not look for cuts in the Introduction, Literature Review, Results & Discussion, or Conclusion — those four chapters carry the other 85% of your mark (Intro+LitReview alone are 55%), so trimming there risks far more than it saves. If you want me to search those chapters too, say so, but I'd treat that as a last resort.

**Honesty check**: these 9 passages total **~370 words**. That is well short of the 5,500 words you want to remove. These are only the passages I'm confident are pure duplication or filler — safe to delete outright, no rewriting needed, no risk to content. Getting further toward 5,500 needs either a real tightening/rewrite pass over Methodology and Implementation's denser paragraphs (I've listed candidates for that at the bottom, separately, since those need rewording rather than straight deletion), or confirming whether your module counts references toward the word limit — worth checking before cutting further, since it could close a big chunk of the gap on its own.

---

## Chapter 4 — Implementation

### 4.1.1 Development Environment and Dependencies (page 64)

> The above-mentioned software development setup mainly focuses on the implementation rather than the methodology of the project. The purpose of this setup was to ensure that the same software configuration used for training and evaluating the models can be reproduced again after rebuilding the development environment

**Why:** generic closing summary — restates the section's own point without adding new information. ~46 words.

---

### 4.2 ROS 2 Package and System Architecture (page 63)

> The earlier implementation draft confirms this separation between the packages and the seven main runtime nodes which are used in the system.

**Why:** meta-commentary about an earlier draft of your own document — not content, just a note-to-self that made it into the final text. ~22 words.

---

### 3.2.2 Detector Validation (page 39) — duplicate table caption

> **Table 3.2 — Detector comparison used to justify the deployed detector**:

**Why:** this exact caption line appears twice in a row, immediately before the real one ("Table 3.2 — Detector comparison used to justify the deployed detector: (Figure 3.2 → this Rviz detection screenshot)"). Delete the first, plain copy; keep the second one (though the "(Figure 3.2 → this Rviz detection screenshot)" note attached to it looks like a leftover editing reminder too — worth deciding whether to insert that screenshot as Figure 3.2 or drop the note). ~11 words removed outright.

---

### 4.3.2 Person and Group Processing (page 67–68)

> Group construction is performed after detecting the persons. The person centres are compared by using the normalised distance formulation explained in Chapter 3, after that the connected persons are assigned into the same candidate group. The group-processing stage stores the membership indices, detected group size, centroid and bounding information which are required for the further processing stages.

**Why:** restates the group-detection method already fully explained in Section 3.3, just in different words. ~57 words.

> The offline O-space estimation uses the detected group centroid because reliable orientation information of the participants are not available in the source data. This implementation follows the position-based approximation explained in Section 3.3. The limitations and manual validation of this approximation have already been discussed in that section and therefore, it is not repeated again in this chapter.

**Why:** the last sentence says "it is not repeated again" — but the paragraph itself re-explains the mechanism it claims not to repeat. ~58 words.

---

### 4.6.2 Rule-Based Approach Policy (page 72)

> The final implementation includes the practical safeguards which were described in Chapter 3, including standoff distance, clearance from individual people, correction for the body radius, gap selection, goal throttling and recovery from prolonged stalls.

**Why:** this is close to a word-for-word repeat of the safeguards list already given in Section 3.6.3 ("goal throttling, maintaining clearance from individual people, considering robot and person body radii, selecting a clear angular gap..."). ~34 words.

---

### 4.8 Gazebo Environment Implementation (page 74) — the strongest catch

> The simulated human models were not completely created from scratch. The human actor models and animations were selected from LIRS-HMLG, which is the Laboratory of Intelligent Robotic Systems – Human Models Library for Gazebo. It is an open-source human-model library developed for Gazebo-based HRI simulation. The library contains different human models and animations including walking, running, sitting and talking, which makes it suitable for creating the social evaluation environment used in this project (Tukhtamanov et al., 2022).

**Why:** this paragraph is almost identical to one already in Section 3.6.2 ("The human models and animations used in the Gazebo environment were selected from the LIRS-HMLG (Laboratory of Intelligent Robotic Systems – Human Models Library for Gazebo). Tukhtamanov et al. (2022) developed LIRS-HMLG to provide more varied and realistic human models for Gazebo. The library contains different types of human appearances and animations including walking, running, talking and sitting..."). Same source, same description, same citation, two chapters. This is your single biggest safe cut. ~76 words.

---

### 4.9 Metrics Recording and Experimental Automation (page 76)

> This experimental infrastructure allowed the final evaluation to be carried out systematically across the three different policies and two detector conditions rather than running each experiment as separately configured demonstrations.

**Why:** closing recap sentence that doesn't add anything beyond what the section already said. ~30 words.

---

### 4.10 Testing and Implementation Reliability (page 77)

> The above testing methods support the end-to-end validation of the complete perception–policy–navigation system. However, the implementation mainly focused on validating the complete robotic system and conventional isolated unit-test coverage still remains as an area that can be improved.

**Why:** restates the section's own opening point and its "First/Second/Third" structure without adding anything new. ~38 words.

---

**Running total: 9 passages, ~370 words.**

---

## Beyond these 9: paragraphs worth tightening (not listed above, because these need rewording, not straight deletion)

I didn't quote exact cuts here since shortening these means rewriting rather than deleting a self-contained passage — but if you want to close more of the 5,500-word gap yourself, these are the highest-value places to compress:

- **Section 4.10** (page 76–77): the three "First,... Second,... Third,..." testing paragraphs could likely be combined into one tighter paragraph — each currently runs 4–6 sentences describing one testing method; a marker mainly needs to know the three methods existed and what they checked.
- **Section 4.5** (page 70–71, Parameter Tuning and Model Selection): the grid-search configuration list (n_estimators/max_depth/min_samples_leaf combinations) is precise and worth keeping, but the surrounding prose explaining it could be tightened by roughly a third without losing the numbers.
- **Section 3.6** as a whole (pages 47–58, Simulation Environment and System Integration): this is your longest subsection by far and has some genuine narrative repetition between 3.6.1/3.6.2/3.6.3 in how it reintroduces the live system before each new topic — a general 15–20% density pass here (tightening sentences rather than cutting content) could plausibly recover 400–600 words on its own.
- **Section 4.4.1** (page 68, Approach-Event and Target Generation): the paragraph on the v2 investigation ("The later v2 investigation introduced another segmentation procedure...") already says it won't repeat Section 3.5's findings, then spends a full sentence summarising them anyway — could be trimmed to one clause.

Combined with the 370 words above, a real pass over these four spots could plausibly get you into the 1,200–1,800 word range — still short of 5,500, which is why I'd flag the references-word-count question to your module leader or in the handbook before cutting further: if references are excluded from the 12,000–15,000 cap, your real gap to close is smaller than it currently looks.
