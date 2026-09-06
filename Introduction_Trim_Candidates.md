# Introduction — Trim Candidates

**Read this first.** What you pasted is Chapter 1 (Introduction), not Implementation. I flagged this before you answered, and your reply didn't say "yes, cut the Introduction hard anyway" — so I'm not attempting the 50% cut here. Introduction + Literature Review together carry 55% of your whole module mark (the CRG's LO3, by far the biggest weighting), and this chapter is one of the two places a marker forms their first impression of "does this project have a comprehensive, well-justified background." Removing half of it — including any of the literature grounding, the aims/objectives, or the findings preview — would cost you more than it saves. Methodology and Implementation together are only 15%; that's genuinely where a 50% cut is a much safer trade.

What I did instead: read the whole thing closely and pulled out everything that's genuinely redundant, padded, or restates something said two sentences earlier — the kind of thing that's safe to delete because nothing is lost. That comes to about **280 words out of ~2,600** (roughly 11%), not 50%. I'd rather tell you that honestly than manufacture a bigger number by cutting content the CRG is explicitly rewarding.

If you still want a much larger cut specifically from this chapter, say so explicitly and I'll do it — but I'd be cutting real lit-review and aims content at that point, not padding, and I want that to be your informed choice, not something I do by default.

---

## Opening (before 1.1)

> The objective of this study is to investigate whether learning from human demonstrations can provide a socially appropriate approach behaviour for a service robot when interacting with conversational groups.

**Why:** this is the paragraph's closing sentence, and it says almost exactly what the paragraph's opening sentence already said ("This dissertation investigates whether socially appropriate group approach behaviour can be learned from non-expert human demonstrations..."). One bookend sentence can go. ~30 words.

*(Separately — not a content cut, a wording fix: "Lets Picture the service robots are increasingly used..." reads like a leftover instruction/draft fragment rather than a finished sentence. Worth rewording regardless of any word-count goal.)*

---

## 1.1 Motivation and Problem Statement — closing of the gap paragraph

> Therefore, this dissertation focuses on detecting people and conversational groups from the demonstration data, estimating the social geometry of the group, and learning an appropriate stopping pose from teleoperation behaviour. The learned approach is then compared directly with a geometric rule using both offline evaluation and live simulation. The main focus of this project is specifically on the final stopping pose, including where the robot should stop and the direction it should face, rather than learning the complete end to end process of detecting a distant group and navigating towards it.

**Why:** this repeats the project's scope almost exactly as already stated in the opening paragraph ("The study mainly focuses on the stopping pose of the robot, which includes where the robot should stop... Behavioural Cloning is used... evaluated against a geometric rule-based baseline using held-out demonstration data and a live Gazebo simulation"). You've now said "compares learned vs. geometric rule using offline + live evaluation, focused on stopping pose" twice in one chapter. ~65 words.

---

## 1.2 Aims and Objectives — closing paragraph

> The abovementioned objectives maintain the four objective structure which was established in the original project plan. However, the objectives were further defined in order to make the outcomes measurable and suitable for evaluating the overall performance of the project within the dissertation.

**Why:** this is a note about your own drafting process (comparing this version to your original proposal), not findings or methodology — a marker doesn't need to know the objectives were "further defined" from an earlier plan. ~35 words.

---

## 1.3 Why This Problem, and Why This Dataset — dataset background

> From the direct inspection of the dataset, it was found that the sessions were collected from two different recording generations. Sessions 1 and 3 contain per frame facial-landmark annotations, whereas the remaining 22 sessions do not contain the same person-position ground truth. The available gaze information was also not suitable for estimating reliable facing direction of each person.

**Why:** this level of dataset detail is repeated almost identically in Methodology Section 3.1 ("Sessions 1 and 3 were collected using an earlier recording configuration and these sessions contain per-frame facial-landmark annotations. The remaining 22 sessions were recorded using a segmented recording format..."). The Introduction only needs to justify *why* you chose this dataset despite its limitations — the *what exactly is different between the sessions* belongs in Methodology, where it already lives. ~55 words.

---

## 1.6 Alignment with the MSc Robotics and AI Programme — repeated conclusion

> The above mentioned components indicate that the project combines machine learning, autonomous robotics, software integration and human centred evaluation. Therefore, the project is closely related with the interdisciplinary nature of an MSc Robotics and AI programme, because it includes both the development of an intelligent learning system and its implementation within a robotic environment.

**Why:** this is the section's third and final paragraph, and it draws almost the same conclusion as the first paragraph's closing sentence ("These different components shows that the project combines both machine learning and robotics methods throughout the development of the system."). You state "this project combines ML and robotics" as a conclusion twice in the same short section. Keep one, cut the other. ~55 words.

---

**Running total: ~240 words** (the fifth item above brings it to roughly 240–280 depending on how much of the surrounding sentence you take with it).

## What I deliberately left alone, and why

- **The Kendon/Hall/Kruse/Ríos-Martínez/Argall/Ravichandar/Ross-Gordon-Bagnell literature paragraphs** — this is exactly the "comprehensive review of relevant literature" the CRG rewards at Merit/Distinction. Cutting here would directly work against your mark.
- **Section 1.2's four objectives** — required content, and each one already states a measurable success criterion (80% recall, 70% O-space accuracy, 0.4 m/20° error), which is exactly what "SMART" objectives means to a marker. Don't touch these.
- **Section 1.4 (Summary of the Main Findings)** — this is a strong, unusual thing to include in an Introduction (a results preview), and it's genuinely valuable rather than padding. Leave it fully intact.
- **Section 1.7 (Dissertation Structure)** — standard, expected roadmap content; markers use this for navigation, not a target for cuts.

If you want, I can do the same close pass on the real Implementation chapter next, where a 50% target is actually a reasonable ask given its 15% weighting — just paste that section's text (or point me to it in the file I already have) and I'll go through it the same way.
