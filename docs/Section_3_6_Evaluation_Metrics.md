# 3.6 Evaluation Metrics

*Expanded in response to supervisor comment [AP5]: "Describe in more detail the evaluation metrics you are using."*

---

The evaluation uses two families of metrics, separated by what each one requires in
order to be computed honestly. **Offline metrics** are calculated directly from the
recorded human demonstrations and measure how closely a policy reproduces the pose a
human demonstrator chose. **Simulation metrics** require an executed trajectory and
metric person positions, and measure how the robot actually behaves around people.

The separation is not cosmetic. The PLUS-HRI recordings provide person positions only
as two-dimensional pixel coordinates from uncalibrated video, so any spatial metric
expressed in metres — O-space intrusion, minimum distance to a person, group
cut-through — cannot be derived from the dataset without fabricating a scale factor.
Those metrics are therefore measured exclusively in simulation, where ground-truth
person positions are known in metres from the world definition. Metrics that are
properties of a *path* rather than a *prediction* — path length, navigation time,
collision-free rate, task success — likewise require Nav2 to have driven the robot,
and so are also simulation-only.

## 3.6.1 Offline metrics

Both offline metrics are computed on the three held-out test sessions, which are
excluded from training and validation. Each test row records a moment at which a human
demonstrator completed an approach, so the demonstrated stop pose is known in real
metres and radians from the robot's own odometry. Each policy predicts a pose in the
same robot-relative frame, making the comparison direct.

**Approach-position error** is the Euclidean distance between the predicted stop
position and the demonstrated stop position:

```
e_pos = sqrt( (x_pred − x_demo)² + (y_pred − y_demo)² )      [metres]
```

**Approach-orientation error** is the absolute difference between predicted and
demonstrated heading, wrapped to the interval (−π, π] so that a 350° error is correctly
reported as 10°:

```
e_yaw = | atan2( sin(θ_pred − θ_demo), cos(θ_pred − θ_demo) ) |    [radians → degrees]
```

The acceptance thresholds are inherited from Objective 4 of the project proposal:
**position error below 0.4 m** and **orientation error below 20°**.

For each policy, four figures are reported: mean and median position error, and mean
and median orientation error. Median is reported alongside mean because a small number
of very large errors can dominate a mean and conceal otherwise reasonable typical
behaviour. In addition, the **percentage of test rows meeting each threshold** is
reported, since a policy may have an acceptable mean while satisfying the threshold on
a minority of individual cases.

Two reference policies are evaluated alongside the learned models to make the numbers
interpretable:

- a **naive** policy that always predicts the training-set mean pose, establishing the
  floor below which a model has learned nothing;
- the **rule-based** policy applied offline, which is the comparison point named in
  Objective 4.

## 3.6.2 Simulation metrics

Seven metrics are recorded per trial by `metrics_recorder_node`, which samples the
robot pose at **10 Hz** from the TF transform `map → base_footprint` and writes a JSON
record containing both the summary values and the full trajectory.

All social metrics are scored against the **true person positions** in the world's
`.groundtruth.json` file, never against the robot's own detections. This avoids a
circularity that would otherwise flatter a failing perception system: a policy that
never detected a group would report no intrusion on that group, and would be rewarded
for its own blindness.

| Metric | Type | Definition |
|---|---|---|
| Task success | binary | A socially valid pose was achieved |
| Collision-free | binary | No obstacle within 0.30 m after self-hit calibration |
| O-space intrusion | binary + count | Robot footprint overlapped a group's O-space |
| Minimum distance to any person | metres | Closest approach to any group member |
| Group cut-through events | count | Path segments passing between two group members |
| Path length | metres | Total distance travelled |
| Navigation time | seconds | Trial duration |

### Task success

A trial succeeds if, **at any sample**, the robot simultaneously satisfies a distance
and a heading condition with respect to **any** ground-truth group, and no collision
occurred during the trial:

```
0.5 m ≤ d(robot, group_centre) ≤ 2.0 m        AND
heading error to group centre ≤ 45°           (same sample)
```

Three design decisions in this definition warrant explanation.

**Scored at any point, not at the final pose.** The robot executes a patrol mission and
returns to its start point by design, so its final pose is never beside a group.
Scoring the last sample marked every trial a failure regardless of behaviour — one
trial reached 1.101 m from a person, a textbook social distance, and was recorded as
failed purely because it subsequently drove home. The objective asks whether the robot
*ever achieved* a socially appropriate pose, and the metric now asks exactly that.

**Distance and heading must hold in the same sample.** Scoring the single closest
sample was also unsound, because the nearest point on a path is frequently a tangential
fly-past in which the robot is at the correct distance while travelling across the
group's face. Such passes scored 88–90° heading error and failed three trials in which
the robot had in fact settled at 1.50 m facing the group within one degree. Requiring
the two conditions to co-occur removes both failure modes.

**Scored against every ground-truth group.** An earlier implementation scored against
the last perceived centroid, which at trial end held whatever perception happened to
see most recently — often a false positive against a wall, or a group across the room
the robot never visited. One trial came within 0.51 m of a person and intruded on an
O-space, yet was scored as never having held a valid pose, because it had approached a
real group correctly and was then judged against a phantom. Treating every real group
as a valid target corrects this.

The lower bound of 0.5 m rather than 0.8 m reflects the approach policy's own geometry:
it targets 0.7 m clearance to the nearest *person* while standing in a gap in the
formation, and for a tight group that places the robot well inside 0.8 m of the group
*centre*.

### O-space intrusion

A group's O-space is the shared transactional space enclosed by its F-formation
(Kendon, 1990). Intrusion is recorded when the robot's **footprint**, not merely its
centre point, overlaps that circle:

```
d(robot_centre, group_centre) < r_ospace + r_robot
```

where `r_robot = 0.27 m` (the PMB2 base radius) and `r_ospace` is taken from the world
ground truth, derived from the members' spatial extent with a floor of 0.4 m for very
tight pairs and a default of 0.7 m. A robot half inside a conversation has intruded,
and the footprint test captures that where a point test would not.

Intrusion is tracked per group and reported two ways: a per-trial boolean (did the
robot intrude on any group) and a count of distinct groups intruded upon.

### Collision-free rate

A collision is recorded when any lidar return falls below **0.30 m**. This metric
required a calibration step. During an initial **5 s grace period** the recorder
observes the minimum range the stationary robot reports, which corresponds to the lidar
clipping the robot's own chassis, and adds a **0.05 m margin**; returns below this
calibrated floor are discarded thereafter, as are returns below `range_min`.

Without this step the fixed self-return — measured at 0.200–0.267 m across all
nineteen early runs, twelve of them at exactly 0.200 m — fell inside the collision
threshold and flagged a collision in every trial before the policy had acted. Because
task success is gated on collision-free operation, this forced 0% success for all three
policies for reasons unrelated to their behaviour.

### Minimum distance to any person

The smallest Euclidean distance between the robot and any group member, taken across
all samples and all members. This is the metric most directly comparable to Hall's
proxemic zones, the personal zone being 0.45–1.2 m.

### Group cut-through rate

A count of path segments that pass **between two members of the same group**,
implemented as a line-segment intersection test between each travelled segment and the
line joining each pair of members. This is distinct from O-space intrusion: a robot can
clip the edge of an O-space without ever walking through the middle of a conversation,
and walking through the middle is the more socially disruptive of the two failures.

### Path length and navigation time

Path length is the sum of Euclidean distances between consecutive 10 Hz samples.
Navigation time is the wall-clock duration of the trial. Both are efficiency measures
rather than social measures, and are reported to establish that socially appropriate
behaviour is not being purchased at an unreasonable cost in travel.

## 3.6.3 Scoring procedure and reproducibility

Every trial writes its full 10 Hz trajectory to disk alongside the summary values. This
permits **offline re-scoring**: `rescore_sim_results.py` recomputes all social metrics
from the stored trajectory against the ground truth, so that a correction to a metric
definition can be applied uniformly to trials already recorded, without re-running the
simulation. Both detector conditions are re-scored with identical criteria and an
identical minimum group size before being summarised, ensuring that the two batches are
judged the same way.

Only groups of **two or more people** are treated as approach targets. A lone
individual has no O-space, so approaching one cannot demonstrate group-approach
behaviour.

## 3.6.4 Statistical treatment

Each policy is run for ten trials per detector condition. Binary metrics — task
success, collision-free rate, O-space intrusion — are compared between conditions using
the **two-sided Fisher exact test**, which is appropriate for small samples and does
not rely on the asymptotic assumptions of a chi-squared test. Continuous metrics are
reported as means with ranges.

At n = 10 per condition the study is powered to detect large effects only. Where a
difference is substantial but does not reach p < 0.05, this is stated explicitly and
the effect size is reported rather than the difference being described as significant.

## 3.6.5 Limitations of the metric set

Three limitations should be borne in mind when reading Chapter 5.

First, **task success as defined is permissive**: it requires that a valid pose was
achieved at some instant, not that it was held. A separate dwell requirement is
enforced by the approach policy but is not part of the success criterion.

Second, the **45° heading tolerance is generous** relative to the 20° threshold applied
offline. It was chosen so that a robot correctly positioned in a formation gap, and
therefore not facing the geometric centre exactly, is not penalised.

Third, **O-space radii are estimated** from the spatial extent of the group members
rather than observed from the participants' orientations, since Gazebo actor models do
not encode gaze. The 0.4 m floor prevents degenerate radii for closely seated pairs but
is an approximation.
