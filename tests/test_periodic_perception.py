# Run from the project root:  python3 tests/test_periodic_perception.py
"""
UNIT TEST for the perception node's periodic (threaded) detection mode.

    python3 tests/test_periodic_perception.py

============================================================================
WHAT IS BEING TESTED AND WHY IT NEEDED A TEST
============================================================================
LocateAnything-3B takes seconds per image, while the ROS node that calls it
must stay responsive - publishing markers, redrawing the annotated camera view
and answering callbacks throughout. The solution is "periodic mode": inference
runs on a WORKER THREAD and the result is collected later.

Threaded handoffs are easy to get subtly wrong, and the failure modes are
nasty: a result delivered twice, a result silently dropped, or the node
freezing while it waits. None of those show up as an error message - they show
up as a robot that behaves oddly, hours into an overnight run.

============================================================================
HOW IT TESTS WITHOUT A ROBOT
============================================================================
There is no ROS, no Gazebo and no GPU here. The ROS imports are replaced with
stubs and a small fake object stands in for the node, exposing only what
periodic_step actually touches. A sleep imitates the slow model. The real
scheduling logic is then exercised directly, in under two seconds.

Four properties are checked:

    1. the first call dispatches work and returns immediately
    2. calls arriving DURING inference do not block and do not re-dispatch
    3. the result is delivered exactly once, and carries the frame it was
       computed FROM - not whatever the camera is showing now, which would
       place detected people where the robot has since moved to
    4. the next look waits for retrigger_period_s

Point 3 is the one that matters most in practice: transforms are looked up at
the captured frame's timestamp, so mixing up which frame a detection belongs
to puts people metres from where they really are.
"""
import sys, time, types, threading
sys.path.insert(0, 'src/tiago_group_approach')

# Stub the ROS imports group_perception_node needs at import time.
for name in ['rclpy','rclpy.node','rclpy.duration','rclpy.qos','message_filters',
             'geometry_msgs','geometry_msgs.msg','sensor_msgs','sensor_msgs.msg',
             'std_msgs','std_msgs.msg','visualization_msgs','visualization_msgs.msg',
             'tf2_ros','cv2']:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['rclpy.node'].Node = object
sys.modules['rclpy.duration'].Duration = lambda **k: None
for m,attrs in [('rclpy.qos',['QoSProfile','ReliabilityPolicy','HistoryPolicy']),
                ('geometry_msgs.msg',['Point','PointStamped','Pose','PoseArray']),
                ('sensor_msgs.msg',['CameraInfo','Image']),
                ('std_msgs.msg',['ColorRGBA']),
                ('visualization_msgs.msg',['Marker','MarkerArray']),
                ('tf2_ros',['Buffer','TransformListener','TransformException'])]:
    for a in attrs: setattr(sys.modules[m], a, type(a,(object,),{}))
sys.modules['tf2_ros'].TransformException = type('TransformException',(Exception,),{})

from tiago_group_approach.group_perception_node import GroupPerceptionNode

class Fake:
    """Minimal stand-in exposing exactly what periodic_step touches."""
    mode='periodic'; _worker=None; _pending=None; _last_dispatch_time=0.0
    _last_boxes=[]; _last_depth=None
    def __init__(self): self._worker_lock=threading.Lock(); self.t=1000.0; self.published=[]
    def get_clock(self):
        outer=self
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=outer.t*1e9))
    def get_parameter(self,n): return types.SimpleNamespace(value=10.0)
    def get_logger(self): return types.SimpleNamespace(info=lambda *a,**k:None, warn=lambda *a,**k:None)
    def publish_debug_image(self,*a,**k): self.published.append(k.get('note'))
    def depth_to_metres(self,d,e): return d
    def detect_people_boxes(self,rgb):
        time.sleep(1.5)                      # stand in for an 8.4 s inference
        return [(1,2,3,4)]

periodic_step = GroupPerceptionNode.periodic_step
infer = GroupPerceptionNode._infer_worker
Fake.periodic_step = periodic_step
Fake._infer_worker = infer

import numpy as np
sys.modules['tiago_group_approach.group_perception_node'].imgmsg_to_array = lambda m: np.zeros((4,4,3),dtype=np.uint8)

f = Fake()
msg1 = types.SimpleNamespace(encoding='bgr8', header='FRAME_1')
msg2 = types.SimpleNamespace(encoding='bgr8', header='FRAME_2')

r = f.periodic_step(msg1, msg1, 'preview')
assert r is None, "first call must dispatch, not return a result"
assert f._worker is not None and f._worker.is_alive(), "worker should be running"
print("1. dispatched without blocking                       OK")

t0=time.time()
for _ in range(5):
    f.t += 0.05
    assert f.periodic_step(msg2, msg2, 'preview') is None
assert time.time()-t0 < 0.5, "callbacks blocked while inference was in flight"
assert f.published[-1] == '(looking...)', f"expected live view, got {f.published[-1]}"
print("2. stayed responsive during inference                OK")

f._worker.join(timeout=10)
f.t += 2.0
r = f.periodic_step(msg2, msg2, 'preview')
assert r is not None, "result was never delivered"
rgb_msg, depth_msg, rgb, depth_m, dets = r
assert rgb_msg.header == 'FRAME_1', f"must carry the ORIGINAL frame, got {rgb_msg.header}"
assert dets == [(1,2,3,4)], dets
print("3. delivered result, carrying the captured frame     OK")

assert f.periodic_step(msg2, msg2, 'preview') is None, "result delivered twice"
print("4. result consumed exactly once                      OK")

f.t += 1.0
f.periodic_step(msg2, msg2, 'preview')
assert f._worker is None or not f._worker.is_alive(), "re-looked before the period elapsed"
f.t += 12.0
f.periodic_step(msg2, msg2, 'preview')
assert f._worker is not None and f._worker.is_alive(), "never re-looked after the period"
print("5. honoured retrigger_period_s                       OK")
print("\nALL PERIODIC-MODE CHECKS PASSED")
