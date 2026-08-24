#!/usr/bin/env python3
# Run from the project root:  python3 tests/test_periodic_perception.py
"""Exercise periodic_step's threading handoff without ROS.

Verifies: (1) the first call dispatches and returns None, (2) calls during a
slow inference return None rather than blocking or re-dispatching, (3) the
result is delivered exactly once, carrying the ORIGINAL frame, (4) the next
look only happens after retrigger_period_s.
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
