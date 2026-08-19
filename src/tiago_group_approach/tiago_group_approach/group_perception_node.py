#!/usr/bin/env python3
"""
Phase G: LIVE group perception. Turns TIAGo's camera stream into a group
centroid in the map frame, which the policy nodes consume.

    RGB + depth  ->  YOLO person detection  ->  back-project to 3D metres
                 ->  transform to map frame  ->  cluster into groups
                 ->  publish /group_centroid

WHY THIS NODE MATTERS MORE THAN IT LOOKS
-----------------------------------------
Every perception result in this project so far is in 2D IMAGE PIXELS, because
the recorded PLUS-HRI video has no depth channel and no camera calibration.
That single limitation is why:
  - group clustering had to normalise distances by bounding-box width as a
    crude perspective proxy (scripts/cluster_groups.py),
  - the O-space "centre" is a pixel coordinate, not a position,
  - Objective 3's "within 0.3 m" criterion was unmeasurable and had to be
    re-specified in person-widths,
  - and 6 of the 8 proposal metrics could not be computed at all.

In simulation none of that applies. TIAGo publishes a real depth image and a
real CameraInfo, so each detected person can be back-projected to actual
METRES and transformed into the map frame. Clustering then happens in real
world coordinates, the group centroid is a real position, and the metrics
become measurable. This node is where the project stops being an offline
data-analysis exercise and becomes a robot system.

TOPICS
------
Subscribes:
    ~/rgb            sensor_msgs/Image        (default /head_front_camera/rgb/image_raw)
    ~/depth          sensor_msgs/Image        (default /head_front_camera/depth/image_raw)
    ~/camera_info    sensor_msgs/CameraInfo   (default /head_front_camera/rgb/camera_info)
Publishes:
    /group_centroid          geometry_msgs/PointStamped   (map frame) - what the policy nodes consume
    /detected_people         geometry_msgs/PoseArray      (map frame) - every person, for debugging
    /group_markers           visualization_msgs/MarkerArray - RViz overlay

DETECTOR BACKENDS
-----------------
    detector:=yolo            (default) YOLOv8n, ~0.005 s/frame (~200 FPS)
    detector:=locateanything  nvidia/LocateAnything-3B via the local service in
                              scripts/locateanything_service.py

Both were measured against the same ground-truth frames from sessions 1 and 3:
    LocateAnything-3B : 100.0% recall (30/30), 25.63 s/frame
    YOLOv8n           :  96.7% recall (29/30), ~0.005 s/frame
The recall difference is a SINGLE frame and the Wilson 95% intervals overlap
heavily (88.6-100% vs 83.3-99.4%), so it is not a statistically meaningful
gap. The speed difference is roughly 5000x and is unambiguous.

Hence the design: YOLOv8n is the default for the live loop, and
LocateAnything-3B is selectable so the proposal's named model can be
demonstrated end-to-end in the running system. When LA-3B is selected the node
automatically drops to one-shot mode (see `oneshot` below), because at 25 s per
frame continuous perception is impossible - the robot pauses, looks once,
decides, and then relies on Nav2's own reactive obstacle avoidance while
driving. That is a real architectural consequence of a measured constraint, and
is worth stating as such in the dissertation rather than hiding.

PARAMETERS
----------
    detector           (yolo) 'yolo' or 'locateanything'
    service_url        (http://127.0.0.1:8765) LocateAnything service endpoint
    oneshot            (auto)  'auto' | 'true' | 'false'. In one-shot mode the
                              node detects ONCE, publishes the centroid, and
                              then stops until re-triggered by publishing to
                              /perception/trigger. 'auto' enables it for
                              locateanything and disables it for yolo.
    group_distance_m   (1.5)  two people join the same group if within this
                              distance. Chosen from the F-formation literature:
                              a standing conversational group typically has an
                              O-space 1.2-1.5 m across, so ~1.5 m separates
                              "in this huddle" from "over there". Tune with
                              evidence, not vibes.
    min_group_size     (2)    a single person is not a conversational group.
    confidence         (0.4)  YOLO person-detection threshold.
    max_range_m        (8.0)  ignore detections beyond this - depth gets noisy
                              and distant groups are not approach candidates.
    publish_rate_hz    (2.0)  perception rate. Deliberately NOT video rate:
                              the approach decision is not a reactive control
                              loop, and running YOLO flat-out wastes CPU that
                              Gazebo needs.

RUN
---
    ros2 run tiago_group_approach group_perception_node
    # then watch it:
    ros2 topic echo /group_centroid
"""

from __future__ import annotations

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_ros import TransformException


# ---------------------------------------------------------------------------
# Image conversion WITHOUT cv_bridge.
#
# cv_bridge's compiled boost extension was built against numpy 1.x. This
# container has numpy 2.2.6 in ~/.local (required by pandas/scikit-learn for
# the offline pipeline), so importing cv_bridge dies with:
#     AttributeError: _ARRAY_API not found
#     ImportError: numpy.core.multiarray failed to import
# Unlike matplotlib or scipy, cv_bridge CANNOT be fixed with pip - it is a
# compiled ROS binary, and rebuilding it against numpy 2 is far more work than
# the 20 lines below.
#
# A sensor_msgs/Image is just a raw byte buffer plus a shape and an encoding
# string, so converting it by hand is straightforward and removes an entire
# class of dependency breakage.
# ---------------------------------------------------------------------------
_ENCODINGS = {
    'rgb8':    (np.uint8,   3),
    'bgr8':    (np.uint8,   3),
    'rgba8':   (np.uint8,   4),
    'bgra8':   (np.uint8,   4),
    'mono8':   (np.uint8,   1),
    'mono16':  (np.uint16,  1),
    '8UC1':    (np.uint8,   1),
    '8UC3':    (np.uint8,   3),
    '16UC1':   (np.uint16,  1),
    '32FC1':   (np.float32, 1),
    '64FC1':   (np.float64, 1),
}


def imgmsg_to_array(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> numpy array. Replaces cv_bridge.imgmsg_to_cv2()."""
    if msg.encoding not in _ENCODINGS:
        raise ValueError(f"unsupported image encoding: {msg.encoding!r}")
    dtype, channels = _ENCODINGS[msg.encoding]

    # Respect the publisher's endianness before interpreting the buffer.
    np_dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    data = np.frombuffer(msg.data, dtype=np_dtype)

    # `step` is the row stride in BYTES and may include padding, so reshape by
    # step and then trim - assuming width*channels would corrupt padded images.
    itemsize = np.dtype(dtype).itemsize
    stride = msg.step // itemsize
    data = data.reshape(msg.height, stride)[:, :msg.width * channels]

    if channels > 1:
        data = data.reshape(msg.height, msg.width, channels)
    return np.ascontiguousarray(data)


class GroupPerceptionNode(Node):
    def __init__(self):
        super().__init__('group_perception_node')

        self.declare_parameter('rgb_topic', '/head_front_camera/rgb/image_raw')
        self.declare_parameter('depth_topic', '/head_front_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/head_front_camera/rgb/camera_info')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('group_distance_m', 1.5)
        self.declare_parameter('min_group_size', 2)
        self.declare_parameter('confidence', 0.4)
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('publish_rate_hz', 2.0)
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('detector', 'yolo')
        self.declare_parameter('service_url', 'http://127.0.0.1:8765')
        self.declare_parameter('oneshot', 'auto')

        self.camera_info: CameraInfo | None = None
        self.last_process_time = 0.0
        self.model = None

        self.detector_kind = str(self.get_parameter('detector').value).lower()
        if self.detector_kind not in ('yolo', 'locateanything'):
            self.get_logger().warn(
                f"Unknown detector '{self.detector_kind}', falling back to 'yolo'.")
            self.detector_kind = 'yolo'

        # One-shot mode: LocateAnything at ~25 s/frame cannot drive a continuous
        # perception loop, so it detects once and waits to be re-triggered.
        oneshot_param = str(self.get_parameter('oneshot').value).lower()
        if oneshot_param == 'auto':
            self.oneshot = (self.detector_kind == 'locateanything')
        else:
            self.oneshot = oneshot_param in ('true', '1', 'yes')
        self.oneshot_done = False

        if self.detector_kind == 'yolo':
            # Imported lazily so the node gives a clear error rather than a
            # stack trace if ultralytics is missing from the ROS environment.
            try:
                from ultralytics import YOLO
            except ImportError:
                self.get_logger().fatal(
                    "ultralytics is not installed for this Python interpreter.\n"
                    "  Install it with:  pip install ultralytics\n"
                    "  NOTE: use the SYSTEM python (deactivate la3b_env first) - "
                    "the ROS node runs under /usr/bin/python3, not the LocateAnything venv."
                )
                raise
            model_name = self.get_parameter('model').value
            self.get_logger().info(f"Detector: YOLOv8 ({model_name})")
            self.model = YOLO(model_name)
        else:
            url = self.get_parameter('service_url').value
            self.get_logger().info(
                f"Detector: LocateAnything-3B via {url}\n"
                f"  Measured ~25.6 s/frame on this hardware, so one-shot mode is "
                f"{'ON' if self.oneshot else 'OFF'}.\n"
                f"  The service must already be running IN THE la3b_env venv:\n"
                f"    source la3b_env/bin/activate\n"
                f"    python3 scripts/locateanything_service.py")
            self.check_service()

        # --- TF -------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Publishers -------------------------------------------------------
        self.centroid_pub = self.create_publisher(PointStamped, '/group_centroid', 10)
        self.people_pub = self.create_publisher(PoseArray, '/detected_people', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/group_markers', 10)
        # Annotated camera view, so you can SEE what the detector is calling a
        # person. Add an Image display in RViz on /perception/image_annotated.
        self.debug_image_pub = self.create_publisher(
            Image, '/perception/image_annotated', 1)

        # --- Subscribers ------------------------------------------------------
        # Sensor data is best-effort; using RELIABLE here silently drops everything.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        # sensor_qos, NOT the default depth-10 profile.
        #
        # A plain `10` means RELIABLE. Gazebo publishes camera_info as SENSOR
        # DATA, which is BEST_EFFORT, and ROS 2 QoS compatibility is one-way: a
        # BEST_EFFORT subscriber happily receives from a RELIABLE publisher,
        # but a RELIABLE subscriber receives NOTHING from a BEST_EFFORT one -
        # silently. The topic lists, `count_publishers` returns 1, everything
        # looks connected, and not a single message ever arrives. That is
        # exactly what "Waiting for CameraInfo..." forever meant.
        self.create_subscription(CameraInfo, info_topic,
                                 self.camera_info_callback, sensor_qos)

        rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)
        # RGB and depth are published separately and never share an exact
        # timestamp, so an ApproximateTimeSynchronizer is required. slop=0.3s is
        # generous but these are static-ish scenes and a dropped pair costs a
        # whole perception cycle.
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.3)
        self.sync.registerCallback(self.image_callback)

        # Re-arm one-shot mode on demand.
        from std_msgs.msg import Empty as EmptyMsg
        self.create_subscription(EmptyMsg, '/perception/trigger',
                                 self.trigger_callback, 10)

        self.get_logger().info(
            f"Group perception ready.\n"
            f"  rgb   : {rgb_topic}\n"
            f"  depth : {depth_topic}\n"
            f"  info  : {info_topic}\n"
            f"  group_distance_m={self.get_parameter('group_distance_m').value}, "
            f"min_group_size={self.get_parameter('min_group_size').value}\n"
            f"Waiting for CameraInfo..."
        )

    def trigger_callback(self, _msg) -> None:
        """Re-arm a one-shot detector for another look."""
        self.oneshot_done = False
        self.last_process_time = 0.0
        self.get_logger().info("Re-triggered: will detect on the next frame.")

    # -------------------------------------------------------- detector backends
    def check_service(self) -> None:
        """Warn early if the LocateAnything service is not up, rather than
        failing on the first frame."""
        import urllib.request
        url = self.get_parameter('service_url').value.rstrip('/') + '/health'
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                import json as _json
                info = _json.loads(resp.read())
            self.get_logger().info(f"LocateAnything service OK: {info}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f"LocateAnything service not reachable at {url} ({exc}). "
                "Start it in the la3b_env venv before running this node.")

    def detect_people_boxes(self, rgb) -> list[tuple[int, int, int, int]]:
        """Return [(x1, y1, x2, y2), ...] from whichever backend is selected."""
        if self.detector_kind == 'yolo':
            results = self.model(rgb, classes=[0],
                                 conf=self.get_parameter('confidence').value,
                                 verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []
            return [tuple(int(v) for v in box)
                    for box in boxes.xyxy.cpu().numpy()]

        # --- LocateAnything-3B over the local service ---
        import base64
        import json as _json
        import urllib.request

        import cv2
        ok, buffer = cv2.imencode('.jpg', rgb)
        if not ok:
            self.get_logger().warn("JPEG encoding failed; skipping frame.")
            return []

        payload = _json.dumps({
            'image': base64.b64encode(buffer.tobytes()).decode('ascii')
        }).encode()

        url = self.get_parameter('service_url').value.rstrip('/') + '/detect'
        request = urllib.request.Request(
            url, data=payload, headers={'Content-Type': 'application/json'})
        try:
            # Generous timeout: 25 s/frame measured, plus headroom.
            with urllib.request.urlopen(request, timeout=120.0) as resp:
                result = _json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"LocateAnything request failed: {exc}")
            return []

        if 'error' in result:
            self.get_logger().error(f"LocateAnything error: {result['error']}")
            return []

        self.get_logger().info(
            f"LocateAnything: {len(result.get('boxes', []))} person(s) in "
            f"{result.get('inference_seconds', '?')}s")
        return [(int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2']))
                for b in result.get('boxes', [])]

    # ------------------------------------------------------------------ utils
    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_info is None:
            self.get_logger().info(
                f"CameraInfo received: {msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}")
        self.camera_info = msg

    def depth_to_metres(self, depth_image: np.ndarray, encoding: str) -> np.ndarray:
        """Gazebo publishes 32FC1 metres; real sensors often use 16UC1 millimetres."""
        if encoding == '16UC1' or depth_image.dtype == np.uint16:
            return depth_image.astype(np.float32) / 1000.0
        return depth_image.astype(np.float32)

    def person_depth(self, depth_m: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float | None:
        """
        Robust depth for one detection.

        Takes the MEDIAN of a small patch around the box centre rather than the
        single centre pixel: a person's bounding box contains background
        showing through around the limbs, and one unlucky pixel can land metres
        behind them. Median over the torso region is far more stable.
        """
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Torso-ish patch: middle of the box horizontally, upper-middle vertically.
        half_w = max(2, (x2 - x1) // 6)
        half_h = max(2, (y2 - y1) // 8)
        py = y1 + (y2 - y1) // 3          # upper third ~ chest height
        r0, r1 = max(0, py - half_h), min(depth_m.shape[0], py + half_h)
        c0, c1 = max(0, cx - half_w), min(depth_m.shape[1], cx + half_w)

        patch = depth_m[r0:r1, c0:c1]
        valid = patch[np.isfinite(patch) & (patch > 0.1)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def backproject(self, u: float, v: float, depth: float) -> tuple[float, float, float]:
        """Pixel + depth -> 3D point in the camera OPTICAL frame (x right, y down, z forward)."""
        k = self.camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return x, y, depth

    # ------------------------------------------------------------- clustering
    def cluster_people(self, points: list[tuple[float, float]]) -> list[list[int]]:
        """
        Connected-components clustering in REAL METRES.

        Same algorithm as the offline scripts/cluster_groups.py, but far more
        defensible here: offline it had to normalise pixel distance by
        bounding-box width to fake perspective correction. With depth we
        compare true ground-plane distances, so the threshold is a real,
        citable proxemic distance rather than an image-space heuristic.
        """
        threshold = self.get_parameter('group_distance_m').value
        n = len(points)
        if n == 0:
            return []

        adjacency = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.dist(points[i], points[j])
                if d < threshold:
                    adjacency[i][j] = adjacency[j][i] = True

        visited = [False] * n
        components: list[list[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            stack, comp = [start], []
            visited[start] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for other in range(n):
                    if adjacency[node][other] and not visited[other]:
                        visited[other] = True
                        stack.append(other)
            components.append(sorted(comp))
        return components

    # ---------------------------------------------------------------- main cb
    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self.camera_info is None:
            return

        # One-shot mode (LocateAnything): detect once, then wait to be
        # re-triggered. At ~25 s/frame a continuous loop is not possible, and
        # repeatedly queueing 25-second inferences would make the node
        # unresponsive rather than merely slow.
        if self.oneshot and self.oneshot_done:
            return

        # Throttle: perception runs at publish_rate_hz, not camera rate.
        now = self.get_clock().now().nanoseconds / 1e9
        min_period = 1.0 / max(0.1, self.get_parameter('publish_rate_hz').value)
        if now - self.last_process_time < min_period:
            return
        self.last_process_time = now

        try:
            rgb = imgmsg_to_array(rgb_msg)
            # YOLO expects BGR (OpenCV convention); Gazebo publishes rgb8.
            if rgb_msg.encoding == 'rgb8':
                rgb = rgb[:, :, ::-1]
            depth_raw = imgmsg_to_array(depth_msg)
        except Exception as exc:
            self.get_logger().warn(f"image conversion failed: {exc}")
            return

        depth_m = self.depth_to_metres(np.asarray(depth_raw), depth_msg.encoding)

        detections = self.detect_people_boxes(rgb)
        self.publish_debug_image(rgb, detections, depth_m, rgb_msg.header)
        if self.oneshot:
            # Mark done regardless of outcome, so a frame with nobody in it does
            # not cause another 25-second inference immediately afterwards.
            self.oneshot_done = True
            self.get_logger().info(
                "One-shot detection complete. Re-trigger with:  "
                "ros2 topic pub --once /perception/trigger std_msgs/msg/Empty {}")
        if not detections:
            self.publish_markers([], [])
            return

        max_range = self.get_parameter('max_range_m').value
        optical_frame = rgb_msg.header.frame_id
        map_frame = self.get_parameter('map_frame').value

        # --- Back-project each detection and transform into the map frame ----
        people_map: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in detections:
            depth = self.person_depth(depth_m, x1, y1, x2, y2)
            if depth is None or depth > max_range:
                continue

            u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            px, py, pz = self.backproject(u, v, depth)

            point = PointStamped()
            point.header.frame_id = optical_frame
            point.header.stamp = rgb_msg.header.stamp
            point.point.x, point.point.y, point.point.z = px, py, pz

            try:
                # Transform at the image's own timestamp so the robot's motion
                # between capture and processing does not smear positions.
                tf = self.tf_buffer.lookup_transform(
                    map_frame, optical_frame, rclpy.time.Time.from_msg(rgb_msg.header.stamp),
                    timeout=Duration(seconds=0.2))
                from tf2_geometry_msgs import do_transform_point
                in_map = do_transform_point(point, tf)
            except (TransformException, ImportError) as exc:
                self.get_logger().warn(f"TF {optical_frame}->{map_frame} failed: {exc}",
                                       throttle_duration_sec=5.0)
                return

            # Per-detection provenance. Without this a false positive is
            # indistinguishable from a real person: the pipeline reported a
            # confident group at (4.4, 2.8) while the only person in the world
            # stood at (-3.0, 0.0), and nothing in the log said why.
            self.get_logger().info(
                f"  detection bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
                f"depth={depth:.2f} m  camera=({px:.2f},{py:.2f},{pz:.2f})  "
                f"map=({in_map.point.x:.2f},{in_map.point.y:.2f})")

            people_map.append((in_map.point.x, in_map.point.y))

        if not people_map:
            self.publish_markers([], [])
            return

        self.publish_people(people_map, map_frame)

        # --- Cluster and pick the group to approach --------------------------
        components = self.cluster_people(people_map)
        min_size = self.get_parameter('min_group_size').value
        groups = [c for c in components if len(c) >= min_size]

        if not groups:
            self.get_logger().info(
                f"{len(people_map)} person(s) seen, but no group of >= {min_size}",
                throttle_duration_sec=10.0)
            self.publish_markers(people_map, [])
            return

        centroids = []
        for comp in groups:
            xs = [people_map[i][0] for i in comp]
            ys = [people_map[i][1] for i in comp]
            centroids.append((sum(xs) / len(xs), sum(ys) / len(ys), len(comp)))

        # Approach the LARGEST group; ties broken by proximity to the robot is
        # a reasonable future refinement, but largest-first matches the offline
        # `is_largest_group` convention used throughout Phase C.
        centroids.sort(key=lambda c: -c[2])
        cx, cy, size = centroids[0]

        msg = PointStamped()
        msg.header.frame_id = map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = cx, cy, 0.0
        self.centroid_pub.publish(msg)

        self.get_logger().info(
            f"{len(people_map)} people -> {len(groups)} group(s); "
            f"approaching group of {size} at ({cx:.2f}, {cy:.2f})",
            throttle_duration_sec=5.0)

        self.publish_markers(people_map, centroids)

    # -------------------------------------------------------------- publishing
    def publish_debug_image(self, bgr, detections, depth_m, header) -> None:
        """
        Republish the camera image with detection boxes drawn on it.

        Worth the few milliseconds: a bounding box round a chair tells you
        instantly what a wrong group centroid means, whereas coordinates in a
        log leave you guessing whether perception, TF or the policy is at
        fault.
        """
        # Deliberately NOT gated on subscriber count. RViz's Image display
        # subscribes lazily, so a count of zero at the moment the first frames
        # arrive meant nothing was ever published, the display stayed grey, and
        # it only sprang to life after manually re-picking the topic. Drawing a
        # few boxes on a 640x480 frame twice a second is cheap; the confusion
        # was not.
        try:
            import cv2
            img = np.ascontiguousarray(bgr.copy())
            for (x1, y1, x2, y2) in detections:
                d = self.person_depth(depth_m, x1, y1, x2, y2)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                label = f"person {d:.1f}m" if d is not None else "person ?m"
                cv2.putText(img, label, (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, f"detections: {len(detections)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            msg = Image()
            msg.header = header
            msg.height, msg.width = img.shape[0], img.shape[1]
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = msg.width * 3
            msg.data = img.tobytes()
            self.debug_image_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"debug image failed: {exc}",
                                   throttle_duration_sec=30.0)

    def publish_people(self, people: list[tuple[float, float]], frame: str) -> None:
        msg = PoseArray()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in people:
            pose = Pose()
            pose.position.x, pose.position.y = x, y
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.people_pub.publish(msg)

    def publish_markers(self, people, centroids) -> None:
        frame = self.get_parameter('map_frame').value
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.frame_id = frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        for idx, (x, y) in enumerate(people):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'people'
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.85
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.4
            m.scale.z = 1.7
            m.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.55)
            array.markers.append(m)

        for idx, (x, y, size) in enumerate(centroids):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'group_centre'
            m.id = idx
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            # first (largest) group highlighted - that is the approach target
            m.color = (ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9) if idx == 0
                       else ColorRGBA(r=1.0, g=0.7, b=0.2, a=0.7))
            array.markers.append(m)

            ring = Marker()
            ring.header.frame_id = frame
            ring.header.stamp = stamp
            ring.ns = 'ospace'
            ring.id = idx
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x, ring.pose.position.y, ring.pose.position.z = x, y, 0.02
            ring.pose.orientation.w = 1.0
            ring.scale.x = ring.scale.y = 1.4   # ~O-space diameter
            ring.scale.z = 0.02
            ring.color = ColorRGBA(r=1.0, g=0.4, b=0.4, a=0.25)
            array.markers.append(ring)

        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = GroupPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
