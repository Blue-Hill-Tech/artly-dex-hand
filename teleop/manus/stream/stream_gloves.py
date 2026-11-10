"""Print live MANUS hand skeletons using the bundled Linux SDK in integrated mode."""

from pathlib import Path
import sys
import threading
import time

# Use the SDK already unpacked beside agi-os; no SDK installation is needed.
SDK_PYTHON = (
    Path(__file__).resolve().parents[4] / "MANUS_Core_3.2.0_SDK_Linux" / "Python"   # beside agi-os (this file: teleop/manus/stream/)
)
sys.path.insert(0, str(SDK_PYTHON))

from manus_sdk import ManusSDK, ffi, lib
from manus_sdk.generated._enums import AxisPolarity, AxisView, SDKReturnCode, Side


def check(result, operation):
    """SDK calls return a status code instead of raising Python exceptions."""
    if result != SDKReturnCode.Success:
        raise RuntimeError(f"{operation}: {SDKReturnCode(result).name}")


class GloveStream:
    def __init__(self):
        self.sdk = ManusSDK()  # Keeps the native callback registered/alive.
        self.lock = threading.Lock()
        self.latest = {}  # Pending newest pose per glove ID.
        self.error = None
        self.glove_sides = {}
        self.frame_id = 0

    def on_landscape(self, landscape):
        """Device metadata identifies each glove's physical left/right side."""
        sides = {}
        for index in range(landscape.gloveDevices.gloveCount):
            glove = landscape.gloveDevices.gloves[index]
            sides[int(glove.id)] = {
                Side.Left: "Left", Side.Right: "Right",
            }.get(glove.side, "Unknown")
        with self.lock:
            self.glove_sides = sides

    def on_skeletons(self, stream):
        """Called by an SDK thread whenever a frame arrives, not by our loop."""
        received = time.perf_counter()
        self.frame_id += 1
        frame_id = self.frame_id
        try:
            frame = []
            for index in range(stream.skeletonsCount):
                info = ffi.new("RawSkeletonInfo *")
                check(lib.CoreSdk_GetRawSkeletonInfo(index, info), "Get skeleton info")
                nodes = ffi.new("SkeletonNode[]", info.nodesCount)
                check(
                    lib.CoreSdk_GetRawSkeletonData(index, nodes, info.nodesCount),
                    "Get skeleton data",
                )
                # Copy into Python values before returning from the callback.
                # Do not pass pointers to SDK-owned stream data to another thread.
                hand = {"glove_id": int(info.gloveId), "nodes": [],
                        "frame_id": frame_id, "received": received}
                for node in nodes:
                    p = node.transform.position
                    q = node.transform.rotation
                    hand["nodes"].append({
                        "id": int(node.id),
                        "position": (float(p.x), float(p.y), float(p.z)),
                        "rotation": (float(q.x), float(q.y), float(q.z), float(q.w)),
                    })
                frame.append(hand)
            with self.lock:
                # A callback may contain only one glove. Preserve the other
                # glove's pending pose while replacing older poses for this one.
                for hand in frame:
                    self.latest[hand["glove_id"]] = hand
        except Exception as exc:
            # Exceptions cannot propagate across the native callback boundary.
            with self.lock:
                self.error = exc

    def run(self):
        check(lib.CoreSdk_InitializeIntegrated(), "Initialize integrated SDK")
        try:
            check(self.sdk.register_landscape_callback(self.on_landscape),
                  "Register device metadata callback")
            check(self.sdk.register_raw_skeleton_callback(self.on_skeletons),
                  "Register skeleton callback")

            # Right-handed coordinates, Z up, X away from viewer, meters.
            coordinates = ffi.new("CoordinateSystemVUH *")
            coordinates.handedness = Side.Right
            coordinates.up = AxisPolarity.PositiveZ
            coordinates.view = AxisView.XFromViewer
            coordinates.unitScale = 1.0
            check(lib.CoreSdk_InitializeCoordinateSystemWithVUH(coordinates[0], True),
                  "Set coordinate system")

            # An empty host connects to the Core embedded in this process.
            host = ffi.new("ManusHost *")
            check(lib.CoreSdk_ConnectToHost(host[0]), "Connect integrated SDK")
            print("Waiting for glove skeletons. Move your fingers; Ctrl+C stops.")

            last_frame = time.monotonic()
            while True:
                time.sleep(0.5)  # Print at 2 Hz; SDK callbacks run at their own rate.
                with self.lock:
                    frame, self.latest = self.latest, {}
                    error = self.error
                    sides = self.glove_sides.copy()
                if error is not None:
                    raise error
                if not frame:
                    if time.monotonic() - last_frame > 5:
                        print("No recent data: check gloves, dongle, and close SDKClient.")
                        last_frame = time.monotonic()
                    continue
                last_frame = time.monotonic()
                for hand in frame.values():
                    side = sides.get(hand['glove_id'], "Unknown")
                    print(f"{side} glove {hand['glove_id']:08X}: {len(hand['nodes'])} nodes")
                    # All nodes are available in hand['nodes']; show the last one.
                    if hand["nodes"]:
                        node = hand["nodes"][-1]
                        p = ", ".join(f"{v:.4f}" for v in node["position"])
                        q = ", ".join(f"{v:.4f}" for v in node["rotation"])
                        print(f"  Node {node['id']}: position xyz=({p}) m; quaternion xyzw=({q})")
        finally:
            check(lib.CoreSdk_ShutDown(), "Shut down SDK")


if __name__ == "__main__":
    try:
        GloveStream().run()
    except KeyboardInterrupt:
        print("\nStopped; SDK released.")
    except RuntimeError as exc:
        sys.exit(str(exc))
