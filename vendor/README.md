# Package availability

The public engineering package includes [Python SDK source](../sdk/python),
[ROS2 source](../sdk/ros2/hand_control), [CAD](../cad),
[simulation models](../simulation), [technical references](../docs/README.md)
and [reference images](../media/README.md).

The Windows controller, its manuals and original delivery archives are not
distributed in this repository. Contact Artly for an approved controller package
and matching setup instructions. The executable was not run during the offline
engineering review.

The [published asset manifest](../provenance/asset-manifest.json) records the
repository path, byte size and SHA-256 hash of each listed asset. From the
repository root, verify the files with:

```bash
python3 tools/verify_assets.py
```

STEP models use canonical paths under `cad/`. Mesh filenames remain where the
models expect them. [NOTICE](../NOTICE.md) describes attribution and license
scope; file availability does not establish hardware qualification or a complete
redistribution license.
