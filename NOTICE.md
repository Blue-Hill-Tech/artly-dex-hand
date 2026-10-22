# Attribution and licensing

Artly Dex Hand is the dexterous-hand product family from Blue Hill Tech, Inc.
(DBA Artly). D21 is its first model. This repository organizes engineering
materials for integration and evaluation; it does not transfer ownership of
third-party materials to Blue Hill Tech, Inc.

- The standalone Python SDK README says `MIT`, but its supplied folder contains
  no standalone license text or copyright-holder notice. Complete licensing and
  redistribution terms remain to be confirmed.
- The ROS2 `hand_control` package declares Apache-2.0 and retains its supplied
  [LICENSE](sdk/ros2/hand_control/LICENSE). This does not establish licensing for
  separately supplied CAD, media, documentation or controller software.
- No repository-wide open-source license is granted or inferred. Redistribution
  and branding use remain subject to the relevant rights holders' terms.

The published subset retains original asset bytes except for four ROS2 display
metadata fields and metadata removal from three interface PDFs. ROS2 package
descriptions and URDF robot names use Artly D21; package names, link/joint names,
mesh references and driver behavior are unchanged. PDF page contents are unchanged.
Original packages and full source provenance are retained outside this repository.

The [asset manifest](provenance/asset-manifest.json) records paths, sizes and
SHA-256 hashes for the published engineering assets. The generated model render
is derived from the supplied simulation geometry; its source and output hashes
are recorded in [render provenance](provenance/model-render.json).
