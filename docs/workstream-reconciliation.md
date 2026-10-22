# Technical assets and the sales workstream

Reviewed October 21, 2026 (Pacific).

The [Artly Dexterous Hand Sales and Marketing Workstream](https://docs.google.com/document/d/1uXUU8qXy83FfybKG147cluvHgJuXlqmzoIhIsZiStuU/edit)
is the living sales and marketing planning document. This repository provides
public engineering resources for D21. The workstream was read for reconciliation
and was not modified during repository preparation.

The linked document retains its own access permissions. Public access to this
repository does not grant access to that document, and it is not required for
SDK, CAD or simulation use.

| Workstream area | Public resources | Remaining work |
| --- | --- | --- |
| SDK and API | Python RS485/CAN FD source, API reference, example | Firmware and adapter compatibility, physical validation, complete licensing |
| ROS2 integration | `hand_control` 0.0.1, source, launch, RViz and meshes | ROS2 Jazzy runtime qualification and documented implementation fixes |
| CAD and mechanical interface | Hand and RealMan adapter STEP files, mechanical PDFs | Confirm exact shipped variant and adapter fit |
| URDF / MJCF | Left/right models, meshes, MuJoCo and PyBullet examples | Coupling and dynamics validation against the physical hand |
| Protocol and electrical information | Reviewed technical notes, English/Chinese register maps, electrical reference | Resolve connector and revision differences against actual hardware |
| Photos and demonstrations | Three photographed technical sheets and one electrical diagram | Product photography, measured application demonstrations and videos |
| Desktop tooling | Contact Artly for an approved controller package; not distributed here | Installation, hardware operation and license review |

Asset availability does not establish customer-release acceptance. Physical
performance, firmware update and recovery procedures, supported configurations,
software redistribution rights and acceptance criteria remain open. Proposed
owners, dates and deliverables in the workstream remain planning items.
