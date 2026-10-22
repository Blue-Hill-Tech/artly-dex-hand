# Artly Dex Hand documentation

D21 is the first Artly Dex Hand model. Current SDKs, CAD and simulation assets
in this repository belong to D21.

## Start here

- [D21 model overview](models/d21.md): specifications, interfaces and available resources.
- [Setup guide](setup.md): Python, ROS2 and standalone simulation.
- [Validation and known limitations](validation.md): offline checks and remaining qualification.

## Technical references

| Reference | Available material |
| --- | --- |
| Specifications and protocol | [Reviewed specifications and technical notes](product-review.md) |
| Register addresses | [English workbook](reference/en/register-address-table.xlsx) · [Chinese workbook](reference/zh/register-address-table.xlsx) |
| Electrical interface | [Electrical reference](reference/zh/electrical-interface.pdf) |
| Mechanical interface | [Mechanical reference](reference/zh/mechanical-interface.pdf) · [Earlier drawing](reference/zh/D21-0813.pdf) |
| Installation and indicators | [Technical reference images](../media/README.md) |

Reported values are documentation references, not independent test results. The
[product review](product-review.md) identifies source boundaries and discrepancies.

## Software and models

- [Python SDK and API](../sdk/python/API.md)
- [ROS2 package](../sdk/ros2/hand_control)
- [Simulation assets](../simulation) and [setup](setup.md#standalone-simulation)
- [STEP CAD](../cad)
- [Package availability](../vendor/README.md)
- [Images and video availability](../media/README.md)

## Review and provenance

- [Product and document review](product-review.md)
- [SDK and simulation review](sdk-review.md)
- [Published asset manifest](../provenance/asset-manifest.json)
- [Offline validation results](../provenance/offline-validation.json)
- [Attribution and license scope](../NOTICE.md)

The [workstream reconciliation](workstream-reconciliation.md) relates the public
engineering resources to the sales and marketing plan. Its linked Google Doc
retains its own access permissions and is not needed to use these resources.
Proposed dates and owners in that plan are not release commitments.
