# Artly Dex Hand D21: product and documentation review

Reviewed October 21, 2026 (Pacific). This is a source review, not a physical
hand test or certification review. Some original references are retained privately;
the citations below identify the basis of the engineering notes without publishing
those source documents.

## Source key

- **S**: privately retained English product and installation reference, six pages,
  derived from technical-sheet photographs dated July 15, 2026.
- **P**: privately retained English D21 communication protocol, 27 pages.
- **R**: [English register workbook](reference/en/register-address-table.xlsx), four sheets.
- **RC**: [Chinese register workbook](reference/zh/register-address-table.xlsx), four sheets.
- **E**: [Electrical interface reference](reference/zh/electrical-interface.pdf), three pages.
- **I**: [Earlier interface drawing](reference/zh/D21-0813.pdf), three pages.
- **M**: [Mechanical interface reference](reference/zh/mechanical-interface.pdf), two pages.
- **H**: privately retained English controller manual, 24 pages; describes V1.0,
  August 2026. The executable was not run and its version was not independently verified.

Metadata dates describe files, not qualification or guaranteed firmware compatibility.

## Supplier specification table

Every row below is **supplier-reported; not independently measured**. S p.1 is visually consistent with the privately retained original specification photographs dated July 15, 2026.

| Parameter | Supplier-reported value | Source |
| --- | --- | --- |
| Total / active degrees of freedom | 21 / 16 | S p.1 |
| Drive transmission | Linkage drive | S p.1 |
| Reference dimensions | 200 × 85 × 47 mm (L × W × D) | S p.1 |
| Bare hand mass | 815 g, excluding optional external sensors | S p.1 |
| Maximum fingertip force | 25 N | S p.1 |
| Maximum static whole-hand grasp load | 20 kg | S p.1; static, not a dynamic payload rating |
| Minimum open-close cycle time | 0.8 s | S p.1; load/test definition unverified |
| Repetitive positioning accuracy | ±0.2 mm | S p.1; test method unprovided |
| Motor | Coreless BLDC motor | S p.1 |
| Tactile sensing | 3D tactile array on all five fingertips | S p.1; shipped options unverified |
| Minimum detectable force | ≤0.1 N | S p.1; threshold, not force accuracy |
| Control modes | Position / force-position hybrid | S p.1; protocol exposes position and torque mode settings |
| Control frequency / latency | 10 kHz / 0.2 ms | S p.1; loop and measurement boundaries unspecified |
| Host interfaces | RS485 / CAN FD | S p.1; P pp.3–6 |
| No-load grasp cycle life | ≥1.5 million cycles | S p.1; loaded endurance unprovided |
| Electrical supply | 12 V rated; 240 W maximum; 20 A maximum | E p.1 embedded figure; I p.3 corroborates 12 V / 240 W |

Do not reinterpret the 10 kHz claim as host command/telemetry performance. P p.15 explicitly says **joint-position data update rate 50 Hz**. The supplier frequency and latency are distinct quantities, not automatically contradictory. No physical force, load, precision, timing, thermal, or endurance test was performed.

## Interface and setup facts

P pp.3–6 and R `Specifications ` A1:D7 establish:

- Default slave ID **0x01**.
- RS485 **500000 bps, 8 data bits, 1 stop bit, no parity**, fixed according to the protocol.
- CAN FD **1 Mbps arbitration / 2 Mbps data**, fixed according to the protocol.
- Modbus RTU framing over RS485 and encapsulation over CAN FD; vendor-specific function **0x05 reads 8-bit tactile registers**. Do not claim complete unmodified standard-Modbus compatibility.
- Function 0x03 reads holding registers; 0x04 reads 16-bit inputs; 0x06 writes one holding register; 0x10 writes multiple holding registers.
- Position commands occupy holding addresses 0–15. Speed settings use 16–20 (one per finger); mode at 21; tactile zeroing at 47; slave ID at 58; fan at 59; save configuration at 60; hand enable at 66; torque parameters at 68–83. R ` Holding Register Addresses` B3:E30.
- P p.13 says hand defaults to **enabled after power-on** and disable stops joint motion; this is supplier-described behavior, not a qualified emergency stop.
- R ` 8 Bit Input Registers` B3:E22 and P pp.18–19 describe 116 tactile points for thumb and 52 per other finger, with X/Y/Z distributions and resultant values. Do not equate these to independent normal-force accuracy claims.
- H p.5 lists Windows 10/11 only for the GUI, 8 GB RAM, ≥1 GB disk, ≥1920×1080 display and one USB port. Linux/macOS GUI support is explicitly referred to technical support for custom versions. H p.6 calls it a portable executable with no Python installation, requiring the matching USB-to-serial adapter driver. These are GUI requirements, not Python/ROS2 platform qualifications.

## Electrical and mechanical notes for engineering documentation

E pp.1–3 provide the newer electrical reference:

- Separate XT30UM two-pin power and GHD-10AB ten-pin communication connectors.
- **Pins 1–6 are reserved EtherCAT connections, currently disabled; leave them unconnected. EtherCAT is not an available interface in this delivery.**
- Pin 7: CANFD_H, black; pin 8: CANFD_L, white.
- Pin 9: RS485 A, red; pin 10: RS485 B, yellow-green.
- Only one interface/bus type may be connected at a time; do not mix CAN FD and RS485 connections.
- Power off before plugging/unplugging communication cables or changing wiring; the supplier warns hot plugging can damage communication ICs.
- For long cable runs with the hand at a bus physical endpoint, the supplier calls for a 120 Ω termination across the relevant differential pair; shielded twisted-pair wiring is recommended. Exact assembled-cable termination and supply protection have not been verified on hardware.
- Do not use a signal pin labelled GND as the power return by assumption; use the separate power connector and released cable for the exact revision.

S p.2 / IMG_9863 describe M3 hand-to-base screws and M4 external mounting, preserving cable passages/cooling openings. I p.1 shows **4-M3 depth 4.5**, 46 × 24 hole pattern. The photographed base adapter shows **4-M4 depth 5**, 4-Ø3.2 and 50/46/24 mm dimensions with ±0.03 qualifiers. These refer to different mounting features; do not merge them into one fabrication drawing. Released drawing revision, screw length, torque, mounting load limits, and exact shipped configuration remain to be confirmed.

## Concrete documentation inconsistencies

Preserve originals and put these in known limitations; do not silently correct supplier material.

1. **Thumb rotation conversion:** P p.20 Table 3.17 lists −25° to +35° but gives `value / 2000 × 70 − 35`, which yields −35° to +35°. SDK inspection confirms the code uses −35° to +35°. Require supplier/firmware confirmation before relying on endpoint conversion.
2. **Velocity read addresses:** P pp.15–16 section title says addresses 16–31 but Table 3.12 rows incorrectly display 0–15. Both English R ` 16 Bit Input Registers` B19:B34 and Chinese RC corresponding rows give 16–31; SDK inspection confirms code reads 16–31.
3. **Tactile X/Y boundary:** P p.21 table summarizes −12.8 to +12.7, but the explicit piecewise formula makes raw128 +12.8 and raw129 −12.7. SDK inspection confirms code follows the piecewise formula (`<=128`). Ask supplier to resolve raw128 sign and encoding before calibration/force use.
4. **Electrical document revisions:** I p.3 labels communication connector SHC-10AB and pin8 `CANFD_F`; E p.1 instead calls it GHD-10AB, and E p.2 / supplied wiring image label pin8 CANFD_L. Pin drawings also use different views/orientations. Treat E as a newer reference, but verify exact shipped connector orientation/revision before wiring.
5. **Command example count:** P p.23 Example1 prose says addresses0–2 / three joints but request count `00 04` and eight response data bytes describe four registers. Do not copy example bytes without reconciling intended count and calculating CRC.
6. **Product control terminology:** S says position / force-position hybrid; P and R expose position / torque mode. Do not claim the provided SDK implements a validated force-position hybrid controller merely because the product table uses that phrase.

## Media reviewed

The original source set contained five photographs of printed technical sheets
and one electrical diagram. The published [media index](../media/README.md)
contains the mechanical, wiring and indicator sheets plus the electrical diagram.
The two specification photographs are retained privately. No demonstration video
was found in the source tree or the three original archives, including nested ZIPs.
The model banner is a render of the supplied geometry, not product photography.

## Validation performed and remaining gaps

- During the original source review, parsed eight PDF files with pypdf and extracted all pages; counts 6,3,27,24,22,2,24,3 = 111 pages total.
- Visually reviewed all six pages of the English product PDF, all six supplied images, electrical PDF all three pages, older interface PDF pp.1/3, and protocol pp.15/16/20/21. This caught an embedded 20 A power caption not present in text extraction.
- Opened both XLSX files read only with bundled openpyxl. ZIP CRC checks pass, four sheets each, zero formula cells. Inspected all nonempty row values and key register/communication ranges. This is file readability/inventory validation, not confirmation that firmware implements each register.
- English workbook worksheet extents: 7×4,81×8,111×5,22×5. Chinese:6×4,81×8,111×5,26×5; blank/formatting rows inflate some extents. Chinese tactile sheet has inter-finger blank rows.
- Reconciled previous electrical/SDK gaps: pinout, electrical ratings, register maps, Python/ROS2 and simulation files are available here for review. The controller, manuals and protocol originals are retained privately; availability does not establish qualification.
- Still unverified: exact shipped hand/firmware/adapter revision; physical pinout; motion limits/conversion conflicts; real bus timing; Python/ROS2 functional operation; GUI execution; tactile calibration/accuracy; fault recovery and safe-stop behavior; dynamic grasping and loaded endurance; environment/ingress/cleaning/food-service claims; supply/transient protection; warranty and third-party certification evidence.
- No controller executable or actuator example was run. No device opened, bus connected, motor enabled, or motion commanded.
