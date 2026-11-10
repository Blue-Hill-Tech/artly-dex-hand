// C++ port of the vendor's Python transport for the AAC D21 hand:
// Modbus RTU frames carried in CAN FD frames (request ID 0x010, response
// ID 0x020) over a CANable 2 adapter speaking slcan on a serial tty.
// Register semantics mirror hand/aac/Python SDK/D21ModbusCore.py exactly.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dexrt {

struct CanFrame {
  uint32_t id = 0;
  bool fd = false;
  bool brs = false;
  std::vector<uint8_t> data;
};

class SlcanSerial {
 public:
  SlcanSerial() = default;
  ~SlcanSerial();
  SlcanSerial(const SlcanSerial&) = delete;
  SlcanSerial& operator=(const SlcanSerial&) = delete;

  // Opens the tty, configures raw 8N1 at tty_baudrate, then issues the same
  // slcan sequence python-can uses: C, S<nominal>, Y<data>, O.
  void open(const std::string& device, int tty_baudrate = 10000000, int nominal_bitrate = 1000000,
            int data_bitrate = 2000000, double sleep_after_open_s = 2.0);
  void close();
  bool is_open() const { return fd_ >= 0; }

  void send(const CanFrame& frame, double timeout_s);
  bool recv(CanFrame& frame, double timeout_s);  // false on timeout

  static std::string encode(const CanFrame& frame);
  static bool decode(const std::string& line, CanFrame& frame);
  static int len2dlc(size_t length);
  static size_t dlc2len(int dlc);

 private:
  void write_line(const std::string& line, double timeout_s);
  int fd_ = -1;
  std::string buffer_;
};

// Register scaling per firmware (mirrors D21ModbusCore.REGISTER_MAPS in the Python SDK).
//   v1: original firmware, 0-2000 registers.  v2: new firmware, 0-40000 registers, signed velocity readback,
//   current and torque centred on 20000.
struct D21RegisterMap {
  const char* name;
  int full_scale;        // position / speed register at 100 % of the range
  int signed_zero;       // register value of 0 for current, torque and (v2) velocity
  double velocity_scale; // deg/s per count
  int velocity_offset;   // v1 0 (unsigned), v2 20000
  int canfd_data_bitrate;  // fixed by the firmware: v1 2 Mbit/s, v2 5 Mbit/s (nominal stays 1 Mbit/s)
  int thumb_torque_max;    // largest |torque| count for motors 1-4: v1.1 manual table 3.5 limits them to +-10000 on v2
};
inline constexpr D21RegisterMap kD21RegisterMapV1{"v1", 2000, 1000, 400.0 / 2000, 0, 2000000, 1000};
inline constexpr D21RegisterMap kD21RegisterMapV2{"v2", 40000, 20000, 400.0 / 20000, 20000, 5000000, 10000};
// "v1" or "v2" from the D21_REGISTER_MAP environment variable; v2 when unset (same rule as the Python backend).
const D21RegisterMap& d21_register_map_from_env();

class D21Modbus {
 public:
  static constexpr uint32_t kRequestId = 0x010;
  static constexpr uint32_t kResponseId = 0x020;
  static constexpr size_t kMaxResponseLen = 59;  // CAN FD payload budget used by the vendor SDK
  static const std::array<std::array<double, 2>, 16> kAngleLimitsDeg;

  D21Modbus(SlcanSerial& bus, uint8_t slave, double timeout_s = 0.1,
            const D21RegisterMap& map = kD21RegisterMapV2);

  // Raw Modbus.
  std::vector<uint16_t> read_holding(uint16_t address, uint16_t count);
  std::vector<uint16_t> read_input(uint16_t address, uint16_t count);
  void write_single(uint16_t address, uint16_t value);
  void write_multiple(uint16_t address, const std::vector<uint16_t>& values);

  // D21 registers.
  std::array<double, 16> read_positions_deg();   // input 0-15
  std::array<double, 16> read_velocities_raw();  // input 16-31, deg/s (v1 unsigned, v2 signed)
  std::array<double, 16> read_currents_a();      // input 32-47
  std::array<double, 16> read_commanded_positions_deg();  // holding 0-15
  std::array<int, 16> read_torques();            // holding 68-83, mA (-1000..1000)
  // Throws when the hand's position registers do not match this object's map (old firmware never exceeds 2000).
  void check_register_map();
  const D21RegisterMap& register_map() const { return map_; }
  int read_control_mode();                       // holding 21: 1 position, 2 torque
  int read_hand_enable();                        // holding 66
  void write_positions_deg(const std::array<double, 16>& degrees);
  void write_speeds_percent(const std::array<double, 5>& percent);  // holding 16-20
  // mA, -1000..1000; holding 68-83 as two frames: 1-15, 16. v2 assumes 20000 counts = 1 A (unverified).
  void write_torques(const std::array<int, 16>& torques);
  void set_control_mode(int mode);

  // Frame helpers, exposed for tests.
  static uint16_t crc16(const uint8_t* data, size_t length);
  static std::vector<uint8_t> build_request(uint8_t slave, uint8_t function, uint16_t address, uint16_t count_or_value,
                                            const std::vector<uint16_t>* values = nullptr);
  static std::vector<uint8_t> parse_read_response(const std::vector<uint8_t>& response, uint8_t slave, uint8_t function);
  static void check_write_response(const std::vector<uint8_t>& response, const std::vector<uint8_t>& request);
  static uint16_t angle_to_register(int joint, double degrees, int full_scale = kD21RegisterMapV2.full_scale);
  static double register_to_angle(int joint, uint16_t value, int full_scale = kD21RegisterMapV2.full_scale);

 private:
  std::vector<uint8_t> transact(const std::vector<uint8_t>& request);
  SlcanSerial& bus_;
  uint8_t slave_;
  double timeout_s_;
  D21RegisterMap map_;
};

}  // namespace dexrt
