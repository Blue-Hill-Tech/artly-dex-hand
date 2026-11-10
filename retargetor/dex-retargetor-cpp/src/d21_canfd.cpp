#include "dexrt/d21_canfd.hpp"

#include <asm/termbits.h>  // struct termios2, BOTHER; do not include <termios.h> in this file
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace dexrt {
namespace {

const char* const kHex = "0123456789ABCDEF";

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

bool parse_hex(const std::string& s, size_t begin, size_t count, uint32_t& value) {
  if (begin + count > s.size()) return false;
  value = 0;
  for (size_t i = 0; i < count; ++i) {
    const int v = hex_value(s[begin + i]);
    if (v < 0) return false;
    value = (value << 4) | static_cast<uint32_t>(v);
  }
  return true;
}

std::string bitrate_code(int bitrate) {
  switch (bitrate) {
    case 10000: return "S0";
    case 20000: return "S1";
    case 50000: return "S2";
    case 100000: return "S3";
    case 125000: return "S4";
    case 250000: return "S5";
    case 500000: return "S6";
    case 750000: return "S7";
    case 1000000: return "S8";
    default: throw std::invalid_argument("slcan: unsupported nominal bitrate " + std::to_string(bitrate));
  }
}

std::string data_bitrate_code(int bitrate) {
  switch (bitrate) {
    case 2000000: return "Y2";
    case 5000000: return "Y5";
    default: throw std::invalid_argument("slcan: unsupported data bitrate " + std::to_string(bitrate));
  }
}

}  // namespace

// ---------------------------------------------------------------- SlcanSerial

SlcanSerial::~SlcanSerial() {
  try {
    close();
  } catch (...) {
  }
}

int SlcanSerial::len2dlc(size_t length) {
  static const size_t table[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64};
  for (int dlc = 0; dlc < 16; ++dlc) {
    if (table[dlc] >= length) return dlc;
  }
  return 15;
}

size_t SlcanSerial::dlc2len(int dlc) {
  static const size_t table[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64};
  if (dlc < 0 || dlc > 15) throw std::invalid_argument("slcan: DLC out of range");
  return table[dlc];
}

void SlcanSerial::open(const std::string& device, int tty_baudrate, int nominal_bitrate, int data_bitrate,
                       double sleep_after_open_s) {
  if (is_open()) throw std::runtime_error("slcan: already open");
  const std::string nominal = bitrate_code(nominal_bitrate);
  const std::string data = data_bitrate_code(data_bitrate);
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (fd_ < 0) throw std::runtime_error("slcan: cannot open " + device + ": " + std::strerror(errno));
  try {
    struct termios2 tio;
    std::memset(&tio, 0, sizeof(tio));
    if (ioctl(fd_, TCGETS2, &tio) != 0) throw std::runtime_error("slcan: TCGETS2 failed");
    tio.c_cflag &= ~(CBAUD | CSIZE | PARENB | CSTOPB | CRTSCTS);
    tio.c_cflag |= BOTHER | CS8 | CREAD | CLOCAL;
    tio.c_iflag = 0;
    tio.c_oflag = 0;
    tio.c_lflag = 0;
    tio.c_ispeed = tio.c_ospeed = static_cast<unsigned>(tty_baudrate);
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;
    if (ioctl(fd_, TCSETS2, &tio) != 0) throw std::runtime_error("slcan: TCSETS2 failed");
    if (sleep_after_open_s > 0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(sleep_after_open_s));
    }
    ioctl(fd_, TCFLSH, TCIFLUSH);
    buffer_.clear();
    // python-can: close channel, set nominal and data bitrates, open channel.
    write_line("C", 0.5);
    write_line(nominal, 0.5);
    write_line(data, 0.5);
    write_line("O", 0.5);
  } catch (...) {
    ::close(fd_);
    fd_ = -1;
    throw;
  }
}

void SlcanSerial::close() {
  if (!is_open()) return;
  try {
    write_line("C", 0.5);
  } catch (...) {
  }
  ::close(fd_);
  fd_ = -1;
  buffer_.clear();
}

void SlcanSerial::write_line(const std::string& line, double timeout_s) {
  if (!is_open()) throw std::runtime_error("slcan: not open");
  const std::string out = line + "\r";
  size_t done = 0;
  const double deadline = now_s() + timeout_s;
  while (done < out.size()) {
    const ssize_t n = ::write(fd_, out.data() + done, out.size() - done);
    if (n > 0) {
      done += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      throw std::runtime_error(std::string("slcan: write failed: ") + std::strerror(errno));
    }
    const double remaining = deadline - now_s();
    if (remaining <= 0) throw std::runtime_error("slcan: serial write deadline expired");
    pollfd pfd{fd_, POLLOUT, 0};
    ::poll(&pfd, 1, static_cast<int>(std::ceil(remaining * 1000)));
  }
}

std::string SlcanSerial::encode(const CanFrame& frame) {
  if (frame.id > 0x7FF) throw std::invalid_argument("slcan: only standard 11-bit IDs are supported");
  std::string s;
  if (frame.fd) {
    if (frame.data.size() > 64) throw std::invalid_argument("slcan: FD payload exceeds 64 bytes");
    s += frame.brs ? 'b' : 'd';
  } else {
    if (frame.data.size() > 8) throw std::invalid_argument("slcan: classic payload exceeds 8 bytes");
    s += 't';
  }
  s += kHex[(frame.id >> 8) & 0xF];
  s += kHex[(frame.id >> 4) & 0xF];
  s += kHex[frame.id & 0xF];
  // Pad after the payload to the next FD DLC size, never inside it (matches the vendor transport).
  std::vector<uint8_t> data = frame.data;
  if (frame.fd) data.resize(dlc2len(len2dlc(data.size())), 0);
  s += kHex[frame.fd ? len2dlc(data.size()) : data.size()];
  for (uint8_t b : data) {
    s += kHex[b >> 4];
    s += kHex[b & 0xF];
  }
  return s;
}

bool SlcanSerial::decode(const std::string& line, CanFrame& frame) {
  if (line.empty()) return false;
  const char kind = line[0];
  size_t id_len = 0;
  bool fd = false, brs = false, extended = false;
  switch (kind) {
    case 't': id_len = 3; break;
    case 'T': id_len = 8; extended = true; break;
    case 'd': id_len = 3; fd = true; break;
    case 'D': id_len = 8; fd = true; extended = true; break;
    case 'b': id_len = 3; fd = true; brs = true; break;
    case 'B': id_len = 8; fd = true; brs = true; extended = true; break;
    default: return false;  // remote frames, acknowledgements, status replies
  }
  uint32_t id = 0, dlc = 0;
  if (!parse_hex(line, 1, id_len, id) || !parse_hex(line, 1 + id_len, 1, dlc)) return false;
  const size_t length = fd ? dlc2len(static_cast<int>(dlc)) : dlc;
  if (!fd && length > 8) return false;
  std::vector<uint8_t> data(length);
  for (size_t i = 0; i < length; ++i) {
    uint32_t byte = 0;
    if (!parse_hex(line, 2 + id_len + 2 * i, 2, byte)) return false;
    data[i] = static_cast<uint8_t>(byte);
  }
  if (extended) return false;  // the hand never uses extended IDs
  frame.id = id;
  frame.fd = fd;
  frame.brs = brs;
  frame.data = std::move(data);
  return true;
}

void SlcanSerial::send(const CanFrame& frame, double timeout_s) { write_line(encode(frame), timeout_s); }

bool SlcanSerial::recv(CanFrame& frame, double timeout_s) {
  if (!is_open()) throw std::runtime_error("slcan: not open");
  const double deadline = now_s() + timeout_s;
  char chunk[512];
  for (;;) {
    // Deliver a complete line from the buffer first.
    const size_t end = buffer_.find('\r');
    if (end != std::string::npos) {
      std::string line = buffer_.substr(0, end);
      buffer_.erase(0, end + 1);
      line.erase(std::remove(line.begin(), line.end(), '\a'), line.end());
      if (decode(line, frame)) return true;
      continue;
    }
    const double remaining = deadline - now_s();
    if (remaining <= 0) return false;
    pollfd pfd{fd_, POLLIN, 0};
    const int ready = ::poll(&pfd, 1, static_cast<int>(std::ceil(remaining * 1000)));
    if (ready < 0 && errno != EINTR) throw std::runtime_error(std::string("slcan: poll failed: ") + std::strerror(errno));
    if (ready <= 0) continue;
    const ssize_t n = ::read(fd_, chunk, sizeof(chunk));
    if (n > 0) {
      buffer_.append(chunk, static_cast<size_t>(n));
    } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      throw std::runtime_error(std::string("slcan: read failed: ") + std::strerror(errno));
    } else if (n == 0) {
      throw std::runtime_error("slcan: serial device disconnected");
    }
    if (buffer_.size() > 1 << 16) buffer_.clear();  // runaway garbage without terminators
  }
}

// ------------------------------------------------------------------ D21Modbus

const std::array<std::array<double, 2>, 16> D21Modbus::kAngleLimitsDeg = {{
    {0, 100}, {-35, 35}, {0, 50}, {0, 70},          // thumb abduction, rotation, cmc, mcp
    {-20, 20}, {0, 90}, {0, 100},                   // index abduction, mcp, pip
    {-20, 20}, {0, 90}, {0, 100},                   // middle
    {-20, 20}, {0, 90}, {0, 100},                   // ring
    {-20, 20}, {0, 90}, {0, 100},                   // pinky
}};

const D21RegisterMap& d21_register_map_from_env() {
  const char* value = std::getenv("D21_REGISTER_MAP");
  if (value == nullptr || std::string(value).empty() || std::string(value) == "v2") return kD21RegisterMapV2;
  if (std::string(value) == "v1") return kD21RegisterMapV1;
  throw std::invalid_argument("d21: D21_REGISTER_MAP must be v1 or v2");
}

D21Modbus::D21Modbus(SlcanSerial& bus, uint8_t slave, double timeout_s, const D21RegisterMap& map)
    : bus_(bus), slave_(slave), timeout_s_(timeout_s), map_(map) {
  if (slave < 1 || slave > 254) throw std::invalid_argument("d21: slave address must be in [1, 254]");
  if (!(timeout_s > 0)) throw std::invalid_argument("d21: timeout must be positive");
}

uint16_t D21Modbus::crc16(const uint8_t* data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1) ? static_cast<uint16_t>((crc >> 1) ^ 0xA001) : static_cast<uint16_t>(crc >> 1);
    }
  }
  return crc;
}

std::vector<uint8_t> D21Modbus::build_request(uint8_t slave, uint8_t function, uint16_t address, uint16_t count_or_value,
                                              const std::vector<uint16_t>* values) {
  std::vector<uint8_t> req{slave, function, static_cast<uint8_t>(address >> 8), static_cast<uint8_t>(address & 0xFF),
                           static_cast<uint8_t>(count_or_value >> 8), static_cast<uint8_t>(count_or_value & 0xFF)};
  if (values != nullptr) {
    req.push_back(static_cast<uint8_t>(values->size() * 2));
    for (uint16_t v : *values) {
      req.push_back(static_cast<uint8_t>(v >> 8));
      req.push_back(static_cast<uint8_t>(v & 0xFF));
    }
  }
  const uint16_t crc = crc16(req.data(), req.size());
  req.push_back(static_cast<uint8_t>(crc & 0xFF));  // CRC is little-endian on the wire
  req.push_back(static_cast<uint8_t>(crc >> 8));
  return req;
}

std::vector<uint8_t> D21Modbus::parse_read_response(const std::vector<uint8_t>& resp, uint8_t slave, uint8_t function) {
  if (resp.size() < 5) throw std::runtime_error("d21: response header incomplete");
  if (resp[0] != slave) throw std::runtime_error("d21: slave address mismatch");
  if (resp[1] == (function | 0x80)) {
    throw std::runtime_error("d21: Modbus exception, error code " + std::to_string(resp.size() > 3 ? resp[2] : 0));
  }
  if (resp[1] != function) throw std::runtime_error("d21: function code mismatch");
  const size_t total = 3 + resp[2];
  if (resp.size() < total + 2) throw std::runtime_error("d21: data incomplete");
  const uint16_t received = static_cast<uint16_t>(resp[total] | (resp[total + 1] << 8));
  if (received != crc16(resp.data(), total)) throw std::runtime_error("d21: CRC check failed");
  return std::vector<uint8_t>(resp.begin() + 3, resp.begin() + static_cast<long>(total));
}

void D21Modbus::check_write_response(const std::vector<uint8_t>& resp, const std::vector<uint8_t>& req) {
  if (resp.size() < 5) throw std::runtime_error("d21: write response header incomplete");
  if (resp[0] != req[0]) throw std::runtime_error("d21: slave address mismatch");
  const uint8_t function = req[1];
  const bool exception = resp[1] == (function | 0x80);
  if (!exception && resp[1] != function) throw std::runtime_error("d21: function code mismatch");
  const size_t size = exception ? 5 : 8;
  if (resp.size() < size) throw std::runtime_error("d21: write response incomplete");
  const uint16_t received = static_cast<uint16_t>(resp[size - 2] | (resp[size - 1] << 8));
  if (received != crc16(resp.data(), size - 2)) throw std::runtime_error("d21: CRC check failed");
  if (exception) throw std::runtime_error("d21: Modbus exception, error code " + std::to_string(resp[2]));
  if (!std::equal(resp.begin() + 2, resp.begin() + 6, req.begin() + 2)) {
    throw std::runtime_error("d21: write acknowledgement echo mismatch");
  }
}

std::vector<uint8_t> D21Modbus::transact(const std::vector<uint8_t>& request) {
  const double deadline = now_s() + timeout_s_;
  CanFrame frame;
  frame.id = kRequestId;
  frame.fd = true;
  frame.brs = true;
  frame.data = request;
  bus_.send(frame, timeout_s_);
  const uint8_t function = request[1];
  for (;;) {
    const double remaining = deadline - now_s();
    if (remaining <= 0) throw std::runtime_error("d21: Modbus response timeout");
    CanFrame reply;
    if (!bus_.recv(reply, remaining)) throw std::runtime_error("d21: Modbus response timeout");
    // Only the matching response counts; periodic/timestamp frames are ignored like the vendor SDK does.
    if (reply.id != kResponseId || reply.data.size() < 5 || reply.data[0] != slave_) continue;
    if (reply.data[1] != function && reply.data[1] != (function | 0x80)) continue;
    return reply.data;
  }
}

std::vector<uint16_t> D21Modbus::read_holding(uint16_t address, uint16_t count) {
  const uint16_t max_count = kMaxResponseLen / 2;
  std::vector<uint16_t> out;
  for (uint16_t off = 0; off < count;) {
    const uint16_t n = std::min<uint16_t>(count - off, max_count);
    const auto req = build_request(slave_, 0x03, address + off, n);
    const auto data = parse_read_response(transact(req), slave_, 0x03);
    for (size_t i = 0; i + 1 < data.size(); i += 2) out.push_back(static_cast<uint16_t>((data[i] << 8) | data[i + 1]));
    off += n;
  }
  return out;
}

std::vector<uint16_t> D21Modbus::read_input(uint16_t address, uint16_t count) {
  const uint16_t max_count = kMaxResponseLen / 2;
  std::vector<uint16_t> out;
  for (uint16_t off = 0; off < count;) {
    const uint16_t n = std::min<uint16_t>(count - off, max_count);
    const auto req = build_request(slave_, 0x04, address + off, n);
    const auto data = parse_read_response(transact(req), slave_, 0x04);
    for (size_t i = 0; i + 1 < data.size(); i += 2) out.push_back(static_cast<uint16_t>((data[i] << 8) | data[i + 1]));
    off += n;
  }
  return out;
}

void D21Modbus::write_single(uint16_t address, uint16_t value) {
  const auto req = build_request(slave_, 0x06, address, value);
  check_write_response(transact(req), req);
}

void D21Modbus::write_multiple(uint16_t address, const std::vector<uint16_t>& values) {
  if (values.empty() || values.size() > 123) throw std::invalid_argument("d21: invalid register count");
  const auto req = build_request(slave_, 0x10, address, static_cast<uint16_t>(values.size()), &values);
  check_write_response(transact(req), req);
}

uint16_t D21Modbus::angle_to_register(int joint, double degrees, int full_scale) {
  if (joint < 0 || joint > 15) throw std::invalid_argument("d21: joint index out of range");
  const double lo = kAngleLimitsDeg[joint][0], hi = kAngleLimitsDeg[joint][1];
  if (!std::isfinite(degrees) || degrees < lo || degrees > hi) {
    throw std::invalid_argument("d21: joint " + std::to_string(joint) + " angle out of allowed range");
  }
  const long reg = std::lround((degrees - lo) / (hi - lo) * full_scale);
  return static_cast<uint16_t>(std::clamp<long>(reg, 0, full_scale));
}

double D21Modbus::register_to_angle(int joint, uint16_t value, int full_scale) {
  if (joint < 0 || joint > 15) throw std::invalid_argument("d21: joint index out of range");
  if (value > full_scale) throw std::runtime_error("d21: joint register out of range");
  const double lo = kAngleLimitsDeg[joint][0], hi = kAngleLimitsDeg[joint][1];
  return lo + static_cast<double>(value) / full_scale * (hi - lo);
}

void D21Modbus::check_register_map() {
  const auto raw = read_input(0, 16);
  uint16_t top = 0;
  for (uint16_t v : raw) top = std::max(top, v);
  const D21RegisterMap& detected = top <= kD21RegisterMapV1.full_scale ? kD21RegisterMapV1 : kD21RegisterMapV2;
  if (detected.full_scale != map_.full_scale) {
    throw std::runtime_error(std::string("d21: hand reports ") + detected.name + " position registers but the map is " +
                             map_.name + (detected.full_scale == kD21RegisterMapV1.full_scale
                                              ? ": flash the new firmware or set D21_REGISTER_MAP=v1"
                                              : ": unset D21_REGISTER_MAP"));
  }
}

std::array<double, 16> D21Modbus::read_positions_deg() {
  const auto raw = read_input(0, 16);
  if (raw.size() != 16) throw std::runtime_error("d21: expected 16 position registers");
  std::array<double, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = register_to_angle(i, raw[i], map_.full_scale);
  return out;
}

std::array<double, 16> D21Modbus::read_velocities_raw() {
  const auto raw = read_input(16, 16);
  if (raw.size() != 16) throw std::runtime_error("d21: expected 16 velocity registers");
  std::array<double, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = (static_cast<int>(raw[i]) - map_.velocity_offset) * map_.velocity_scale;
  return out;
}

std::array<double, 16> D21Modbus::read_currents_a() {
  const auto raw = read_input(32, 16);
  if (raw.size() != 16) throw std::runtime_error("d21: expected 16 current registers");
  std::array<double, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = (static_cast<int>(raw[i]) - map_.signed_zero) / static_cast<double>(map_.signed_zero);
  return out;
}

std::array<double, 16> D21Modbus::read_commanded_positions_deg() {
  const auto raw = read_holding(0, 16);
  if (raw.size() != 16) throw std::runtime_error("d21: expected 16 commanded position registers");
  std::array<double, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = register_to_angle(i, raw[i], map_.full_scale);
  return out;
}

std::array<int, 16> D21Modbus::read_torques() {
  const auto raw = read_holding(68, 16);
  if (raw.size() != 16) throw std::runtime_error("d21: expected 16 torque registers");
  std::array<int, 16> out{};
  const double per_ma = map_.signed_zero / 1000.0;
  for (int i = 0; i < 16; ++i) out[i] = static_cast<int>(std::lround((static_cast<int>(raw[i]) - map_.signed_zero) / per_ma));
  return out;
}

int D21Modbus::read_control_mode() { return read_holding(21, 1).at(0); }
int D21Modbus::read_hand_enable() { return read_holding(66, 1).at(0); }

void D21Modbus::write_positions_deg(const std::array<double, 16>& degrees) {
  std::vector<uint16_t> regs(16);
  for (int i = 0; i < 16; ++i) regs[i] = angle_to_register(i, degrees[i], map_.full_scale);
  write_multiple(0, regs);
}

void D21Modbus::write_speeds_percent(const std::array<double, 5>& percent) {
  std::vector<uint16_t> regs(5);
  for (int i = 0; i < 5; ++i) {
    if (!std::isfinite(percent[i]) || percent[i] < 0 || percent[i] > 100) {
      throw std::invalid_argument("d21: speed percentage out of [0, 100]");
    }
    regs[i] = static_cast<uint16_t>(
        std::clamp<long>(std::lround(percent[i] / 100.0 * map_.full_scale), 0, map_.full_scale));
  }
  write_multiple(16, regs);
}

void D21Modbus::write_torques(const std::array<int, 16>& torques) {
  // Old firmware (v1) stores but does not act on one write covering all 16 torque registers; writes of
  // up to 15 registers work anywhere in 68-83 (bench tests 2026-10-23): motors 1-15 (keeps every
  // differential pair together), then motor 16. New firmware (v2) acts on a single 16-register write
  // (bench over RS-485, 2026-11-02), which saves one round trip per tick.
  static const std::vector<std::pair<int, int>> kGroupsV1 = {{1, 15}, {16, 1}};
  static const std::vector<std::pair<int, int>> kGroupsV2 = {{1, 16}};
  const auto& kGroups = map_.full_scale == kD21RegisterMapV1.full_scale ? kGroupsV1 : kGroupsV2;
  const double per_ma = map_.signed_zero / 1000.0;  // v1 1 count/mA (bench); v2 20 counts/mA (assumed)
  const int thumb_ma = std::min(1000, static_cast<int>(map_.thumb_torque_max / per_ma));  // motors 1-4
  for (int i = 0; i < 16; ++i) {
    const int limit = i < 4 ? thumb_ma : 1000;
    if (torques[i] < -limit || torques[i] > limit) {
      throw std::invalid_argument("d21: torque for motor " + std::to_string(i + 1) + " out of [-" +
                                  std::to_string(limit) + ", " + std::to_string(limit) + "] mA");
    }
  }
  for (const auto& [first, count] : kGroups) {
    std::vector<uint16_t> regs;
    for (int m = first; m < first + count; ++m) {
      regs.push_back(static_cast<uint16_t>(std::lround(torques[m - 1] * per_ma) + map_.signed_zero));
    }
    write_multiple(static_cast<uint16_t>(67 + first), regs);
  }
}

void D21Modbus::set_control_mode(int mode) {
  if (mode != 1 && mode != 2) throw std::invalid_argument("d21: control mode must be 1 or 2");
  write_single(21, static_cast<uint16_t>(mode));
}

}  // namespace dexrt
