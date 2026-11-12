// Frame-level checks for the CAN FD/Modbus port; no hardware involved.
#include "check.hpp"
#include "dexrt/d21_canfd.hpp"

int main() {
  using dexrt::D21Modbus;
  using dexrt::SlcanSerial;

  // Modbus CRC16 reference: "01 03 00 00 00 0A" -> C5 CD on the wire (CRC = 0xCDC5).
  const uint8_t ref[] = {0x01, 0x03, 0x00, 0x00, 0x00, 0x0A};
  CHECK(D21Modbus::crc16(ref, sizeof(ref)) == 0xCDC5);

  // Vendor protocol spec, Examples 9 and 10 (torque mode on; +500 on motors 1 and 2), byte for byte.
  CHECK(D21Modbus::build_request(1, 0x06, 21, 2) ==
        std::vector<uint8_t>({0x01, 0x06, 0x00, 0x15, 0x00, 0x02, 0x19, 0xCF}));
  const std::vector<uint16_t> two{1500, 1500};
  CHECK(D21Modbus::build_request(1, 0x10, 68, 2, &two) ==
        std::vector<uint8_t>({0x01, 0x10, 0x00, 0x44, 0x00, 0x02, 0x04, 0x05, 0xDC, 0x05, 0xDC, 0x35, 0x93}));

  // Request layout matches struct.pack('>BBHH') + little-endian CRC.
  const auto read = D21Modbus::build_request(1, 0x04, 0, 16);
  CHECK(read.size() == 8);
  CHECK(read[0] == 1 && read[1] == 0x04 && read[2] == 0 && read[3] == 0 && read[4] == 0 && read[5] == 16);
  const uint16_t crc = D21Modbus::crc16(read.data(), 6);
  CHECK(read[6] == (crc & 0xFF) && read[7] == (crc >> 8));

  const std::vector<uint16_t> values{1000, 2000, 0};
  const auto write = D21Modbus::build_request(1, 0x10, 68, 3, &values);
  CHECK(write.size() == 6 + 1 + 6 + 2);
  CHECK(write[6] == 6 && write[7] == 0x03 && write[8] == 0xE8 && write[9] == 0x07 && write[10] == 0xD0);

  // Read response parsing, including CRC failure and exception replies.
  std::vector<uint8_t> resp{1, 0x04, 4, 0x01, 0x02, 0x03, 0x04};
  const uint16_t rcrc = D21Modbus::crc16(resp.data(), resp.size());
  resp.push_back(rcrc & 0xFF);
  resp.push_back(rcrc >> 8);
  resp.resize(12, 0);  // CAN FD padding after the RTU frame
  const auto data = D21Modbus::parse_read_response(resp, 1, 0x04);
  CHECK(data.size() == 4 && data[0] == 1 && data[3] == 4);
  auto corrupt = resp;
  corrupt[4] ^= 0xFF;
  CHECK_THROWS(D21Modbus::parse_read_response(corrupt, 1, 0x04));
  std::vector<uint8_t> exception{1, 0x84, 0x02};
  const uint16_t ecrc = D21Modbus::crc16(exception.data(), exception.size());
  exception.push_back(ecrc & 0xFF);
  exception.push_back(ecrc >> 8);
  CHECK_THROWS(D21Modbus::parse_read_response(exception, 1, 0x04));

  // Write acknowledgement must echo address/count.
  auto ack = std::vector<uint8_t>(write.begin(), write.begin() + 6);
  const uint16_t acrc = D21Modbus::crc16(ack.data(), 6);
  ack.push_back(acrc & 0xFF);
  ack.push_back(acrc >> 8);
  D21Modbus::check_write_response(ack, write);
  auto wrong = ack;
  wrong[3] = 69;
  CHECK_THROWS(D21Modbus::check_write_response(wrong, write));

  // Angle <-> register mapping matches the vendor tables: new firmware (v2, default) 0..40000, old (v1) 0..2000.
  CHECK(D21Modbus::angle_to_register(1, 0.0) == 20000);   // thumb rotation -35..35
  CHECK(D21Modbus::angle_to_register(5, 90.0) == 40000);  // index mcp 0..90
  CHECK(D21Modbus::angle_to_register(4, -20.0) == 0);     // index abduction -20..20
  CHECK_NEAR(D21Modbus::register_to_angle(1, 20000), 0.0, 1e-12);
  CHECK_NEAR(D21Modbus::register_to_angle(6, 10000), 25.0, 1e-12);
  CHECK_THROWS(D21Modbus::register_to_angle(6, 40001));
  CHECK_THROWS(D21Modbus::angle_to_register(5, 91.0));
  const int v1 = dexrt::kD21RegisterMapV1.full_scale;
  CHECK(D21Modbus::angle_to_register(1, 0.0, v1) == 1000);
  CHECK(D21Modbus::angle_to_register(5, 90.0, v1) == 2000);
  CHECK_NEAR(D21Modbus::register_to_angle(6, 500, v1), 25.0, 1e-12);
  CHECK_THROWS(D21Modbus::register_to_angle(6, 2001, v1));

  // slcan FD framing: 'b' + 3-hex id + DLC code + hex payload padded to the DLC size.
  dexrt::CanFrame frame;
  frame.id = 0x010;
  frame.fd = true;
  frame.brs = true;
  frame.data = read;  // 8 bytes -> DLC 8, no padding
  std::string hex;
  for (uint8_t b : read) {
    hex += "0123456789ABCDEF"[b >> 4];
    hex += "0123456789ABCDEF"[b & 0xF];
  }
  CHECK(SlcanSerial::encode(frame) == "b0108" + hex);
  frame.data = write;  // 15 bytes -> DLC code 10 ('A'), padded to 16 bytes
  const std::string encoded = SlcanSerial::encode(frame);
  CHECK(encoded.size() == 1 + 3 + 1 + 32);
  CHECK(encoded[4] == 'A');
  dexrt::CanFrame decoded;
  CHECK(SlcanSerial::decode(encoded, decoded));
  CHECK(decoded.id == 0x010 && decoded.fd && decoded.brs && decoded.data.size() == 16);
  CHECK(std::equal(write.begin(), write.end(), decoded.data.begin()));
  CHECK(!SlcanSerial::decode("", decoded));
  CHECK(!SlcanSerial::decode("z", decoded));
  CHECK(!SlcanSerial::decode("b02", decoded));
  CHECK(SlcanSerial::len2dlc(15) == 10 && SlcanSerial::dlc2len(10) == 16 && SlcanSerial::len2dlc(9) == 9 &&
        SlcanSerial::dlc2len(9) == 12 && SlcanSerial::len2dlc(64) == 15);
  return dexrt_test::finish("test_modbus");
}
