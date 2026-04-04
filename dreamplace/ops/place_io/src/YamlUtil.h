/**
 * @file   YamlUtil.h
 * @brief  Helper functions for YAML PIC circuit processing
 */

#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace YamlParser {

// Define your custom exception class
class ParserException : public std::exception {
 private:
  std::string message;

 public:
  // Constructor to set the custom message
  ParserException(const std::string& msg) : message(msg) {}

  // Override the what() method
  // 'noexcept' is important: it promises this function won't throw an exception itself
  const char* what() const noexcept override { return message.c_str(); }
};

/// Split a net connection string "instance_name,port_name" into pair
inline std::pair<std::string, std::string> splitNetConnection(std::string const& conn) {
  std::string trimmed = conn;
  // remove leading/trailing whitespace and quotes
  while (!trimmed.empty() && (trimmed.front() == ' ' || trimmed.front() == '\'' || trimmed.front() == '"'))
    trimmed.erase(trimmed.begin());
  while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\'' || trimmed.back() == '"'))
    trimmed.pop_back();

  std::size_t pos = trimmed.find(',');
  if (pos == std::string::npos) return std::make_pair(trimmed, std::string());

  std::string instName = trimmed.substr(0, pos);
  std::string portName = trimmed.substr(pos + 1);

  // trim whitespace
  while (!instName.empty() && instName.back() == ' ')
    instName.pop_back();
  while (!portName.empty() && portName.front() == ' ')
    portName.erase(portName.begin());

  return std::make_pair(instName, portName);
}

/// Convert orientation string to standard form (already N/S/E/W/FN/FS/FW/FE)
inline std::string normalizeOrient(std::string const& orient) {
  if (orient == "N" || orient == "S" || orient == "E" || orient == "W" || orient == "FN" || orient == "FS" || orient == "FE" || orient == "FW")
    return orient;
  return "N";  // default
}

/// Convert placement status string to standard form
inline std::string normalizeStatus(std::string const& status) {
  if (status == "PLACED" || status == "FIXED" || status == "UNPLACED") return status;
  return "UNPLACED";  // default
}

}  // namespace YamlParser
