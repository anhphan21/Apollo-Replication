/**
 * @file   YamlDriver.h
 * @brief  YAML parser driver for PIC circuit format
 */

#pragma once

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <set>
#include <string>

#include "YamlDataBase.h"
#include "YamlUtil.h"

namespace YamlParser {

/// Parse a YAML PIC circuit file and call callbacks on db
inline bool read(YamlDataBase& db, std::string const& filename) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(filename);
  } catch (YAML::Exception const& e) {
    std::cerr << "YAML parse error: " << e.what() << std::endl;
    return false;
  }

  try {
    YamlSettings settings;
    if (root["settings"] && root["settings"].IsMap()) {
      auto const& s = root["settings"];
      if (s["version"]) settings.version = s["version"].as<std::string>();
      if (s["design"]) settings.design = s["design"].as<std::string>();
      if (s["units_distance_microns"]) settings.units_distance_microns = s["units_distance_microns"].as<float>();
      if (s["die_area"] && s["die_area"].IsSequence()) {
        auto const& die = s["die_area"];
        if (die.size() >= 2 && die[0].IsSequence() && die[1].IsSequence()) {
          settings.die_xl = die[0][0].as<float>();
          settings.die_yl = die[0][1].as<float>();
          settings.die_xh = die[1][0].as<float>();
          settings.die_yh = die[1][1].as<float>();
        }
      }
      if (s["num_instances"]) settings.num_instances = s["num_instances"].as<unsigned int>();
      if (s["num_nets"]) settings.num_nets = s["num_nets"].as<unsigned int>();
      if (s["num_ports"]) settings.num_ports = s["num_ports"].as<unsigned int>();
      if (s["wg_radius"]) settings.wg_radius = s["wg_radius"].as<float>();
    }
    db.yaml_settings_cbk(settings);

    // 2. Parse library (macros) - must be done before instances
    std::set<std::string> siteNames;
    if (root["library"] && root["library"].IsMap()) {
      auto const& lib = root["library"];
      for (auto it : lib) {
        YamlMacro macro;
        macro.name    = it.first.as<std::string>();
        auto const& m = it.second;
        if (!m.IsMap()) {
          std::cerr << "YAML warning: library entry '" << macro.name << "' is not a map, skipping\n";
          continue;
        }

        if (m["type"] && !m["type"].IsNull()) macro.type = m["type"].as<std::string>();
        if (m["origin"] && m["origin"].IsSequence() && m["origin"].size() >= 2) {
          macro.origin_x = m["origin"][0].as<float>();
          macro.origin_y = m["origin"][1].as<float>();
        }
        if (m["size"] && m["size"].IsSequence() && m["size"].size() >= 2) {
          macro.size_x = m["size"][0].as<float>();
          macro.size_y = m["size"][1].as<float>();
        }
        if (m["site"] && !m["site"].IsNull()) {
          macro.site = m["site"].as<std::string>();
          if (!macro.site.empty()) siteNames.insert(macro.site);
        }

        // Parse pins
        if (m["pins"] && m["pins"].IsMap()) {
          auto const& pins = m["pins"];
          for (auto pit : pins) {
            YamlMacroPin pin;
            pin.name      = pit.first.as<std::string>();
            auto const& p = pit.second;
            if (!p.IsMap()) continue;

            if (p["pin_offset_x"]) pin.pin_offset_x = p["pin_offset_x"].as<float>();
            if (p["pin_offset_y"]) pin.pin_offset_y = p["pin_offset_y"].as<float>();
            if (p["pin_width"]) pin.pin_width = p["pin_width"].as<float>();
            if (p["pin_orient"]) pin.pin_orient = p["pin_orient"].as<float>();
            if (p["pin_layer"]) {
              auto const& layer = p["pin_layer"];
              if (layer.IsSequence()) {
                for (std::size_t i = 0; i < layer.size(); ++i)
                  pin.pin_layer.emplace_back(layer[i].as<int>());
              } else if (!layer.IsNull()) {
                pin.pin_layer.emplace_back(layer.as<int>());
              }
            }

            macro.pins.emplace_back(pin);
          }
        }

        db.yaml_macro_cbk(macro);
      }
    }

    // 2b. Initialize site and rows if exactly 1 unique site found in library
    if (siteNames.size() == 1) {
      db.yaml_init_site_row_cbk(*siteNames.begin());
    } else {
      throw ParserException("Design has 2 types of site !");
    }

    // 3. Prepare: reserve space
    db.yaml_prepare_cbk(settings.num_instances, settings.num_nets, settings.num_ports);

    // 4. Parse instances
    if (root["instances"] && root["instances"].IsMap()) {
      auto const& insts = root["instances"];
      for (auto it : insts) {
        YamlInstance inst;
        inst.name     = it.first.as<std::string>();
        auto const& v = it.second;
        if (!v.IsMap()) continue;

        if (v["component"] && !v["component"].IsNull()) inst.component = v["component"].as<std::string>();

        if (v["settings"] && v["settings"].IsMap()) {
          auto const& s = v["settings"];
          if (s["macro_type"] && !s["macro_type"].IsNull()) inst.macro_type = s["macro_type"].as<std::string>();
          if (s["placement"] && s["placement"].IsSequence()) {
            auto const& pl = s["placement"];
            if (pl.size() >= 4) {
              inst.status = normalizeStatus(pl[0].as<std::string>());
              if (pl[1].IsSequence() && pl[1].size() >= 2) {
                inst.x = pl[1][0].as<float>();
                inst.y = pl[1][1].as<float>();
              }
              inst.orient = normalizeOrient(pl[2].as<std::string>());
              if (pl[3].IsSequence() && pl[3].size() >= 4) {
                inst.halo_left  = pl[3][0].as<float>();
                inst.halo_down  = pl[3][1].as<float>();
                inst.halo_right = pl[3][2].as<float>();
                inst.halo_up    = pl[3][3].as<float>();
                inst.has_halo   = true;
              }
            }
          }
        }
        db.yaml_instance_cbk(inst);
      }
    }

    // 5. Parse nets
    if (root["nets"] && root["nets"].IsMap()) {
      auto const& nets = root["nets"];
      for (auto it : nets) {
        YamlNet net;
        net.name          = it.first.as<std::string>();
        auto const& conns = it.second;
        if (!conns.IsSequence()) continue;
        for (std::size_t i = 0; i < conns.size(); ++i) {
          std::string connStr = conns[i].as<std::string>();
          net.connections.push_back(splitNetConnection(connStr));
        }
        db.yaml_net_cbk(net);
      }
    }

    // 6. Parse ports
    if (root["ports"] && root["ports"].IsMap()) {
      auto const& ports = root["ports"];
      for (auto it : ports) {
        YamlPort port;
        port.name = it.first.as<std::string>();
        if (it.second.IsMap()) {
          if (it.second["direction"]) port.direction = it.second["direction"].as<std::string>();
        }
        db.yaml_port_cbk(port);
      }
    }

    // 7. Parse constraints
    if (root["constraints"] && root["constraints"].IsMap()) {
      auto const& constrs = root["constraints"];
      for (auto it : constrs) {
        YamlConstraint constr;
        constr.name   = it.first.as<std::string>();
        auto const& c = it.second;
        if (!c.IsMap()) continue;

        if (c["type"] && !c["type"].IsNull()) constr.type = c["type"].as<std::string>();
        if (c["settings"] && c["settings"].IsMap()) {
          auto const& cs = c["settings"];
          if (cs["anchor"] && !cs["anchor"].IsNull()) constr.anchor = cs["anchor"].as<std::string>();
        }
        if (c["objects"] && c["objects"].IsSequence()) {
          auto const& objs = c["objects"];
          for (std::size_t i = 0; i < objs.size(); ++i)
            constr.objects.push_back(objs[i].as<std::string>());
        }
        db.yaml_constraint_cbk(constr);
      }
    }

  } catch (YAML::Exception const& e) {
    std::cerr << "YAML processing error: " << e.what() << std::endl;
    return false;
  }

  // 8. End
  db.yaml_end_cbk();

  return true;
}

}  // namespace YamlParser
