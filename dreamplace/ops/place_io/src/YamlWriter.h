/**
 * @file   YamlWriter.h
 * @brief  Writer to export placement data in YAML PIC circuit format
 */

#pragma once

#include "YamlDataBase.h"
#include <yaml-cpp/yaml.h>
#include <string>
#include <fstream>
#include <vector>

namespace YamlParser {

/// Write YAML output from a YamlDataBase
/// This class collects data and writes it to a YAML file
class YamlWriter
{
public:
    YamlWriter() {}

    void setSettings(YamlSettings const& s) { m_settings = s; }
    void addMacro(YamlMacro const& m) { m_macros.push_back(m); }
    void addInstance(YamlInstance const& inst) { m_instances.push_back(inst); }
    void addNet(YamlNet const& net) { m_nets.push_back(net); }
    void addPort(YamlPort const& port) { m_ports.push_back(port); }
    void addConstraint(YamlConstraint const& c) { m_constraints.push_back(c); }

    bool write(std::string const& filename) const
    {
        YAML::Emitter out;
        out << YAML::BeginMap;

        // instances
        out << YAML::Key << "instances" << YAML::Value << YAML::BeginMap;
        for (auto const& inst : m_instances)
        {
            out << YAML::Key << inst.name << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "component" << YAML::Value << inst.component;
            out << YAML::Key << "settings" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "macro_type" << YAML::Value << inst.macro_type;
            out << YAML::Key << "placement" << YAML::Value << YAML::BeginSeq;
            out << inst.status;
            out << YAML::Flow << YAML::BeginSeq << inst.x << inst.y << YAML::EndSeq;
            out << inst.orient;
            out << YAML::Flow << YAML::BeginSeq << inst.halo_left << inst.halo_down << inst.halo_right << inst.halo_up << YAML::EndSeq;
            out << YAML::EndSeq;
            out << YAML::EndMap; // settings
            out << YAML::EndMap; // instance
        }
        out << YAML::EndMap; // instances

        // nets
        out << YAML::Key << "nets" << YAML::Value << YAML::BeginMap;
        for (auto const& net : m_nets)
        {
            out << YAML::Key << net.name << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (auto const& conn : net.connections)
            {
                out << (conn.first + "," + conn.second);
            }
            out << YAML::EndSeq;
        }
        out << YAML::EndMap; // nets

        // ports
        out << YAML::Key << "ports" << YAML::Value << YAML::BeginMap;
        for (auto const& port : m_ports)
        {
            out << YAML::Key << port.name << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "direction" << YAML::Value << port.direction;
            out << YAML::EndMap;
        }
        out << YAML::EndMap; // ports

        // settings
        out << YAML::Key << "settings" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "version" << YAML::Value << m_settings.version;
        out << YAML::Key << "design" << YAML::Value << m_settings.design;
        out << YAML::Key << "units_distance_microns" << YAML::Value << m_settings.units_distance_microns;
        out << YAML::Key << "die_area" << YAML::Value << YAML::BeginSeq;
        out << YAML::Flow << YAML::BeginSeq << m_settings.die_xl << m_settings.die_yl << YAML::EndSeq;
        out << YAML::Flow << YAML::BeginSeq << m_settings.die_xh << m_settings.die_yh << YAML::EndSeq;
        out << YAML::EndSeq;
        out << YAML::Key << "num_instances" << YAML::Value << m_settings.num_instances;
        out << YAML::Key << "num_nets" << YAML::Value << m_settings.num_nets;
        out << YAML::Key << "num_ports" << YAML::Value << m_settings.num_ports;
        out << YAML::Key << "wg_radius" << YAML::Value << m_settings.wg_radius;
        out << YAML::EndMap; // settings

        // library
        out << YAML::Key << "library" << YAML::Value << YAML::BeginMap;
        for (auto const& macro : m_macros)
        {
            out << YAML::Key << macro.name << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "property" << YAML::Value << YAML::Null;
            out << YAML::Key << "type" << YAML::Value << macro.type;
            out << YAML::Key << "origin" << YAML::Value << YAML::Flow << YAML::BeginSeq << macro.origin_x << macro.origin_y << YAML::EndSeq;
            out << YAML::Key << "size" << YAML::Value << YAML::Flow << YAML::BeginSeq << macro.size_x << macro.size_y << YAML::EndSeq;
            out << YAML::Key << "site" << YAML::Value << macro.site;

            out << YAML::Key << "pins" << YAML::Value << YAML::BeginMap;
            for (auto const& pin : macro.pins)
            {
                out << YAML::Key << pin.name << YAML::Value << YAML::BeginMap;
                out << YAML::Key << "pin_offset_x" << YAML::Value << pin.pin_offset_x;
                out << YAML::Key << "pin_offset_y" << YAML::Value << pin.pin_offset_y;
                out << YAML::Key << "pin_width" << YAML::Value << pin.pin_width;
                out << YAML::Key << "pin_orient" << YAML::Value << pin.pin_orient;
                out << YAML::Key << "pin_layer" << YAML::Value << YAML::Flow << YAML::BeginSeq;
                for (auto l : pin.pin_layer) out << l;
                out << YAML::EndSeq;
                out << YAML::EndMap; // pin
            }
            out << YAML::EndMap; // pins
            out << YAML::EndMap; // macro
        }
        out << YAML::EndMap; // library

        // constraints
        out << YAML::Key << "constraints" << YAML::Value << YAML::BeginMap;
        for (auto const& c : m_constraints)
        {
            out << YAML::Key << c.name << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "type" << YAML::Value << c.type;
            out << YAML::Key << "settings" << YAML::Value << YAML::Flow << YAML::BeginMap;
            out << YAML::Key << "anchor" << YAML::Value << c.anchor;
            out << YAML::EndMap;
            out << YAML::Key << "objects" << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (auto const& obj : c.objects) out << obj;
            out << YAML::EndSeq;
            out << YAML::EndMap; // constraint
        }
        out << YAML::EndMap; // constraints

        out << YAML::EndMap; // root

        std::ofstream fout(filename);
        if (!fout.is_open()) return false;
        fout << out.c_str();
        fout.close();
        return true;
    }

private:
    YamlSettings m_settings;
    std::vector<YamlMacro> m_macros;
    std::vector<YamlInstance> m_instances;
    std::vector<YamlNet> m_nets;
    std::vector<YamlPort> m_ports;
    std::vector<YamlConstraint> m_constraints;
};

} // namespace YamlParser
