/**
 * @file   YamlDataBase.h
 * @brief  Database for YAML PIC circuit format
 */

#pragma once

#include <string>
#include <vector>
#include <map>

namespace YamlParser {

/// Data structures for YAML PIC circuit parsing

/// Instance placement info
struct YamlInstance
{
    std::string name;           ///< instance name
    std::string component;      ///< component type
    std::string macro_type;     ///< macro type reference
    std::string status;         ///< placement status: PLACED/FIXED/UNPLACED
    float x;                    ///< placement x
    float y;                    ///< placement y
    std::string orient;         ///< orientation: N/S/E/W/FN/FS/FW/FE
    float halo_left;            ///< halo left
    float halo_down;            ///< halo down
    float halo_right;           ///< halo right
    float halo_up;              ///< halo up

    YamlInstance()
        : x(0), y(0)
        , halo_left(0), halo_down(0), halo_right(0), halo_up(0)
    {}
};

/// Pin info within a macro
struct YamlMacroPin
{
    std::string name;           ///< pin name
    float pin_offset_x;         ///< pin offset x from origin
    float pin_offset_y;         ///< pin offset y from origin
    float pin_width;            ///< pin width (waveguide width)
    float pin_orient;           ///< pin orientation angle (0.0, 90.0, 180.0, 270.0)
    std::vector<int> pin_layer; ///< pin layer info [layer, type]

    YamlMacroPin()
        : pin_offset_x(0), pin_offset_y(0)
        , pin_width(0), pin_orient(0)
    {}
};

/// Library macro info
struct YamlMacro
{
    std::string name;           ///< macro name
    std::string type;           ///< macro type (CORE, etc.)
    float origin_x;             ///< origin x
    float origin_y;             ///< origin y
    float size_x;               ///< width
    float size_y;               ///< height
    std::string site;           ///< site name
    std::vector<YamlMacroPin> pins; ///< pins

    YamlMacro()
        : origin_x(0), origin_y(0)
        , size_x(0), size_y(0)
    {}
};

/// Net connection: pairs of (instance_name, port_name)
struct YamlNet
{
    std::string name;
    std::vector<std::pair<std::string, std::string>> connections; ///< (instance_name, port_name)
};

/// Port (I/O of the circuit)
struct YamlPort
{
    std::string name;
    std::string direction; ///< INPUT/OUTPUT/INOUT
};

/// Settings from the YAML
struct YamlSettings
{
    std::string version;
    std::string design;
    float units_distance_microns;
    float die_xl;
    float die_yl;
    float die_xh;
    float die_yh;
    unsigned int num_instances;
    unsigned int num_nets;
    unsigned int num_ports;
    float wg_radius;

    YamlSettings()
        : units_distance_microns(1)
        , die_xl(0), die_yl(0), die_xh(0), die_yh(0)
        , num_instances(0), num_nets(0), num_ports(0)
        , wg_radius(0)
    {}
};

/// Constraint info
struct YamlConstraint
{
    std::string name;           ///< constraint name
    std::string type;           ///< constraint type: alignment/uniform
    std::string anchor;         ///< settings anchor: left/right/bottom/up/lower for alignment, vertical/horizontal for uniform
    std::vector<std::string> objects; ///< list of instance names
};

/// Base class for YAML database callbacks
/// PlaceDB inherits from this and overrides the callbacks
class YamlDataBase
{
public:
    virtual ~YamlDataBase() {}

    /// Callback: set design settings
    virtual void yaml_settings_cbk(YamlSettings const& /*settings*/) {}
    /// Callback: add a library macro
    virtual void yaml_macro_cbk(YamlMacro const& /*macro*/) {}
    /// Callback: initialize site and rows after library parsing (called when exactly 1 unique site is found)
    virtual void yaml_init_site_row_cbk(std::string const& /*siteName*/) {}
    /// Callback: resize and prepare for instances/nets
    virtual void yaml_prepare_cbk(unsigned int /*numInstances*/, unsigned int /*numNets*/, unsigned int /*numPorts*/) {}
    /// Callback: add an instance
    virtual void yaml_instance_cbk(YamlInstance const& /*inst*/) {}
    /// Callback: add a net
    virtual void yaml_net_cbk(YamlNet const& /*net*/) {}
    /// Callback: add a port
    virtual void yaml_port_cbk(YamlPort const& /*port*/) {}
    /// Callback: add a constraint
    virtual void yaml_constraint_cbk(YamlConstraint const& /*constr*/) {}
    /// Callback: end of YAML parsing
    virtual void yaml_end_cbk() {}
};

} // namespace YamlParser
