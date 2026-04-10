/**
 * @file   YamlDatabase.cpp
 * @brief  PlaceDB callback implementations for YAML PIC circuit format
 */

#include "PlaceDB.h"

DREAMPLACE_BEGIN_NAMESPACE

void PlaceDB::yaml_settings_cbk(YamlParser::YamlSettings const& settings) {
  m_designName = settings.design;
  m_defUnit    = static_cast<int>(settings.units_distance_microns);
  if (m_lefUnit == 0) m_lefUnit = m_defUnit;

  // set die area - use float coordinates directly
  m_dieArea.set(static_cast<coordinate_type>(settings.die_xl),
                static_cast<coordinate_type>(settings.die_yl),
                static_cast<coordinate_type>(settings.die_xh),
                static_cast<coordinate_type>(settings.die_yh));

  dreamplacePrint(kINFO, "YAML design: %s\n", settings.design.c_str());
  dreamplacePrint(kINFO,
                  "YAML die_area: (%g, %g) - (%g, %g)\n",
                  (double) settings.die_xl,
                  (double) settings.die_yl,
                  (double) settings.die_xh,
                  (double) settings.die_yh);
  dreamplacePrint(kINFO, "YAML wg_radius: %g\n", (double) settings.wg_radius);
}

void PlaceDB::yaml_macro_cbk(YamlParser::YamlMacro const& macro) {
  // create and add macro
  std::pair<index_type, bool> insertMacroRet = addMacro(macro.name);
  if (!insertMacroRet.second) {
    dreamplacePrint(kWARN, "duplicate macro in YAML: %s\n", macro.name.c_str());
    return;
  }

  Macro& m = m_vMacro.at(insertMacroRet.first);
  m.setClassName(macro.type);
  m.setSiteName(macro.site);
  m.setInitOrigin(static_cast<coordinate_type>(macro.origin_x), static_cast<coordinate_type>(macro.origin_y));

  // set bounding box (origin is at 0,0 after init)
  m.set(kXLOW, 0).set(kYLOW, 0);
  m.set(kXHIGH, static_cast<coordinate_type>(macro.size_x));
  m.set(kYHIGH, static_cast<coordinate_type>(macro.size_y));

  // add macro pins
  for (auto const& yamlPin : macro.pins) {
    std::pair<index_type, bool> insertPinRet = m.addMacroPin(yamlPin.name);
    if (!insertPinRet.second) {
      dreamplacePrint(kWARN, "duplicate macro pin in YAML: %s.%s\n", macro.name.c_str(), yamlPin.name.c_str());
      continue;
    }

    MacroPin& mPin = m.macroPin(insertPinRet.first);
    mPin.setDirect(SignalDirectEnum::INOUT);  // PIC pins are bidirectional by default
    mPin.setPortOrientAngle(yamlPin.pin_orient);  // store port orientation from YAML library

    // create a port with offset-based bounding box
    index_type macroPortId = mPin.addMacroPort();
    MacroPort& macroPort   = mPin.macroPort(macroPortId);

    // set pin bounding box from offset
    coordinate_type ox = static_cast<coordinate_type>(yamlPin.pin_offset_x);
    coordinate_type oy = static_cast<coordinate_type>(yamlPin.pin_offset_y);
    coordinate_type hw = static_cast<coordinate_type>(yamlPin.pin_width / 2.0f);

    Box<coordinate_type> pinBox(ox - hw, oy - hw, ox + hw, oy + hw);
    macroPort.boxes().push_back(pinBox);
    deriveMacroPortBbox(macroPort);
    deriveMacroPinBbox(mPin);
  }

  dreamplacePrint(
      kDEBUG, "YAML macro: %s, size=(%g, %g), pins=%lu\n", macro.name.c_str(), (double) macro.size_x, (double) macro.size_y, macro.pins.size());
}

void PlaceDB::yaml_init_site_row_cbk(std::string const& siteName) {
  // create the site with width=1, height=1
  if (m_mSiteName2Index.find(siteName) == m_mSiteName2Index.end()) {
    m_vSite.emplace_back();
    Site& site = m_vSite.back();
    site.setId(m_vSite.size() - 1);
    site.setName(siteName);
    site.setSize(kX, 1);
    site.setSize(kY, 1);
    m_mSiteName2Index[siteName] = site.id();
    m_vSiteUsedCount.emplace_back(1);
    m_coreSiteId = site.id();
  }

  dreamplacePrint(kINFO, "YAML: single site '%s' found in library (width=1, height=1)\n", siteName.c_str());

  // generate placement rows from die area with row height = 1 unit
  if (m_vRow.empty()) {
    coordinate_type rowHeight = siteHeight();  // 1 unit
    coordinate_type siteW     = siteWidth();   // 1 unit
    coordinate_type dieXL     = m_dieArea.xl();
    coordinate_type dieYL     = m_dieArea.yl();
    coordinate_type dieXH     = m_dieArea.xh();
    coordinate_type dieYH     = m_dieArea.yh();

    index_type numRows = static_cast<index_type>((dieYH - dieYL) / rowHeight);
    for (index_type i = 0; i < numRows; ++i) {
      m_vRow.emplace_back();
      Row& row = m_vRow.back();
      row.setId(i);
      row.setName("yaml_row_" + std::to_string(i));
      row.setMacroName(siteName);
      row.set(dieXL, dieYL + i * rowHeight, dieXH, dieYL + (i + 1) * rowHeight);
      row.setStep(siteW, rowHeight);
      m_rowBbox.encompass(row);
    }

    dreamplacePrint(kINFO, "YAML: generated %lu rows (height=%g, site_width=%g) from die area\n",
        m_vRow.size(), (double)rowHeight, (double)siteW);
  }
}

void PlaceDB::yaml_prepare_cbk(unsigned int numInstances, unsigned int numNets, unsigned int numPorts) {
  dreamplacePrint(kINFO, "YAML prepare: %u instances, %u nets, %u ports\n", numInstances, numNets, numPorts);
  prepare(0, numInstances, numPorts, numNets, 0);
}

void PlaceDB::yaml_instance_cbk(YamlParser::YamlInstance const& inst) {
  // find macro
  string2index_map_type::const_iterator foundMacro = m_mMacroName2Index.find(inst.macro_type);
  if (foundMacro == m_mMacroName2Index.end()) {
    dreamplacePrint(kERROR, "YAML instance %s: macro_type %s not found in library\n", inst.name.c_str(), inst.macro_type.c_str());
    return;
  }

  // create and add node
  std::pair<index_type, bool> insertRet = addNode(inst.name);
  if (!insertRet.second) {
    dreamplacePrint(kWARN, "duplicate instance in YAML: %s\n", inst.name.c_str());
    return;
  }

  Node& node             = m_vNode.at(insertRet.first);
  NodeProperty& property = m_vNodeProperty.at(node.id());
  property.setMacroId(foundMacro->second);
  Macro const& macro = m_vMacro.at(property.macroId());

  // set position and size
  coordinate_type px = static_cast<coordinate_type>(inst.x);
  coordinate_type py = static_cast<coordinate_type>(inst.y);
  node.set(px, py, px + macro.width(), py + macro.height());

  // set status
  if (inst.status == "FIXED")
    node.setStatus(PlaceStatusEnum::FIXED);
  else if (inst.status == "PLACED")
    node.setStatus(PlaceStatusEnum::PLACED);
  else
    node.setStatus(PlaceStatusEnum::UNPLACED);

  // set orientation
  if (!inst.orient.empty()) node.setOrient(Orient(inst.orient));

  // set init position
  if (node.status() == PlaceStatusEnum::FIXED || node.status() == PlaceStatusEnum::PLACED) node.setInitPos(ll(node));

  // derive multi-row attribute
  deriveMultiRowAttr(node);

  // update statistics
  if (node.status() == PlaceStatusEnum::FIXED) {
    m_numFixed += 1;
    m_vFixedNodeIndex.emplace_back(node.id());
    std::cout << "Add node to fixed: " << m_vFixedNodeIndex.size() << "\n";
  } else {
    m_numMovable += 1;
    m_vMovableNodeIndex.emplace_back(node.id());
    std::cout << "Add node to move: " << m_vMovableNodeIndex.size() << "\n";
  }

  // reserve space for pins
  node.pins().reserve(macro.macroPins().size());
}

void PlaceDB::yaml_net_cbk(YamlParser::YamlNet const& yamlNet) {
  // check validity
  bool ignoreFlag = false;
  if (yamlNet.connections.size() < 2)
    ignoreFlag = true;
  else {
    bool all_same_node = true;
    for (std::size_t i = 1; i < yamlNet.connections.size(); ++i) {
      if (yamlNet.connections[i - 1].first != yamlNet.connections[i].first) {
        all_same_node = false;
        break;
      }
    }
    if (all_same_node) ignoreFlag = true;
  }

  // create net
  std::pair<index_type, bool> insertNetRet = addNet(yamlNet.name);
  if (!insertNetRet.second) {
    m_vDuplicateNet.push_back(yamlNet.name);
    return;
  }
  Net& net = m_vNet.at(insertNetRet.first);

  if (ignoreFlag) {
    m_vNetIgnoreFlag[net.id()] = true;
    m_numIgnoredNet += 1;
  }

  net.pins().reserve(yamlNet.connections.size());
  for (auto const& conn : yamlNet.connections) {
    std::string const& instName = conn.first;
    std::string const& portName = conn.second;

    // find node
    string2index_map_type::const_iterator foundNode = m_mNodeName2Index.find(instName);
    if (foundNode == m_mNodeName2Index.end()) {
      dreamplacePrint(kWARN, "YAML net %s: instance %s not found\n", yamlNet.name.c_str(), instName.c_str());
      continue;
    }

    Node& node = m_vNode[foundNode->second];

    // find macro pin and add pin
    addPin(portName, net, node, instName);
  }
}

void PlaceDB::yaml_port_cbk(YamlParser::YamlPort const& port) {
  // create virtual macro for the port
  std::pair<index_type, bool> insertMacroRet = addMacro(port.name);
  if (!insertMacroRet.second) return;
  Macro& macro = m_vMacro.at(insertMacroRet.first);
  macro.setClassName("DREAMPlace.IOPin");
  macro.set(kXLOW, 0).set(kYLOW, 0);
  macro.set(kXHIGH, 0).set(kYHIGH, 0);

  // create virtual node
  std::pair<index_type, bool> insertNodeRet = addNode(port.name);
  if (!insertNodeRet.second) return;
  Node& node             = m_vNode.at(insertNodeRet.first);
  NodeProperty& property = m_vNodeProperty.at(node.id());
  property.setMacroId(macro.id());
  node.setStatus(PlaceStatusEnum::FIXED);
  node.set(0, 0, 0, 0);

  m_numIOPin += 1;
}

void PlaceDB::yaml_constraint_cbk(YamlParser::YamlConstraint const& constr) { m_vConstraint.push_back(constr); }

void PlaceDB::yaml_end_cbk() {
  // set core site based on max usage
  if (!m_vSiteUsedCount.empty()) {
    auto itMax = std::max_element(m_vSiteUsedCount.begin(), m_vSiteUsedCount.end());
    if (itMax != m_vSiteUsedCount.end()) {
      m_coreSiteId = itMax - m_vSiteUsedCount.begin();
    }
  }

  // create a default site with width=1, height=1 if no sites were defined
  if (m_vSite.empty()) {
    m_vSite.emplace_back();
    Site& site = m_vSite.back();
    site.setId(0);
    site.setName("default_site");
    site.setSize(kX, 1);
    site.setSize(kY, 1);
    m_mSiteName2Index["default_site"] = 0;
    m_vSiteUsedCount.emplace_back(1);
    m_coreSiteId = 0;
  }

  // generate placement rows from die area with row height = 1 unit
  if (m_vRow.empty()) {
    coordinate_type rowHeight = siteHeight();  // 1 unit
    coordinate_type siteW     = siteWidth();   // 1 unit
    coordinate_type dieXL     = m_dieArea.xl();
    coordinate_type dieYL     = m_dieArea.yl();
    coordinate_type dieXH     = m_dieArea.xh();
    coordinate_type dieYH     = m_dieArea.yh();

    index_type numRows = (dieYH - dieYL) / rowHeight;
    for (index_type i = 0; i < numRows; ++i) {
      m_vRow.emplace_back();
      Row& row = m_vRow.back();
      row.setId(i);
      row.setName("yaml_row_" + std::to_string(i));
      row.setMacroName(m_vSite[m_coreSiteId].name());
      row.set(dieXL, dieYL + i * rowHeight, dieXH, dieYL + (i + 1) * rowHeight);
      row.setStep(siteW, rowHeight);
      m_rowBbox.encompass(row);
    }

    dreamplacePrint(kINFO, "YAML: generated %lu rows (height=%d, site_width=%d) from die area\n", m_vRow.size(), (int) rowHeight, (int) siteW);
  }

  dreamplacePrint(kINFO, "YAML parsing complete: %lu nodes, %lu nets, %lu macros\n", m_vNode.size(), m_vNet.size(), m_vMacro.size());
}

DREAMPLACE_END_NAMESPACE
