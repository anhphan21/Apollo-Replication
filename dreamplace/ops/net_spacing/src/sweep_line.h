#include <iostream>
#include <queue>
#include <set>

#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE
template <typename T>
class NetIntersector {
 public:
  struct Point {
    T x, y;
  };

  struct Segment {
    Segment(T x0, T y0, T x1, T y1) : p1(x0, y0), p2(x1, y1) {}
    Point p1, p2;
    int id;
    int crossingCount = 0;
  };

 protected:
  std::vector<Segment> nets;

 public:
  virtual ~NetIntersector() = default;

 private:
  int orientation(Point p, Point q, Point r) const {
    long long val =
        static_cast<long long>(q.y - p.y) * static_cast<long long>(r.x - q.x) - static_cast<long long>(q.x - p.x) * static_cast<long long>(r.y - q.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
  }

  bool onSegment(Point p, Point q, Point r) const {
    return q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y);
  }

  bool doIntersect(const Segment& s1, const Segment& s2) const {
    int o1 = orientation(s1.p1, s1.p2, s2.p1);
    int o2 = orientation(s1.p1, s1.p2, s2.p2);
    int o3 = orientation(s2.p1, s2.p2, s1.p1);
    int o4 = orientation(s2.p1, s2.p2, s1.p2);

    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && onSegment(s1.p1, s2.p1, s1.p2)) return true;
    if (o2 == 0 && onSegment(s1.p1, s2.p2, s1.p2)) return true;
    if (o3 == 0 && onSegment(s2.p1, s1.p1, s2.p2)) return true;
    if (o4 == 0 && onSegment(s2.p1, s1.p2, s2.p2)) return true;

    return false;
  }

  double evaluateY(const Segment& s, T x) const {
    if (s.p2.x == s.p1.x) return static_cast<double>(s.p1.y);
    double slope = static_cast<double>(s.p2.y - s.p1.y) / static_cast<double>(s.p2.x - s.p1.x);
    return static_cast<double>(s.p1.y) + slope * static_cast<double>(x - s.p1.x);
  }

  struct Event {
    T x, y;
    bool isLeftEndpoint;
    Segment segment;

    bool operator>(const Event& other) const {
      if (x == other.x) return y > other.y;
      return x > other.x;
    }
  };

 public:
  void calculateIntersections() {
    // Clear previous runs
    for (auto& net : nets) {
      net.crossingCount = 0;
    }

    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> eventQueue;
    for (const auto& net : nets) {
      Segment s = net;
      if (s.p1.x > s.p2.x) std::swap(s.p1, s.p2);
      eventQueue.push({s.p1.x, s.p1.y, true, s});
      eventQueue.push({s.p2.x, s.p2.y, false, s});
    }

    T current_sweep_x;
    auto activeCompare = [&current_sweep_x, this](const Segment& s1, const Segment& s2) {
      double y1 = this->evaluateY(s1, current_sweep_x);
      double y2 = this->evaluateY(s2, current_sweep_x);
      if (y1 == y2) return s1.id < s2.id;
      return y1 < y2;
    };

    std::set<Segment, decltype(activeCompare)> activeSegments(activeCompare);

    // Use a set to track unique pairs of crossing IDs
    std::set<std::pair<int, int>> uniqueCrossings;

    auto checkAndRecord = [&](const Segment& a, const Segment& b) {
      if (doIntersect(a, b)) {
        int id1 = std::min(a.id, b.id);
        int id2 = std::max(a.id, b.id);
        uniqueCrossings.insert({id1, id2});
      }
    };

    while (!eventQueue.empty()) {
      Event currentEvent = eventQueue.top();
      eventQueue.pop();

      current_sweep_x    = currentEvent.x;
      Segment currentSeg = currentEvent.segment;

      if (currentEvent.isLeftEndpoint) {
        auto result = activeSegments.insert(currentSeg);
        auto it     = result.first;

        auto nextIt = std::next(it);
        if (nextIt != activeSegments.end()) checkAndRecord(*it, *nextIt);

        if (it != activeSegments.begin()) {
          auto prevIt = std::prev(it);
          checkAndRecord(*it, *prevIt);
        }
      } else {
        auto it = activeSegments.find(currentSeg);
        if (it != activeSegments.end()) {
          auto nextIt  = std::next(it);
          bool hasPrev = (it != activeSegments.begin());
          if (hasPrev && nextIt != activeSegments.end()) {
            checkAndRecord(*std::prev(it), *nextIt);
          }
          activeSegments.erase(it);
        }
      }
    }

    // 4. Update the crossing counts in the master vector
    for (const auto& pair : uniqueCrossings) {
      // Find the nets by ID and increment their count
      auto it1 = std::find_if(nets.begin(), nets.end(), [&](const Segment& s) { return s.id == pair.first; });
      auto it2 = std::find_if(nets.begin(), nets.end(), [&](const Segment& s) { return s.id == pair.second; });

      if (it1 != nets.end()) it1->crossingCount++;
      if (it2 != nets.end()) it2->crossingCount++;
    }
  }

  void printResults() const {
    for (const auto& net : nets)
      std::cout << "Net " << net.id << " crosses " << net.crossingCount << " other nets.\n";
  }
};
DREAMPLACE_END_NAMESPACE