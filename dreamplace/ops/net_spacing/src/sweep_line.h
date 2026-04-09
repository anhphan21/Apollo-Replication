#include <iostream>
#include <queue>
#include <set>

#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE
template <typename T>
class NetIntersector {
 public:
  struct Point {
    Point(T x, T y) : x(x), y (y) {}
    T x, y;
  };

  struct Segment {
    Segment(T x0, T y0, T x1, T y1, int idx) : p1(x0, y0), p2(x1, y1), id(idx) {}

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

 public:
  void calculateIntersections() {
    // Clear previous runs
    for (auto& net : nets) {
      net.crossingCount = 0;
    }

    // Sweep-line: when a segment's left endpoint is reached, check it
    // against ALL currently active segments (not just immediate neighbors).
    // This avoids the pitfall of the simplified Bentley-Ottmann that only
    // checks neighbors — segments that are non-adjacent in y-order can
    // still intersect, and a mutable comparator in std::set causes
    // ordering inconsistencies after crossings.

    struct Event {
      T x;
      int segIdx;       // index into nets
      bool isLeft;      // true = left endpoint, false = right endpoint

      bool operator>(const Event& other) const {
        if (x == other.x) return !isLeft && other.isLeft;  // left before right at same x
        return x > other.x;
      }
    };

    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> eventQueue;

    // Normalize so p1.x <= p2.x, then create events
    for (int i = 0; i < static_cast<int>(nets.size()); ++i) {
      if (nets[i].p1.x > nets[i].p2.x) std::swap(nets[i].p1, nets[i].p2);
      eventQueue.push({nets[i].p1.x, i, true});
      eventQueue.push({nets[i].p2.x, i, false});
    }

    // Track which segment indices are currently active
    std::set<int> activeIndices;

    // Track unique crossing pairs
    std::set<std::pair<int, int>> uniqueCrossings;

    while (!eventQueue.empty()) {
      Event ev = eventQueue.top();
      eventQueue.pop();

      if (ev.isLeft) {
        // Check new segment against all currently active segments
        for (int ai : activeIndices) {
          if (doIntersect(nets[ev.segIdx], nets[ai])) {
            int id1 = std::min(nets[ev.segIdx].id, nets[ai].id);
            int id2 = std::max(nets[ev.segIdx].id, nets[ai].id);
            uniqueCrossings.insert({id1, id2});
          }
        }
        activeIndices.insert(ev.segIdx);
      } else {
        activeIndices.erase(ev.segIdx);
      }
    }

    // Update the crossing counts in the master vector
    for (const auto& pair : uniqueCrossings) {
      for (auto& net : nets) {
        if (net.id == pair.first || net.id == pair.second) {
          net.crossingCount++;
        }
      }
    }
  }

  void printResults() const {
    for (const auto& net : nets)
      std::cout << "Net " << net.id << " crosses " << net.crossingCount << " other nets.\n";
  }
};
DREAMPLACE_END_NAMESPACE