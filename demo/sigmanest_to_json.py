#!/usr/bin/env python3
"""
SigmaNest XML → U-Nesting JSON 변환기
=====================================
SigmaNest XML 파일을 파싱하여 U-Nesting JSON 포맷으로 변환

Usage:
    python sigmanest_to_json.py [input.xml] [output.json]
"""

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

@dataclass
class SigmaNestPart:
    name: str
    exterior: List[Tuple[float, float]]
    holes: List[List[Tuple[float, float]]]
    quantity: int = 1
    allowed_rotations: List[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])

def parse_dlist_geometry(dlist) -> List[Tuple[float, float]]:
    """Parse TSNLine and TSNArc from a DList into a point list"""
    segments = []

    # Parse lines
    for line_elem in dlist.findall('.//*[@Type="TSNLine"]'):
        p1_elem = line_elem.find('p1')
        p2_elem = line_elem.find('p2')
        if p1_elem is not None and p2_elem is not None:
            p1 = (float(p1_elem.find('X').text), float(p1_elem.find('Y').text))
            p2 = (float(p2_elem.find('X').text), float(p2_elem.find('Y').text))
            segments.append((p1, p2, False, None))

    # Parse arcs
    for arc_elem in dlist.findall('.//*[@Type="TSNArc"]'):
        cntpnt = arc_elem.find('CntPnt')
        radius_elem = arc_elem.find('Radius')
        start_elem = arc_elem.find('StartAng')
        end_elem = arc_elem.find('EndAng')
        ccw_elem = arc_elem.find('CCW_Dir')

        if (cntpnt is not None and radius_elem is not None and
            start_elem is not None and end_elem is not None):
            cx = float(cntpnt.find('X').text)
            cy = float(cntpnt.find('Y').text)
            r = float(radius_elem.text)
            start_ang = float(start_elem.text)
            end_ang = float(end_elem.text)
            ccw = ccw_elem.text.lower() == 'true' if ccw_elem is not None else False

            start_rad = math.radians(start_ang)
            end_rad = math.radians(end_ang)
            p1 = (cx + r * math.cos(start_rad), cy + r * math.sin(start_rad))
            p2 = (cx + r * math.cos(end_rad), cy + r * math.sin(end_rad))

            arc_data = (cx, cy, r, start_ang, end_ang, ccw)
            segments.append((p1, p2, True, arc_data))

    if not segments:
        return []

    def dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def arc_to_points(arc_data, num_segments=16):
        cx, cy, r, start_ang, end_ang, ccw = arc_data
        points = []
        start_rad = math.radians(start_ang)
        end_rad = math.radians(end_ang)

        if ccw:
            if end_rad < start_rad:
                end_rad += 2 * math.pi
        else:
            if start_rad < end_rad:
                start_rad += 2 * math.pi

        for i in range(num_segments + 1):
            t = i / num_segments
            angle = start_rad + t * (end_rad - start_rad)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))
        return points

    # Build chain
    used = [False] * len(segments)
    chain = []

    p1, p2, is_arc, arc_data = segments[0]
    if is_arc:
        chain.extend(arc_to_points(arc_data))
    else:
        chain.append(p1)
        chain.append(p2)
    used[0] = True

    tolerance = 0.1

    changed = True
    while changed:
        changed = False
        for i, (p1, p2, is_arc, arc_data) in enumerate(segments):
            if used[i]:
                continue

            if dist(chain[-1], p1) < tolerance:
                if is_arc:
                    chain.extend(arc_to_points(arc_data)[1:])
                else:
                    chain.append(p2)
                used[i] = True
                changed = True
            elif dist(chain[-1], p2) < tolerance:
                if is_arc:
                    arc_pts = arc_to_points(arc_data)
                    arc_pts.reverse()
                    chain.extend(arc_pts[1:])
                else:
                    chain.append(p1)
                used[i] = True
                changed = True
            elif dist(chain[0], p2) < tolerance:
                if is_arc:
                    chain = arc_to_points(arc_data)[:-1] + chain
                else:
                    chain.insert(0, p1)
                used[i] = True
                changed = True
            elif dist(chain[0], p1) < tolerance:
                if is_arc:
                    arc_pts = arc_to_points(arc_data)
                    arc_pts.reverse()
                    chain = arc_pts[:-1] + chain
                else:
                    chain.insert(0, p2)
                used[i] = True
                changed = True

    return chain

def parse_sigmanest_xml(xml_path: str) -> List[SigmaNestPart]:
    """SigmaNest XML 파일 파싱"""
    with open(xml_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    root = ET.fromstring(content)
    parts = []

    for part_elem in root.findall('.//*[@Type="TSNPart"]'):
        name_elem = part_elem.find('.//PartName')
        name = name_elem.text if name_elem is not None else f"Part_{len(parts)}"

        abs_angles = part_elem.find('AbsAngles')
        if abs_angles is not None and abs_angles.text:
            rotations = [float(a) for a in abs_angles.text.split(',')]
        else:
            rotations = [0.0, 90.0, 180.0, 270.0]

        qty = 1
        wolist = part_elem.find('WOList')
        if wolist is not None:
            wo = wolist.find('.//*[@Type="TSNWorkOrder"]')
            if wo is not None:
                qty_elem = wo.find('QtyOrdered')
                if qty_elem is not None:
                    qty = int(float(qty_elem.text))

        part_dlist = part_elem.find('DList')
        if part_dlist is None:
            continue

        exterior_points = None
        hole_points_list = []

        for contour_elem in part_dlist.findall('.//*[@Type="TSNContour"]'):
            outside_c = contour_elem.find('OutSideC')
            is_exterior = outside_c is not None and outside_c.text.lower() == 'true'

            contour_dlist = contour_elem.find('DList')
            if contour_dlist is None:
                continue

            points = parse_dlist_geometry(contour_dlist)
            if not points:
                continue

            if is_exterior:
                exterior_points = points
            else:
                hole_points_list.append(points)

        if exterior_points is None:
            print(f"  Warning: No exterior contour for {name}")
            continue

        min_x = min(p[0] for p in exterior_points)
        min_y = min(p[1] for p in exterior_points)

        normalized_exterior = [(p[0] - min_x, p[1] - min_y) for p in exterior_points]
        normalized_holes = [
            [(p[0] - min_x, p[1] - min_y) for p in hole]
            for hole in hole_points_list
        ]

        # Make unique name if duplicate
        base_name = name
        counter = 1
        while any(p.name == name for p in parts):
            name = f"{base_name}_{counter}"
            counter += 1

        parts.append(SigmaNestPart(
            name=name,
            exterior=normalized_exterior,
            holes=normalized_holes,
            quantity=qty,
            allowed_rotations=rotations
        ))

    return parts

def parts_to_unesting_json(parts: List[SigmaNestPart], sheet_height=500.0) -> dict:
    """Convert parts to U-Nesting JSON format (benchmark tool format)"""
    items = []
    for i, part in enumerate(parts):
        # Benchmark tool format: simple_polygon with data array
        # Close the polygon by repeating first point
        exterior = list(part.exterior)
        if exterior and exterior[0] != exterior[-1]:
            exterior.append(exterior[0])

        item = {
            "id": i,  # Numeric ID required
            "name": part.name,  # Keep name for reference
            "demand": part.quantity,
            "allowed_orientations": part.allowed_rotations,
            "shape": {
                "type": "simple_polygon",
                "data": exterior
            }
        }
        # Note: holes not supported in simple_polygon format
        items.append(item)

    return {
        "name": "sigmanest_import",
        "items": items,
        "strip_height": sheet_height
    }

def main():
    import sys

    script_dir = Path(__file__).parent
    xml_path = script_dir / "우등산업.XML"
    output_path = script_dir / "sigmanest_parts.json"

    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])

    print("=" * 60)
    print("SigmaNest XML → U-Nesting JSON Converter")
    print("=" * 60)
    print()

    print(f"[1] Parsing: {xml_path}")
    parts = parse_sigmanest_xml(str(xml_path))
    print(f"    Found {len(parts)} part(s)")
    print()

    for part in parts:
        ext = part.exterior
        if ext:
            xs = [p[0] for p in ext]
            ys = [p[1] for p in ext]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
        else:
            width = height = 0

        print(f"    {part.name}:")
        print(f"      Size: {width:.1f} x {height:.1f} mm")
        print(f"      Vertices: {len(part.exterior)}, Holes: {len(part.holes)}")
        print(f"      Quantity: {part.quantity}")
    print()

    print(f"[2] Converting to U-Nesting JSON format...")
    data = parts_to_unesting_json(parts)
    print()

    print(f"[3] Writing: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    Size: {output_path.stat().st_size:,} bytes")
    print()

    print("=" * 60)
    print("Conversion complete!")
    print()
    print("To run nesting:")
    print(f"  cargo run -p u-nesting-benchmark --release --bin bench-runner -- \\")
    print(f"    run-file {output_path} -s nfp -t 30")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    exit(main())
