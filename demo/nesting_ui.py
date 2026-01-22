#!/usr/bin/env python3
"""
U-Nesting Interactive UI
=========================
SigmaNest XML 파일을 로드하여 네스팅 최적화를 시뮬레이션하는 대화형 UI

Requirements: Python 3.8+ with tkinter (built-in)
Usage: python nesting_ui.py
"""

import json
import math
import os
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ============================================================
# SigmaNest XML Parser (from sigmanest_to_json.py)
# ============================================================

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

    for line_elem in dlist.findall('.//*[@Type="TSNLine"]'):
        p1_elem = line_elem.find('p1')
        p2_elem = line_elem.find('p2')
        if p1_elem is not None and p2_elem is not None:
            p1 = (float(p1_elem.find('X').text), float(p1_elem.find('Y').text))
            p2 = (float(p2_elem.find('X').text), float(p2_elem.find('Y').text))
            segments.append((p1, p2, False, None))

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
            continue

        min_x = min(p[0] for p in exterior_points)
        min_y = min(p[1] for p in exterior_points)

        normalized_exterior = [(p[0] - min_x, p[1] - min_y) for p in exterior_points]
        normalized_holes = [
            [(p[0] - min_x, p[1] - min_y) for p in hole]
            for hole in hole_points_list
        ]

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

# ============================================================
# UI Application
# ============================================================

class NestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("U-Nesting - SigmaNest Optimizer")
        self.root.geometry("1200x800")

        # Data
        self.parts: List[SigmaNestPart] = []
        self.placements = []
        self.current_file = None

        # Find project root (where cargo.toml is)
        self.project_root = self.find_project_root()

        self.setup_ui()

    def find_project_root(self):
        """Find the U-Nesting project root directory"""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "Cargo.toml").exists():
                return current
            current = current.parent
        return Path(__file__).parent.parent

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        # File loading
        ttk.Label(left_frame, text="1. Load File").grid(row=0, column=0, sticky="w", pady=(0, 5))

        ttk.Button(left_frame, text="Load SigmaNest XML",
                   command=self.load_xml).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(left_frame, text="Load JSON",
                   command=self.load_json).grid(row=2, column=0, sticky="ew", pady=2)

        ttk.Separator(left_frame, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=10)

        # Parameters
        ttk.Label(left_frame, text="2. Parameters").grid(row=4, column=0, sticky="w", pady=(0, 5))

        param_frame = ttk.Frame(left_frame)
        param_frame.grid(row=5, column=0, sticky="ew")

        ttk.Label(param_frame, text="Strip Width:").grid(row=0, column=0, sticky="w")
        self.strip_width_var = tk.StringVar(value="2500")
        ttk.Entry(param_frame, textvariable=self.strip_width_var, width=7).grid(row=0, column=1, padx=2)

        ttk.Label(param_frame, text="Strip Height:").grid(row=1, column=0, sticky="w")
        self.strip_height_var = tk.StringVar(value="500")
        ttk.Entry(param_frame, textvariable=self.strip_height_var, width=7).grid(row=1, column=1, padx=2)

        ttk.Label(param_frame, text="Time (s):").grid(row=2, column=0, sticky="w")
        self.time_limit_var = tk.StringVar(value="60")
        ttk.Entry(param_frame, textvariable=self.time_limit_var, width=7).grid(row=2, column=1, padx=2)

        ttk.Label(param_frame, text="Strategy:").grid(row=3, column=0, sticky="w")
        self.strategy_var = tk.StringVar(value="nfp")
        strategy_combo = ttk.Combobox(param_frame, textvariable=self.strategy_var, width=7,
                                       values=["blf", "nfp", "ga", "brkga", "sa"])
        strategy_combo.grid(row=3, column=1, padx=2)

        ttk.Separator(left_frame, orient="horizontal").grid(row=6, column=0, sticky="ew", pady=10)

        # Run
        ttk.Label(left_frame, text="3. Execute").grid(row=7, column=0, sticky="w", pady=(0, 5))

        ttk.Button(left_frame, text="Preview Random",
                   command=self.preview_random).grid(row=8, column=0, sticky="ew", pady=2)

        self.run_button = ttk.Button(left_frame, text="Run Nesting",
                                      command=self.run_nesting)
        self.run_button.grid(row=9, column=0, sticky="ew", pady=2)

        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(left_frame, textvariable=self.progress_var).grid(row=10, column=0, sticky="w", pady=5)

        ttk.Separator(left_frame, orient="horizontal").grid(row=11, column=0, sticky="ew", pady=10)

        # Results
        ttk.Label(left_frame, text="4. Results").grid(row=12, column=0, sticky="w", pady=(0, 5))

        self.result_text = tk.Text(left_frame, width=30, height=10, state="disabled")
        self.result_text.grid(row=13, column=0, sticky="ew")

        # Right panel - Canvas
        right_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        right_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Canvas with scrollbars
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="white", width=800, height=600)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # Scrollbars
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        # Zoom controls
        zoom_frame = ttk.Frame(right_frame)
        zoom_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        ttk.Button(zoom_frame, text="Zoom In", command=lambda: self.zoom(1.2)).pack(side="left", padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(side="left", padx=2)
        ttk.Button(zoom_frame, text="Fit", command=self.fit_view).pack(side="left", padx=2)

        self.zoom_level = 1.0

        # Part list
        list_frame = ttk.LabelFrame(right_frame, text="Parts", padding="5")
        list_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.part_listbox = tk.Listbox(list_frame, height=5)
        self.part_listbox.pack(fill="x", expand=True)
        self.part_listbox.bind("<<ListboxSelect>>", self.on_part_select)

    def load_xml(self):
        """Load SigmaNest XML file"""
        filepath = filedialog.askopenfilename(
            title="Select SigmaNest XML",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            initialdir=self.project_root / "dev"
        )
        if not filepath:
            return

        try:
            self.progress_var.set("Parsing XML...")
            self.root.update()

            self.parts = parse_sigmanest_xml(filepath)
            self.current_file = filepath
            self.placements = []

            self.update_part_list()
            self.draw_parts()

            self.progress_var.set(f"Loaded {len(self.parts)} parts")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load XML:\n{e}")
            self.progress_var.set("Error loading file")

    def load_json(self):
        """Load U-Nesting JSON file"""
        filepath = filedialog.askopenfilename(
            title="Select JSON Dataset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.project_root / "datasets"
        )
        if not filepath:
            return

        try:
            self.progress_var.set("Loading JSON...")
            self.root.update()

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.parts = []
            for item in data.get('items', []):
                shape = item.get('shape', {})
                if shape.get('type') == 'simple_polygon':
                    exterior = [(p[0], p[1]) for p in shape.get('data', [])]
                else:
                    exterior = [(p[0], p[1]) for p in shape.get('exterior', shape.get('data', []))]

                holes = []
                for hole in shape.get('holes', []):
                    holes.append([(p[0], p[1]) for p in hole])

                self.parts.append(SigmaNestPart(
                    name=item.get('name', f"Part_{item.get('id', len(self.parts))}"),
                    exterior=exterior,
                    holes=holes,
                    quantity=item.get('demand', 1),
                    allowed_rotations=item.get('allowed_orientations', [0.0])
                ))

            self.current_file = filepath
            self.placements = []

            if 'strip_width' in data:
                self.strip_width_var.set(str(int(data['strip_width'])))
            if 'strip_height' in data:
                self.strip_height_var.set(str(int(data['strip_height'])))

            self.update_part_list()
            self.draw_parts()

            self.progress_var.set(f"Loaded {len(self.parts)} parts")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON:\n{e}")
            self.progress_var.set("Error loading file")

    def update_part_list(self):
        """Update the parts listbox"""
        self.part_listbox.delete(0, tk.END)
        for part in self.parts:
            self.part_listbox.insert(tk.END,
                f"{part.name} ({len(part.exterior)} pts, qty={part.quantity})")

    def on_part_select(self, event):
        """Handle part selection"""
        selection = self.part_listbox.curselection()
        if selection:
            self.draw_parts(highlight=selection[0])

    def draw_parts(self, highlight=None):
        """Draw parts on canvas"""
        self.canvas.delete("all")

        if not self.parts:
            return

        # Calculate bounds based on whether we have placements
        if self.placements:
            # Use strip-based bounds for placements
            strip_width = float(self.strip_width_var.get())
            strip_height = float(self.strip_height_var.get())

            # Find max x position to determine number of strips
            max_x_pos = 0
            for p in self.placements:
                x_off = p.get('x', 0)
                max_x_pos = max(max_x_pos, x_off + 200)  # Approximate part size

            num_strips = max(1, int(max_x_pos // strip_width) + 1)
            min_x, min_y = 0, 0
            max_x = num_strips * strip_width
            max_y = strip_height
        else:
            # Use part bounds for preview
            all_points = []
            for part in self.parts:
                all_points.extend(part.exterior)

            if not all_points:
                return

            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)

        width = max_x - min_x
        height = max_y - min_y

        # Calculate scale to fit canvas
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        margin = 50

        scale_x = (canvas_w - 2*margin) / max(width, 1)
        scale_y = (canvas_h - 2*margin) / max(height, 1)
        scale = min(scale_x, scale_y) * self.zoom_level

        def transform(x, y):
            return (
                margin + (x - min_x) * scale,
                canvas_h - margin - (y - min_y) * scale  # Flip Y
            )

        # Draw each part
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

        for i, part in enumerate(self.parts):
            color = colors[i % len(colors)]
            fill_color = color if highlight != i else "#ffff00"

            # Draw exterior
            if len(part.exterior) > 2:
                coords = []
                for x, y in part.exterior:
                    tx, ty = transform(x, y)
                    coords.extend([tx, ty])
                self.canvas.create_polygon(coords, fill=fill_color, outline="black",
                                            width=2, stipple="gray50" if highlight != i else "")

            # Draw holes
            for hole in part.holes:
                if len(hole) > 2:
                    coords = []
                    for x, y in hole:
                        tx, ty = transform(x, y)
                        coords.extend([tx, ty])
                    self.canvas.create_polygon(coords, fill="white", outline="red", width=1)

        # Draw placements if available
        if self.placements:
            self.draw_placements(transform, scale)

        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def draw_placements(self, transform, scale):
        """Draw placement results with multi-strip support"""
        strip_width = float(self.strip_width_var.get())
        strip_height = float(self.strip_height_var.get())

        # Group placements by strip (based on x position)
        strips = {}
        for p in self.placements:
            x_off = p.get('x', 0)
            strip_idx = int(x_off // strip_width)
            if strip_idx not in strips:
                strips[strip_idx] = []
            strips[strip_idx].append(p)

        # Ensure at least one strip
        if not strips:
            strips[0] = []

        num_strips = max(strips.keys()) + 1 if strips else 1
        colors = ["#90EE90", "#87CEEB", "#FFB6C1", "#DDA0DD", "#F0E68C", "#98FB98"]

        # Draw each strip
        for strip_idx in range(num_strips):
            strip_x_offset = strip_idx * strip_width

            # Draw strip boundary
            x1, y1 = transform(strip_x_offset, 0)
            x2, y2 = transform(strip_x_offset + strip_width, strip_height)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, dash=(5, 5))

            # Draw strip label
            label_x, label_y = transform(strip_x_offset + 5, strip_height - 10)
            self.canvas.create_text(label_x, label_y, text=f"Strip {strip_idx + 1}",
                                    anchor="sw", fill="blue", font=("Arial", 10, "bold"))

            # Draw placed parts in this strip
            strip_placements = strips.get(strip_idx, [])
            for p in strip_placements:
                part_idx = p.get('geometry_index', 0)
                if part_idx < len(self.parts):
                    part = self.parts[part_idx]
                    x_off = p.get('x', 0)
                    y_off = p.get('y', 0)
                    rotation = p.get('rotation', 0)

                    coords = []
                    for px, py in part.exterior:
                        cos_r = math.cos(rotation)
                        sin_r = math.sin(rotation)
                        rx = px * cos_r - py * sin_r + x_off
                        ry = px * sin_r + py * cos_r + y_off
                        tx, ty = transform(rx, ry)
                        coords.extend([tx, ty])

                    if coords:
                        fill_color = colors[strip_idx % len(colors)]
                        self.canvas.create_polygon(coords, fill=fill_color, outline="darkgreen", width=2)

    def zoom(self, factor):
        """Zoom in/out"""
        self.zoom_level *= factor
        self.draw_parts()

    def fit_view(self):
        """Fit view to content"""
        self.zoom_level = 1.0
        self.draw_parts()

    def preview_random(self):
        """Preview parts randomly distributed across multiple strips (visual only)"""
        if not self.parts:
            messagebox.showwarning("Warning", "No parts loaded!")
            return

        import random

        strip_width = float(self.strip_width_var.get())
        strip_height = float(self.strip_height_var.get())

        # Expand parts by quantity
        expanded = []
        for idx, part in enumerate(self.parts):
            for _ in range(part.quantity):
                expanded.append((idx, part))

        random.shuffle(expanded)

        # Estimate total area and strips needed
        total_area = 0
        for _, part in expanded:
            if part.exterior:
                xs = [p[0] for p in part.exterior]
                ys = [p[1] for p in part.exterior]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                total_area += w * h

        strip_area = strip_width * strip_height
        num_strips = max(1, int(total_area / (strip_area * 0.5)) + 1)

        # Randomly assign parts to strips
        self.placements = []
        for i, (part_idx, part) in enumerate(expanded):
            strip_idx = random.randint(0, num_strips - 1)
            if part.exterior:
                xs = [p[0] for p in part.exterior]
                ys = [p[1] for p in part.exterior]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                # Random position within strip
                x = strip_idx * strip_width + random.uniform(0, max(0, strip_width - w))
                y = random.uniform(0, max(0, strip_height - h))
            else:
                x = strip_idx * strip_width
                y = 0

            self.placements.append({
                'geometry_index': part_idx,
                'x': x,
                'y': y,
                'rotation': 0
            })

        self.draw_parts()
        self.progress_var.set(f"Random preview: {len(expanded)} pieces in {num_strips} strips")

    def run_nesting(self):
        """Run nesting algorithm"""
        if not self.parts:
            messagebox.showwarning("Warning", "No parts loaded!")
            return

        self.run_button.config(state="disabled")
        self.progress_var.set("Running nesting...")
        self.root.update()

        # Run in thread to keep UI responsive
        thread = threading.Thread(target=self._run_nesting_thread)
        thread.start()

    def _run_nesting_thread(self):
        """Background thread for nesting"""
        try:
            # Create temporary JSON file
            json_data = self._parts_to_json()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False,
                                              encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
                temp_file = f.name

            try:
                # Run benchmark tool
                strategy = self.strategy_var.get()
                time_limit = self.time_limit_var.get()

                cmd = [
                    "cargo", "run", "--release",
                    "-p", "u-nesting-benchmark",
                    "--bin", "bench-runner", "--",
                    "run-file", temp_file,
                    "-s", strategy,
                    "-t", time_limit
                ]

                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=int(time_limit) + 60
                )

                # Parse results
                self._parse_results(result.stdout + result.stderr)

            finally:
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            self.root.after(0, lambda: self._show_error("Nesting timed out!"))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            self.root.after(0, self._nesting_complete)

    def _parts_to_json(self):
        """Convert parts to JSON format"""
        items = []
        for i, part in enumerate(self.parts):
            exterior = list(part.exterior)
            if exterior and exterior[0] != exterior[-1]:
                exterior.append(exterior[0])

            items.append({
                "id": i,
                "name": part.name,
                "demand": part.quantity,
                "allowed_orientations": part.allowed_rotations,
                "shape": {
                    "type": "simple_polygon",
                    "data": [[float(x), float(y)] for x, y in exterior]
                }
            })

        return {
            "name": "ui_nesting",
            "items": items,
            "strip_width": float(self.strip_width_var.get()),
            "strip_height": float(self.strip_height_var.get())
        }

    def _parse_results(self, output):
        """Parse benchmark output"""
        lines = output.strip().split('\n')

        result_text = []
        for line in lines:
            if 'length=' in line or 'placed=' in line:
                result_text.append(line.strip())
            elif 'BENCHMARK RESULTS' in line:
                result_text.append("=" * 40)
            elif any(s in line for s in ['BottomLeftFill', 'NfpGuided', 'GeneticAlgorithm']):
                result_text.append(line.strip())

        # Update UI from main thread
        self.root.after(0, lambda: self._update_results('\n'.join(result_text) or output))

    def _update_results(self, text):
        """Update results text widget"""
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state="disabled")

    def _show_error(self, msg):
        """Show error message"""
        messagebox.showerror("Error", msg)
        self.progress_var.set("Error")

    def _nesting_complete(self):
        """Called when nesting is complete"""
        self.run_button.config(state="normal")
        self.progress_var.set("Complete!")

# ============================================================
# Main
# ============================================================

def main():
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # Modern theme

    app = NestingApp(root)

    # Load default file if exists
    default_xml = Path(__file__).parent / "우등산업.XML"
    if default_xml.exists():
        root.after(100, lambda: app.load_xml_file(str(default_xml))
                   if hasattr(app, 'load_xml_file') else None)

    root.mainloop()

if __name__ == "__main__":
    main()
