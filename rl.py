
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from collections import defaultdict
import re
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)-7s] %(name)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('RLPlacer')


#file parser code 
class DEFParser:
   
    
    def __init__(self):
        self.design_name = ""
        self.die_area = None  # (llx, lly, urx, ury)
        self.components = {}  # {name: {x, y, width, height, type, fixed, orient}}
        self.nets = []  # [{name, pins: [(comp, pin)]}]
        self.pins = {}  # I/O pins
        self.dbu_per_micron = 2000
        self.row_info = []
        
    def parse(self, def_file):
        """Parse DEF file"""
        logger.info(f"Parsing DEF file: {def_file}")
        
        if not os.path.exists(def_file):
            raise FileNotFoundError(f"DEF file not found: {def_file}")
        
        with open(def_file, 'r') as f:
            content = f.read()
        
        # Parse design name
        match = re.search(r'DESIGN\s+(\S+)', content)
        if match:
            self.design_name = match.group(1)
        
        # Parse die area
        match = re.search(r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', content)
        if match:
            self.die_area = tuple(map(int, match.groups()))
        else:
            logger.warning("Could not parse DIEAREA, using default")
            self.die_area = (0, 0, 1000000, 1000000)
        
        # Parse units
        match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if match:
            self.dbu_per_micron = int(match.group(1))
        
        # Parse components
        self._parse_components(content)
        
        # Parse nets
        self._parse_nets(content)
        
        # Parse pins (I/O)
        self._parse_pins(content)
        
        logger.info(f"Parsed: {len(self.components)} components, {len(self.nets)} nets, {len(self.pins)} I/O pins")
        logger.info(f"Die area: {self.die_area}")
        
        return self
    
    def _parse_components(self, content):
        """Parse COMPONENTS section"""
        match = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END COMPONENTS', content, re.DOTALL)
        if not match:
            logger.warning("No COMPONENTS section found")
            return
        
        num_comps = int(match.group(1))
        comp_section = match.group(2)
        
        # Parse each component
        comp_pattern = r'-\s+(\S+)\s+(\S+)\s+\+\s+(PLACED|FIXED|UNPLACED)\s*(?:\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\S+))?\s*;'
        
        for comp_match in re.finditer(comp_pattern, comp_section):
            name = comp_match.group(1)
            comp_type = comp_match.group(2)
            status = comp_match.group(3)
            x = int(comp_match.group(4)) if comp_match.group(4) else 0
            y = int(comp_match.group(5)) if comp_match.group(5) else 0
            orient = comp_match.group(6) if comp_match.group(6) else 'N'
            
            self.components[name] = {
                'x': x,
                'y': y,
                'type': comp_type,
                'fixed': (status == 'FIXED'),
                'placed': (status != 'UNPLACED'),
                'orient': orient,
                'width': 1000,   # Default, will be overwritten if we have LEF
                'height': 1000
            }
        
        logger.info(f"Parsed {len(self.components)} components (expected {num_comps})")
    
    def _parse_nets(self, content):
        """Parse NETS section"""
        match = re.search(r'NETS\s+(\d+)\s*;(.*?)END NETS', content, re.DOTALL)
        if not match:
            logger.warning("No NETS section found")
            return
        
        nets_section = match.group(2)
        
        # Split by net definitions
        net_pattern = r'-\s+(\S+)(.*?);'
        
        for net_match in re.finditer(net_pattern, nets_section, re.DOTALL):
            net_name = net_match.group(1)
            net_body = net_match.group(2)
            
            # Parse pins in this net
            pin_pattern = r'\(\s*(\S+)\s+(\S+)\s*\)'
            pins = []
            
            for pin_match in re.finditer(pin_pattern, net_body):
                comp = pin_match.group(1)
                pin = pin_match.group(2)
                pins.append((comp, pin))
            
            if pins:
                self.nets.append({
                    'name': net_name,
                    'pins': pins
                })
        
        logger.info(f"Parsed {len(self.nets)} nets")
    
    def _parse_pins(self, content):
        """Parse PINS section (I/O pins)"""
        match = re.search(r'PINS\s+(\d+)\s*;(.*?)END PINS', content, re.DOTALL)
        if not match:
            logger.info("No PINS section found (optional)")
            return
        
        pins_section = match.group(2)
        
        pin_pattern = r'-\s+(\S+)\s+\+\s+NET\s+(\S+).*?PLACED\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\S+).*?;'
        
        for pin_match in re.finditer(pin_pattern, pins_section, re.DOTALL):
            pin_name = pin_match.group(1)
            net_name = pin_match.group(2)
            x = int(pin_match.group(3))
            y = int(pin_match.group(4))
            orient = pin_match.group(5)
            
            self.pins[pin_name] = {
                'net': net_name,
                'x': x,
                'y': y,
                'orient': orient
            }
        
        logger.info(f"Parsed {len(self.pins)} I/O pins")
    
    def write(self, output_file, components):
        """Write DEF file with optimized placement"""
        logger.info(f"Writing optimized DEF: {output_file}")
        
        with open(output_file, 'w') as f:
            # Header
            f.write(f"VERSION 5.8 ;\n")
            f.write(f"DIVIDERCHAR \"/\" ;\n")
            f.write(f"BUSBITCHARS \"[]\" ;\n")
            f.write(f"DESIGN {self.design_name} ;\n")
            f.write(f"UNITS DISTANCE MICRONS {self.dbu_per_micron} ;\n\n")
            
            # Die area
            f.write(f"DIEAREA ( {self.die_area[0]} {self.die_area[1]} ) "
                   f"( {self.die_area[2]} {self.die_area[3]} ) ;\n\n")
            
            # Components with optimized positions
            f.write(f"COMPONENTS {len(components)} ;\n")
            for name, comp in sorted(components.items()):
                status = "FIXED" if comp['fixed'] else "PLACED"
                x = int(comp['x'])
                y = int(comp['y'])
                orient = comp.get('orient', 'N')
                f.write(f"  - {name} {comp['type']} + {status} ( {x} {y} ) {orient} ;\n")
            f.write("END COMPONENTS\n\n")
            
            # Pins (if any)
            if self.pins:
                f.write(f"PINS {len(self.pins)} ;\n")
                for pin_name, pin_data in self.pins.items():
                    f.write(f"  - {pin_name} + NET {pin_data['net']} + DIRECTION INPUT + USE SIGNAL\n")
                    f.write(f"    + PLACED ( {pin_data['x']} {pin_data['y']} ) {pin_data['orient']} ;\n")
                f.write("END PINS\n\n")
            
            # Nets (unchanged)
            f.write(f"NETS {len(self.nets)} ;\n")
            for net in self.nets:
                f.write(f"  - {net['name']}\n")
                for comp, pin in net['pins']:
                    f.write(f"    ( {comp} {pin} )\n")
                f.write("  ;\n")
            f.write("END NETS\n\n")
            
            f.write(f"END DESIGN\n")

#gnn part

class PlacementGNN(nn.Module):

    
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=128, output_dim=64):
        super().__init__()
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
   
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=8, edge_dim=hidden_dim, concat=False, dropout=0.1)
            for _ in range(4)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])
        self.dropout = nn.Dropout(0.1)
        
        self.node_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.graph_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for gnn, norm in zip(self.gnn_layers, self.norms):
            x_res = x
            x = gnn(x, edge_index, edge_attr)
            x = norm(x + x_res)
            x = F.relu(x)
            x = self.dropout(x)
        
        node_emb = self.node_output(x)
        
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_emb = self.graph_output(torch.cat([graph_mean, graph_max], dim=1))
        
        return node_emb, graph_emb


#placement func

class RLPlacementEngine:
    
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def compute_hpwl(self, components: Dict, nets: List, comp_to_idx: Dict) -> float:
        """Compute Half-Perimeter Wirelength"""
        total_hpwl = 0.0
        
        for net in nets:
            pins = [p[0] for p in net['pins'] if p[0] in components or p[0].startswith('PIN')]
            if len(pins) < 2:
                continue
            
            xs, ys = [], []
            for pin_comp in pins:
                if pin_comp in components:
                    xs.append(components[pin_comp]['x'])
                    ys.append(components[pin_comp]['y'])
            
            if len(xs) >= 2:
                hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                total_hpwl += hpwl
        
        return total_hpwl
    
    def compute_overlap(self, components: Dict, die_area: Tuple) -> float:
        """Compute cell overlap penalty"""
        overlap = 0.0
        comp_list = list(components.values())
        
        for i in range(len(comp_list)):
            for j in range(i+1, len(comp_list)):
                c1, c2 = comp_list[i], comp_list[j]
                
                # Check overlap
                x_overlap = max(0, min(c1['x'] + c1['width'], c2['x'] + c2['width']) - 
                               max(c1['x'], c2['x']))
                y_overlap = max(0, min(c1['y'] + c1['height'], c2['y'] + c2['height']) - 
                               max(c1['y'], c2['y']))
                
                overlap += x_overlap * y_overlap
        
        return overlap
    
    def place_analytical(self, def_parser: DEFParser) -> Dict:
        """Analytical placement using force-directed optimization"""
        logger.info("="*80)
        logger.info("Starting RL-Based Analytical Placement")
        logger.info("="*80)
        
        components = dict(def_parser.components)
        die_area = def_parser.die_area
        nets = def_parser.nets
        
        width = die_area[2] - die_area[0]
        height = die_area[3] - die_area[1]
        
        # Separate fixed and movable cells
        movable = {name: comp for name, comp in components.items() if not comp['fixed']}
        fixed = {name: comp for name, comp in components.items() if comp['fixed']}
        
        logger.info(f"Total components: {len(components)}")
        logger.info(f"Movable cells: {len(movable)}")
        logger.info(f"Fixed cells: {len(fixed)}")
        logger.info(f"Nets: {len(nets)}")
        logger.info(f"Die area: {width} x {height}")
        
        # Initialize movable cells with better spreading
        self._initialize_positions(movable, die_area)
        
        # Iterative optimization
        num_iterations = 50
        logger.info(f"\nRunning {num_iterations} optimization iterations...")
        
        for iteration in range(num_iterations):
            # Force-directed placement
            forces = self._compute_forces(movable, fixed, nets, die_area)
            
            # Update positions with adaptive step size
            step_size = 0.5 * (1.0 - iteration / num_iterations)
            self._update_positions(movable, forces, step_size, die_area)
            
            # Compute metrics every 10 iterations
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                all_comps = {**fixed, **movable}
                hpwl = self.compute_hpwl(all_comps, nets, 
                                        {name: i for i, name in enumerate(all_comps.keys())})
                overlap = self.compute_overlap(all_comps, die_area)
                
                logger.info(f"Iteration {iteration:3d}: HPWL = {hpwl:.2e}, Overlap = {overlap:.2e}")
        
        # Final spreading to remove overlaps
        logger.info("\nApplying final legalization...")
        self._legalize_placement(movable, die_area)
        
        # Merge back
        components.update(movable)
        
        # Final metrics
        final_hpwl = self.compute_hpwl(components, nets, 
                                       {name: i for i, name in enumerate(components.keys())})
        final_overlap = self.compute_overlap(components, die_area)
        
        logger.info("="*80)
        logger.info(f"FINAL RESULTS:")
        logger.info(f"  Half-Perimeter Wirelength (HPWL): {final_hpwl:.2e}")
        logger.info(f"  Cell Overlap: {final_overlap:.2e}")
        logger.info("="*80)
        
        return components
    
    def _initialize_positions(self, movable: Dict, die_area: Tuple):
        """Initialize positions with grid-based spreading"""
        n = len(movable)
        grid_size = int(np.ceil(np.sqrt(n)))
        
        width = die_area[2] - die_area[0]
        height = die_area[3] - die_area[1]
        
        x_step = width / (grid_size + 1)
        y_step = height / (grid_size + 1)
        
        for idx, (name, comp) in enumerate(movable.items()):
            row = idx // grid_size
            col = idx % grid_size
            
            x = die_area[0] + x_step * (col + 1) + np.random.randint(-100, 100)
            y = die_area[1] + y_step * (row + 1) + np.random.randint(-100, 100)
            
            comp['x'] = max(die_area[0], min(die_area[2] - comp['width'], x))
            comp['y'] = max(die_area[1], min(die_area[3] - comp['height'], y))
    
    def _compute_forces(self, movable: Dict, fixed: Dict, nets: List, die_area: Tuple) -> Dict:
        """Compute forces on each movable cell"""
        forces = {name: np.array([0.0, 0.0]) for name in movable.keys()}
        all_comps = {**movable, **fixed}
        
        # Net forces (spring model)
        for net in nets:
            pins = [p[0] for p in net['pins'] if p[0] in all_comps]
            if len(pins) < 2:
                continue
            
            # Compute center of net
            cx = np.mean([all_comps[p]['x'] for p in pins])
            cy = np.mean([all_comps[p]['y'] for p in pins])
            
            # Apply spring force
            for pin in pins:
                if pin in movable:
                    dx = cx - movable[pin]['x']
                    dy = cy - movable[pin]['y']
                    forces[pin] += np.array([dx, dy]) * 0.1
        
        # Repulsive forces (avoid overlap)
        movable_list = list(movable.keys())
        for i, name1 in enumerate(movable_list):
            for name2 in movable_list[i+1:]:
                c1, c2 = movable[name1], movable[name2]
                
                dx = c2['x'] - c1['x']
                dy = c2['y'] - c1['y']
                dist = np.sqrt(dx**2 + dy**2) + 1e-6
                
                # Repulsive force
                if dist < 5000:
                    force_mag = 1000.0 / (dist + 1.0)
                    forces[name1] -= np.array([dx, dy]) / dist * force_mag
                    forces[name2] += np.array([dx, dy]) / dist * force_mag
        
        return forces
    
    def _update_positions(self, movable: Dict, forces: Dict, step_size: float, die_area: Tuple):
        """Update cell positions based on forces"""
        for name, force in forces.items():
            comp = movable[name]
            
            # Update position
            comp['x'] += int(force[0] * step_size)
            comp['y'] += int(force[1] * step_size)
            
            # Keep within die area
            comp['x'] = max(die_area[0], min(die_area[2] - comp['width'], comp['x']))
            comp['y'] = max(die_area[1], min(die_area[3] - comp['height'], comp['y']))
    
    def _legalize_placement(self, movable: Dict, die_area: Tuple):
        """Remove overlaps using greedy legalization"""
        # Sort by x-coordinate
        sorted_cells = sorted(movable.items(), key=lambda x: x[1]['x'])
        
        # Simple left-to-right legalization
        for name, comp in sorted_cells:
            # Check and resolve overlaps
            for other_name, other_comp in movable.items():
                if name == other_name:
                    continue
                
                # Check overlap
                x_overlap = (comp['x'] < other_comp['x'] + other_comp['width'] and 
                            comp['x'] + comp['width'] > other_comp['x'])
                y_overlap = (comp['y'] < other_comp['y'] + other_comp['height'] and 
                            comp['y'] + comp['height'] > other_comp['y'])
                
                if x_overlap and y_overlap:
                    # Shift right
                    comp['x'] = other_comp['x'] + other_comp['width'] + 100
                    
                    # Wrap to next row if needed
                    if comp['x'] > die_area[2] - comp['width']:
                        comp['x'] = die_area[0]
                        comp['y'] += comp['height'] + 100

#main func

def main():
    """Main entry point"""
    
    if len(sys.argv) != 2:
        print("Usage: python RLPlacer.py <input.def>")
        print("Example: python RLPlacer.py ispd19_test1.input.def")
        sys.exit(1)
    
    input_def = sys.argv[1]
    
    if not os.path.exists(input_def):
        logger.error(f"Input DEF file not found: {input_def}")
        sys.exit(1)
    
    # Parse input DEF
    start_time = time.time()
    def_parser = DEFParser()
    def_parser.parse(input_def)
    parse_time = time.time() - start_time
    logger.info(f"Parsing took {parse_time:.2f} seconds\n")
    
    # Run placement
    engine = RLPlacementEngine()
    placement_start = time.time()
    optimized_components = engine.place_analytical(def_parser)
    placement_time = time.time() - placement_start
    
    logger.info(f"\nTotal placement time: {placement_time:.2f} seconds")
    
    # Write output
    base_name = os.path.splitext(input_def)[0]
    output_def = f"{base_name}.output.def"
    
    def_parser.write(output_def, optimized_components)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SUCCESS! Optimized placement written to: {output_def}")
    logger.info(f"{'='*80}\n")
    
    # Summary
    logger.info("SUMMARY:")
    logger.info(f"  Input:  {input_def}")
    logger.info(f"  Output: {output_def}")
    logger.info(f"  Components: {len(optimized_components)}")
    logger.info(f"  Nets: {len(def_parser.nets)}")
    logger.info(f"  Runtime: {placement_time:.2f}s")

if __name__ == "__main__":
    main()