"""
Script to generate sample config.json and sample.def for testing
"""

import json
import os

# Create config.json
config = {
    "design_name": "test_design",
    "def_input": "sample.def",
    "result_dir": "./results",
    "learning_rate": 0.001,
    "num_iterations": 100,
    "target_density": 0.8
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Created config.json")

# Create sample.def
def_content = """VERSION 5.8 ;
DESIGN sample_design ;
UNITS DISTANCE MICRONS 2000 ;

DIEAREA ( 0 0 ) ( 100000 100000 ) ;

COMPONENTS 20 ;
  - cell_0 NAND2_X1 + PLACED ( 0 0 ) N ;
  - cell_1 NOR2_X1 + PLACED ( 0 0 ) N ;
  - cell_2 INV_X1 + PLACED ( 0 0 ) N ;
  - cell_3 BUF_X2 + PLACED ( 0 0 ) N ;
  - cell_4 AND2_X1 + PLACED ( 0 0 ) N ;
  - cell_5 OR2_X1 + PLACED ( 0 0 ) N ;
  - cell_6 XOR2_X1 + PLACED ( 0 0 ) N ;
  - cell_7 NAND2_X1 + PLACED ( 0 0 ) N ;
  - cell_8 NOR2_X1 + PLACED ( 0 0 ) N ;
  - cell_9 INV_X1 + PLACED ( 0 0 ) N ;
  - cell_10 BUF_X2 + PLACED ( 0 0 ) N ;
  - cell_11 AND2_X1 + PLACED ( 0 0 ) N ;
  - cell_12 OR2_X1 + PLACED ( 0 0 ) N ;
  - cell_13 XOR2_X1 + PLACED ( 0 0 ) N ;
  - cell_14 NAND2_X1 + PLACED ( 0 0 ) N ;
  - cell_15 NOR2_X1 + PLACED ( 0 0 ) N ;
  - cell_16 INV_X1 + PLACED ( 0 0 ) N ;
  - cell_17 BUF_X2 + PLACED ( 0 0 ) N ;
  - cell_18 AND2_X1 + PLACED ( 0 0 ) N ;
  - cell_19 OR2_X1 + PLACED ( 0 0 ) N ;
END COMPONENTS

NETS 15 ;
  - net_0
    ( cell_0 A ) ( cell_1 A ) ;
  - net_1
    ( cell_1 Y ) ( cell_2 A ) ;
  - net_2
    ( cell_2 Y ) ( cell_3 A ) ;
  - net_3
    ( cell_3 Y ) ( cell_4 A ) ;
  - net_4
    ( cell_4 Y ) ( cell_5 A ) ( cell_6 A ) ;
  - net_5
    ( cell_5 Y ) ( cell_7 A ) ;
  - net_6
    ( cell_6 Y ) ( cell_8 A ) ;
  - net_7
    ( cell_7 Y ) ( cell_9 A ) ;
  - net_8
    ( cell_8 Y ) ( cell_10 A ) ;
  - net_9
    ( cell_9 Y ) ( cell_11 A ) ( cell_12 A ) ;
  - net_10
    ( cell_10 Y ) ( cell_13 A ) ;
  - net_11
    ( cell_11 Y ) ( cell_14 A ) ;
  - net_12
    ( cell_12 Y ) ( cell_15 A ) ;
  - net_13
    ( cell_13 Y ) ( cell_16 A ) ( cell_17 A ) ;
  - net_14
    ( cell_14 Y ) ( cell_18 A ) ( cell_19 A ) ;
END NETS

END DESIGN
"""

with open('sample.def', 'w') as f:
    f.write(def_content)

print("✅ Created sample.def")
print("\nNow run:")
print("  python RLPlacer.py config.json")