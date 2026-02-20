from graphviz import Digraph

# Create Digraph object
dot = Digraph('Soil_Robot_Straight', comment='Ortho Flowchart')

# 1. CONFIGURATION
# splines='ortho' forces straight lines (right angles only).
# nodesep=0.8 adds horizontal breathing room.
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='0.8')
dot.attr('node', shape='rect', style='rounded,filled', fontname='Arial', fontsize='12')

# 2. DEFINE NODES
dot.node('Start', 'Start', shape='circle', fillcolor='#333333', fontcolor='white', width='0.8')
dot.node('Iterate', 'Iterate over\nSoil Bags', shape='diamond', fillcolor='#E1BEE7', height='1.2')
dot.node('Gantry', 'Gantry Loop', fillcolor='#BBDEFB')
dot.node('Display', 'Display Results', shape='parallelogram', fillcolor='#C8E6C9')
dot.node('End', 'End', shape='circle', fillcolor='#333333', fontcolor='white', width='0.6')

# --- Main Loop Cluster ---
with dot.subgraph(name='cluster_processing') as c:
    c.attr(style='dashed', color='grey', label='Processing Cycle')
    
    # Main Controller
    c.node('Main', 'Main Loop', fillcolor='#FFF9C4', width='2.5')
    
    # Hardware Actions - Forced to stay in one horizontal row
    with c.subgraph() as hardware:
        hardware.attr(rank='same')
        hardware.node('Stir', 'Stir')
        hardware.node('Scoop', 'Scoop')
        hardware.node('Scan', 'Scan Soil\n(AI)')
        hardware.node('Unscoop', 'Unscoop')

# 3. DEFINE EDGES (Logic Flow)

# --- STARTUP ---
dot.edge('Start', 'Iterate', label='Start Coords')

# --- GANTRY (LEFT SIDE LOOP) ---
# Exit West (Left) -> Enter Top of Gantry
dot.edge('Iterate', 'Gantry', label='Next Loc', tailport='w', headport='n')
# Exit Bottom of Gantry -> Return to West (Left) of Iterate
dot.edge('Gantry', 'Iterate', label='Confirm', tailport='s', headport='w', color='#D32F2F')

# --- MAIN LOOP ENTRY ---
dot.edge('Iterate', 'Main', label='Start Process', tailport='s', headport='n')

# --- HARDWARE ACTIONS (THE FORK) ---
# We use compass points (sw, s, se) to spread the arrows out

# Stir
dot.edge('Main', 'Stir', label='Actuate', tailport='sw', headport='n')
dot.edge('Stir', 'Main', label='Done', tailport='s', headport='w', color='#D32F2F')

# Scoop
dot.edge('Main', 'Scoop', label='Actuate', tailport='s', headport='n')
dot.edge('Scoop', 'Main', label='Done', tailport='s', headport='w', color='#D32F2F')

# Scan
dot.edge('Main', 'Scan', label='Scan', tailport='s', headport='n')
dot.edge('Scan', 'Main', label='Data', tailport='s', headport='e', color='#D32F2F')

# Unscoop
dot.edge('Main', 'Unscoop', label='Release', tailport='se', headport='n')
dot.edge('Unscoop', 'Main', label='Confirm', tailport='s', headport='e', color='#D32F2F')

# --- MAIN LOOP RETURN (RIGHT SIDE LOOP) ---
# Loop from Main back to Iterate on the right side
dot.edge('Main', 'Iterate', label='Cycle Complete', tailport='e', headport='e', color='#D32F2F')

# --- END SEQUENCE (FAR RIGHT) ---
dot.edge('Iterate', 'Display', label='Last Item', tailport='e', headport='n')
dot.edge('Display', 'End')

# Render
dot.render('soil_robot_ortho', view=True, format='png')