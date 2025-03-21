#!/usr/bin/env python3
"""
Generate SVG icons for QuangTPS application.
This script creates modern, professional icons for the radiotherapy treatment planning system.
"""

import os
import math

# Colors based on the new style
COLORS = {
    'primary': '#0984e3',     # Main blue color
    'secondary': '#74b9ff',   # Lighter blue
    'accent': '#00b894',      # Green accent
    'warning': '#fdcb6e',     # Yellow warning
    'danger': '#e84118',      # Red danger
    'dark': '#2d3436',        # Dark background
    'light': '#dcdde1',       # Light text/icons
    'white': '#ffffff',       # White
}

# Base directory for the icons
ICON_DIR = os.path.dirname(os.path.abspath(__file__))

def write_svg(filename, content):
    """Write SVG content to a file."""
    with open(os.path.join(ICON_DIR, filename), 'w') as f:
        f.write(content)
    print(f"Created {filename}")

# Application icon
def create_app_icon():
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{COLORS['primary']};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{COLORS['secondary']};stop-opacity:1" />
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="30" fill="url(#grad1)" />
  <path d="M22,20 L42,20 A2,2 0 0,1 44,22 L44,42 A2,2 0 0,1 42,44 L22,44 A2,2 0 0,1 20,42 L20,22 A2,2 0 0,1 22,20 Z" 
        fill="{COLORS['white']}" opacity="0.9"/>
  <circle cx="32" cy="32" r="6" fill="{COLORS['accent']}" />
  <path d="M32,16 L32,24 M16,32 L24,32 M32,40 L32,48 M40,32 L48,32" 
        stroke="{COLORS['white']}" stroke-width="3" stroke-linecap="round"/>
</svg>'''
    write_svg('app_icon.svg', svg)

# Patient icons
def create_patient_icons():
    # New patient icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="22" r="14" fill="{COLORS['primary']}" />
  <path d="M14,54 L14,46 C14,38 22,36 32,36 C42,36 50,38 50,46 L50,54" 
        stroke="{COLORS['primary']}" stroke-width="6" fill="none" stroke-linejoin="round" />
  <line x1="48" y1="12" x2="48" y2="32" stroke="{COLORS['accent']}" stroke-width="4" stroke-linecap="round" />
  <line x1="38" y1="22" x2="58" y2="22" stroke="{COLORS['accent']}" stroke-width="4" stroke-linecap="round" />
</svg>'''
    write_svg('new_patient.svg', svg)
    
    # Open patient icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="22" r="14" fill="{COLORS['primary']}" />
  <path d="M14,54 L14,46 C14,38 22,36 32,36 C42,36 50,38 50,46 L50,54" 
        stroke="{COLORS['primary']}" stroke-width="6" fill="none" stroke-linejoin="round" />
  <path d="M46,42 L52,36 L58,42" stroke="{COLORS['accent']}" stroke-width="3" fill="none" stroke-linecap="round" />
  <line x1="52" y1="36" x2="52" y2="54" stroke="{COLORS['accent']}" stroke-width="3" stroke-linecap="round" />
</svg>'''
    write_svg('open_patient.svg', svg)

# Plan related icons
def create_plan_icons():
    # New plan
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="44" height="44" rx="4" fill="{COLORS['primary']}" />
  <line x1="20" y1="20" x2="44" y2="20" stroke="{COLORS['white']}" stroke-width="3" stroke-linecap="round" />
  <line x1="20" y1="32" x2="44" y2="32" stroke="{COLORS['white']}" stroke-width="3" stroke-linecap="round" />
  <line x1="20" y1="44" x2="44" y2="44" stroke="{COLORS['white']}" stroke-width="3" stroke-linecap="round" />
  <line x1="54" y1="32" x2="64" y2="32" stroke="{COLORS['accent']}" stroke-width="4" stroke-linecap="round" />
  <line x1="59" y1="27" x2="59" y2="37" stroke="{COLORS['accent']}" stroke-width="4" stroke-linecap="round" />
</svg>'''
    write_svg('new_plan.svg', svg)
    
    # Optimization icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{COLORS['danger']};stop-opacity:1" />
      <stop offset="50%" style="stop-color:{COLORS['warning']};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{COLORS['accent']};stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="48" height="48" rx="3" fill="{COLORS['primary']}" opacity="0.2" />
  <path d="M12,52 L52,12" stroke="{COLORS['primary']}" stroke-width="4" stroke-linecap="round" />
  <circle cx="18" cy="46" r="5" fill="url(#grad1)" />
  <circle cx="32" cy="32" r="5" fill="url(#grad1)" />
  <circle cx="46" cy="18" r="5" fill="url(#grad1)" />
</svg>'''
    write_svg('optimize.svg', svg)
    
    # Evaluation icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <path d="M12,44 L22,32 L32,40 L52,16" stroke="{COLORS['primary']}" stroke-width="4" fill="none" />
  <path d="M8,8 L8,56 L56,56" stroke="{COLORS['primary']}" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round" />
  <circle cx="22" cy="32" r="4" fill="{COLORS['accent']}" />
  <circle cx="32" cy="40" r="4" fill="{COLORS['accent']}" />
  <circle cx="52" cy="16" r="4" fill="{COLORS['accent']}" />
</svg>'''
    write_svg('evaluate.svg', svg)

# Function icons
def create_function_icons():
    # Imaging icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="8" width="48" height="48" rx="4" fill="{COLORS['primary']}" />
  <rect x="12" y="12" width="40" height="40" rx="2" fill="{COLORS['white']}" opacity="0.9" />
  <rect x="16" y="16" width="14" height="14" fill="{COLORS['primary']}" opacity="0.7" />
  <rect x="34" y="16" width="14" height="14" fill="{COLORS['dark']}" opacity="0.3" />
  <rect x="16" y="34" width="14" height="14" fill="{COLORS['dark']}" opacity="0.3" />
  <rect x="34" y="34" width="14" height="14" fill="{COLORS['primary']}" opacity="0.7" />
</svg>'''
    write_svg('imaging.svg', svg)
    
    # Contouring icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="8" width="48" height="48" rx="2" fill="{COLORS['primary']}" opacity="0.2" />
  <path d="M16,16 C36,10 48,28 48,48" stroke="{COLORS['primary']}" stroke-width="3" fill="none" />
  <path d="M16,16 C36,10 48,28 48,48" stroke="{COLORS['primary']}" stroke-width="6" fill="none" stroke-dasharray="2,8" />
  <circle cx="16" cy="16" r="4" fill="{COLORS['primary']}" />
  <circle cx="48" cy="48" r="4" fill="{COLORS['primary']}" />
  <circle cx="26" cy="13" r="3" fill="{COLORS['accent']}" />
  <circle cx="36" cy="18" r="3" fill="{COLORS['accent']}" />
  <circle cx="44" cy="28" r="3" fill="{COLORS['accent']}" />
  <circle cx="47" cy="38" r="3" fill="{COLORS['accent']}" />
</svg>'''
    write_svg('contouring.svg', svg)
    
    # Dose icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="doseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{COLORS['primary']};stop-opacity:0.2" />
      <stop offset="100%" style="stop-color:{COLORS['primary']};stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="48" height="48" rx="2" fill="{COLORS['primary']}" opacity="0.1" />
  <circle cx="32" cy="32" r="20" fill="url(#doseGrad)" opacity="0.6" />
  <circle cx="32" cy="32" r="16" fill="url(#doseGrad)" opacity="0.7" />
  <circle cx="32" cy="32" r="12" fill="url(#doseGrad)" opacity="0.8" />
  <circle cx="32" cy="32" r="8" fill="url(#doseGrad)" opacity="0.9" />
  <circle cx="32" cy="32" r="4" fill="{COLORS['primary']}" />
</svg>'''
    write_svg('dose.svg', svg)
    
    # Planning icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="8" width="48" height="48" rx="4" fill="{COLORS['primary']}" opacity="0.1" />
  <path d="M10,40 L18,40 L24,24 L34,48 L42,32 L48,32 L54,32" 
        stroke="{COLORS['primary']}" stroke-width="4" fill="none" stroke-linejoin="round" stroke-linecap="round" />
  <path d="M16,18 L48,18" stroke="{COLORS['primary']}" stroke-width="3" stroke-linecap="round" />
  <path d="M16,54 L48,54" stroke="{COLORS['primary']}" stroke-width="3" stroke-linecap="round" />
  <circle cx="24" cy="24" r="3" fill="{COLORS['accent']}" />
  <circle cx="34" cy="48" r="3" fill="{COLORS['accent']}" />
  <circle cx="42" cy="32" r="3" fill="{COLORS['accent']}" />
</svg>'''
    write_svg('planning.svg', svg)
    
    # Treatment icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="14" width="44" height="36" rx="4" fill="{COLORS['primary']}" />
  <rect x="14" y="18" width="36" height="28" fill="{COLORS['white']}" opacity="0.9" />
  <rect x="20" y="10" width="24" height="8" rx="2" fill="{COLORS['primary']}" />
  <path d="M26,28 L38,28 L38,38 L26,38 Z" fill="{COLORS['primary']}" />
  <rect x="20" y="46" width="8" height="8" rx="1" fill="{COLORS['accent']}" />
  <rect x="36" y="46" width="8" height="8" rx="1" fill="{COLORS['accent']}" />
</svg>'''
    write_svg('treatment.svg', svg)
    
    # QA icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="24" fill="{COLORS['primary']}" opacity="0.1" />
  <path d="M32,16 L38,28 L52,30 L42,40 L44,54 L32,48 L20,54 L22,40 L12,30 L26,28 Z" 
        fill="{COLORS['primary']}" />
  <circle cx="32" cy="32" r="8" fill="{COLORS['white']}" />
  <path d="M29,29 L35,35 M29,35 L35,29" stroke="{COLORS['accent']}" stroke-width="3" stroke-linecap="round" />
</svg>'''
    write_svg('qa.svg', svg)
    
    # Reporting icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <path d="M14,10 L50,10 L50,54 L14,54 Z" fill="{COLORS['white']}" stroke="{COLORS['primary']}" stroke-width="3" />
  <path d="M20,20 L44,20 M20,28 L44,28 M20,36 L44,36 M20,44 L36,44" 
        stroke="{COLORS['primary']}" stroke-width="2" stroke-linecap="round" />
  <circle cx="46" cy="46" r="12" fill="{COLORS['accent']}" />
  <path d="M46,40 L46,52 M40,46 L52,46" stroke="{COLORS['white']}" stroke-width="3" stroke-linecap="round" />
</svg>'''
    write_svg('report.svg', svg)

# Import/Export icons
def create_data_icons():
    # Import icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="8" width="48" height="48" rx="4" fill="{COLORS['primary']}" opacity="0.1" />
  <path d="M16,32 L48,32 M16,32 L26,22 M16,32 L26,42" 
        stroke="{COLORS['primary']}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  <path d="M32,16 L48,16 M32,48 L48,48" 
        stroke="{COLORS['primary']}" stroke-width="4" stroke-linecap="round" />
</svg>'''
    write_svg('import.svg', svg)
    
    # Export icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="8" width="48" height="48" rx="4" fill="{COLORS['primary']}" opacity="0.1" />
  <path d="M48,32 L16,32 M48,32 L38,22 M48,32 L38,42" 
        stroke="{COLORS['primary']}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
  <path d="M32,16 L16,16 M32,48 L16,48" 
        stroke="{COLORS['primary']}" stroke-width="4" stroke-linecap="round" />
</svg>'''
    write_svg('export.svg', svg)

# Help icons
def create_help_icons():
    # About icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="24" fill="{COLORS['primary']}" />
  <circle cx="32" cy="20" r="4" fill="{COLORS['white']}" />
  <path d="M32,28 L32,48" stroke="{COLORS['white']}" stroke-width="6" stroke-linecap="round" />
</svg>'''
    write_svg('about.svg', svg)
    
    # Help icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="24" fill="{COLORS['primary']}" />
  <path d="M32,36 L32,28 C32,24 36,24 36,24 C40,24 40,28 36,30 C34,31 32,32 32,36" 
        stroke="{COLORS['white']}" stroke-width="4" fill="none" />
  <circle cx="32" cy="44" r="3" fill="{COLORS['white']}" />
</svg>'''
    write_svg('help.svg', svg)
    
    # Exit icon
    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect x="8" y="8" width="40" height="48" rx="4" fill="{COLORS['primary']}" />
  <path d="M48,32 L24,32 M48,32 L40,24 M48,32 L40,40" 
        stroke="{COLORS['white']}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
</svg>'''
    write_svg('exit.svg', svg)

if __name__ == "__main__":
    # Create the icons
    create_app_icon()
    create_patient_icons()
    create_plan_icons()
    create_function_icons()
    create_data_icons()
    create_help_icons()
    
    print("All icons created successfully!") 