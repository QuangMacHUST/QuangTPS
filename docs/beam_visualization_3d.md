# 3D Beam Visualization Module

The 3D Beam Visualization Module provides advanced three-dimensional visualization of radiotherapy beams and treatment setup within the QuangTPS treatment planning system.

## Features

- Interactive 3D visualization of patient anatomy, beams, and dose distributions
- View treatment machine geometry including gantry, couch, and collimator
- Manipulate beam parameters and see real-time updates
- Visualize isodose surfaces and dose distributions
- Multiple view modes and camera angles
- Support for various treatment modalities (3D-CRT, IMRT, VMAT)

## Dependencies

This module requires additional Python packages:

- PyVista (>=0.38.1)
- PyVistaQt (>=0.9.0)
- VTK (>=9.2.6)

These dependencies are automatically detected and can be installed through the QuangTPS dependency installer when the module is first accessed.

## Getting Started

### Accessing the 3D Visualization

The 3D visualization module is integrated into the Planning tab under the "Beams" section. When selected, the system will automatically check for required dependencies and offer to install them if needed.

### Basic Navigation

- **Rotate**: Click and drag with the left mouse button
- **Pan**: Click and drag with the middle mouse button
- **Zoom**: Use the mouse wheel or right-click and drag
- **Reset View**: Use the view buttons (Anterior, Posterior, etc.)

### Customizing the View

The module provides various display options to customize the visualization:

- Show/hide beams
- Show/hide structures
- Show/hide isodose surfaces
- Show/hide treatment machine
- Show/hide coordinate axes

## Beam Manipulation

The 3D visualization module allows direct manipulation of beam parameters:

1. Select a beam in the visualization or from the beam list
2. Modify parameters in the control panel (gantry angle, field size, etc.)
3. Apply changes to see the updated beam geometry in real-time

## Using the Beam Visualization Panel

The Beam Visualization Panel provides comprehensive tools for managing and visualizing radiotherapy beams:

### Beam List View

The beam list displays all beams in the current plan with their properties:
- Name
- Gantry angle
- Collimator angle
- Couch angle
- Field size
- Weight

### Beam Editing

Select a beam to edit its properties:
- Beam name
- Angles (gantry, collimator, couch)
- Field size
- Energy
- Weight
- Isocenter position
- MLC configuration

### Beam Operations

Common operations include:
- Add new beam
- Remove beam
- Copy beam
- Calculate dose for individual beam

## Integration with Dose Calculation

The 3D visualization module integrates with dose calculation to visualize:
- Dose distributions as colorwash
- Isodose surfaces at user-defined levels
- DVH calculation based on visualization

## Troubleshooting

### Missing 3D Visualization

If the 3D visualization is not available:

1. Ensure that PyVista, PyVistaQt, and VTK are installed
2. Run the dependency installer manually:
   ```python
   from quangtps.ui.dependency_installer import check_and_install_feature_dependencies
   check_and_install_feature_dependencies("3d_visualization")
   ```

3. Check for graphics driver issues or OpenGL support on your system

### Performance Issues

If the 3D visualization is slow:

1. Reduce the number of displayed structures
2. Lower the resolution of isodose surfaces
3. Hide the treatment machine representation
4. Use a simplified patient outline mode

## Advanced Configuration

Advanced users can configure the visualization module by modifying settings in the configuration file:

```
[Visualization3D]
default_view = anterior
show_machine = true
show_axes = true
isodose_levels = 95, 80, 70, 50, 30, 10
structure_opacity = 0.5
dose_opacity = 0.7
```

## Developer Information

For developers looking to extend the 3D visualization functionality:

- The module is built on PyVista, which provides a high-level interface to VTK
- New visualization elements can be added by extending the existing classes
- Custom beam representations can be implemented by overriding the `_add_beam_representation` method
- Performance-critical operations are optimized for large datasets
