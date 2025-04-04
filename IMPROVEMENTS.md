# Plan Quality Evaluation Improvements

## Overview

We have implemented a comprehensive plan quality evaluation system for QuangTPS, mimicking the functionality available in Eclipse TPS. The system includes clinical protocol management, goal evaluation, and robust reporting capabilities.

## Key Features

### Clinical Protocol Management

- **Protocol Storage**: Protocols are stored as JSON files in the `protocols` directory
- **Protocol Dialog**: A dialog for selecting, importing, and exporting protocols
- **Protocol Editor**: A dedicated editor for creating and modifying protocols, with support for:
  - Editing protocol details (name, description)
  - Adding, editing, and removing clinical goals
  - Importing goals from other protocols

### Plan Quality Evaluation

- **Goal Evaluation**: Each clinical goal is evaluated against actual dose metrics
- **Scoring System**: Overall, target, and OAR scores calculated based on goal achievements
- **Visual Feedback**: Color-coded indicators for passed, acceptable, and failed goals

### Reporting

- **HTML Reports**: Generate comprehensive HTML reports showing:
  - Overall quality scores
  - Individual goal results
  - Progress bars and visual indicators
  - Summary assessment

## Integration with QuangTPS

The plan quality functionality is fully integrated with the existing evaluation tab, allowing users to:

1. Select clinical protocols from a dropdown menu or dedicated dialog
2. View plan quality assessment in a dedicated tab
3. Generate and export plan quality reports
4. Edit protocols directly from the interface

## Sample Protocols

Two sample protocols are included:

1. **Head and Neck**: Containing goals for PTVs, spinal cord, brainstem, parotids, and other OARs
2. **Prostate**: Containing goals for PTV, rectum, bladder, femoral heads, and penile bulb

## Eclipse-Like Experience

The interface closely resembles Eclipse's plan quality evaluation system, with:

- Similar scoring visualization
- Progress bars for goal achievement
- Color-coded indicators (green, orange, red)
- Detailed goal display table

## Technical Implementation

- **Modular Design**: Separated protocol management, evaluation, and UI components
- **Exception Handling**: Robust error handling for missing components
- **Clean Integration**: Minimal changes to existing code while adding significant functionality 