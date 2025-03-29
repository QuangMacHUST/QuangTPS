#!/usr/bin/env python
"""
QA Report Sample Script

This script demonstrates how to use the QuangTPS scripting API to generate
a quality assurance report for a treatment plan.
"""

import os
import time
import json
from datetime import datetime

# Try to import the API - this will work when run within QuangTPS
try:
    import api
except ImportError:
    print("This script is designed to run within QuangTPS script editor")
    print("If running standalone, please use a proper QuangTPS Python environment")
    exit(1)

# QA criteria for different treatment sites
QA_CRITERIA = {
    "General": {
        "PTV": {
            "D95": {"goal": ">= 95%", "priority": "Critical"},
            "Homogeneity Index": {"goal": "<= 0.07", "priority": "Important"},
            "Conformity Index": {"goal": "<= 1.2", "priority": "Important"}
        },
        "Spinal Cord": {
            "Max Dose": {"goal": "< 45 Gy", "priority": "Critical"}
        },
        "Brainstem": {
            "Max Dose": {"goal": "< 54 Gy", "priority": "Critical"}
        }
    },
    "Prostate": {
        "PTV": {
            "D95": {"goal": ">= 95%", "priority": "Critical"},
            "V95%": {"goal": ">= 99%", "priority": "Critical"},
            "Homogeneity Index": {"goal": "<= 0.1", "priority": "Important"}
        },
        "Rectum": {
            "V70Gy": {"goal": "<= 20%", "priority": "Critical"},
            "V50Gy": {"goal": "<= 50%", "priority": "Important"},
            "V40Gy": {"goal": "<= 60%", "priority": "Important"}
        },
        "Bladder": {
            "V70Gy": {"goal": "<= 35%", "priority": "Critical"},
            "V50Gy": {"goal": "<= 50%", "priority": "Important"}
        },
        "Femoral Heads": {
            "V50Gy": {"goal": "<= 5%", "priority": "Important"}
        }
    },
    "Lung": {
        "PTV": {
            "D95": {"goal": ">= 95%", "priority": "Critical"},
            "V95%": {"goal": ">= 99%", "priority": "Critical"}
        },
        "Lung-GTV": {
            "V20Gy": {"goal": "<= 30%", "priority": "Critical"},
            "V5Gy": {"goal": "<= 60%", "priority": "Important"},
            "Mean Dose": {"goal": "<= 20 Gy", "priority": "Important"}
        },
        "Heart": {
            "V30Gy": {"goal": "<= 45%", "priority": "Critical"},
            "Mean Dose": {"goal": "<= 26 Gy", "priority": "Important"}
        },
        "Esophagus": {
            "Mean Dose": {"goal": "<= 34 Gy", "priority": "Important"}
        },
        "Spinal Cord": {
            "Max Dose": {"goal": "<= 45 Gy", "priority": "Critical"}
        }
    }
}

def generate_qa_report(site_type=None):
    """Generate a QA report for the current plan."""
    print("QA Report Script - Starting")
    
    # Get the current patient
    patients = api.get_patients()
    if not patients:
        print("No patients found in the database")
        return
    
    # For demonstration, use the first patient
    patient = api.get_patient_by_id(patients[0]['id'])
    if not patient:
        print("Failed to get patient")
        return
    
    if not api.set_current_patient(patient):
        print("Failed to set current patient")
        return
    
    print(f"Generating QA report for patient: {patient['name']}")
    
    # Get available plans
    plans = api.get_plans()
    if not plans:
        print("No plans found for this patient")
        return
    
    # For demonstration, use the first plan
    plan = plans[0]
    if not api.set_current_plan(plan):
        print("Failed to set current plan")
        return
    
    print(f"Analyzing plan: {plan['name']}")
    
    # Detect plan type if not specified
    if not site_type:
        site_type = detect_plan_type(plan)
    
    print(f"Using QA criteria for: {site_type}")
    
    # Get structures
    structures = api.get_structures()
    if not structures:
        print("No structures found in this plan")
        return
    
    # Get applicable QA criteria
    qa_criteria = get_qa_criteria(site_type)
    
    # Evaluate all QA criteria
    qa_results = evaluate_qa_criteria(qa_criteria, structures)
    
    # Generate the report file
    report_file = create_report_file(patient, plan, site_type, qa_results)
    print(f"QA report generated: {report_file}")
    
    print("QA Report Script - Completed")

def detect_plan_type(plan):
    """Try to determine plan type based on structures and naming."""
    # In a real implementation, this would analyze structures, targets, etc.
    # For this demo, we'll simplify and just check the plan name
    plan_name = plan['name'].lower()
    
    if "prostate" in plan_name:
        return "Prostate"
    elif "lung" in plan_name:
        return "Lung"
    elif "breast" in plan_name:
        return "Breast"
    elif "brain" in plan_name or "srs" in plan_name:
        return "Brain"
    elif "h&n" in plan_name or "head" in plan_name or "neck" in plan_name:
        return "Head and Neck"
    else:
        return "General"

def get_qa_criteria(site_type):
    """Get the QA criteria for the given site type."""
    # Check if we have specific criteria for this site type
    if site_type in QA_CRITERIA:
        return QA_CRITERIA[site_type]
    else:
        # Fall back to general criteria
        return QA_CRITERIA["General"]

def evaluate_qa_criteria(qa_criteria, structures):
    """Evaluate the QA criteria against the current plan."""
    qa_results = {}
    
    # Get the plan evaluation results from the TPS
    evaluation = api.evaluate_plan()
    
    # For each structure in the criteria
    for structure_name, criteria in qa_criteria.items():
        # Find structure in the actual plan
        structure = next((s for s in structures if s['name'] == structure_name), None)
        
        if not structure:
            # Check for similar names (e.g., "PTV70" matches criteria for "PTV")
            structure = next((s for s in structures if structure_name in s['name']), None)
        
        # Skip if structure is not found
        if not structure:
            print(f"Warning: Structure {structure_name} not found")
            continue
        
        # Initialize results for this structure
        qa_results[structure_name] = {}
        
        # Evaluate each metric for this structure
        for metric, metric_criteria in criteria.items():
            # Find matching evaluation result
            result = None
            for eval_result in evaluation:
                if eval_result['structure'] == structure_name and eval_result['metric'] == metric:
                    result = eval_result
                    break
            
            # If not found in evaluation results, calculate it
            if not result:
                result = calculate_metric(structure, metric)
            
            # Skip if the metric couldn't be calculated
            if not result:
                print(f"Warning: Metric {metric} could not be calculated for {structure_name}")
                continue
            
            # Determine if the result passes the criteria
            passes = evaluate_result(result['value'], metric_criteria['goal'])
            
            # Store the result
            qa_results[structure_name][metric] = {
                'value': result['value'],
                'goal': metric_criteria['goal'],
                'result': 'PASS' if passes else 'FAIL',
                'priority': metric_criteria['priority']
            }
    
    return qa_results

def calculate_metric(structure, metric):
    """Calculate a specific metric for a structure."""
    # In a real implementation, this would calculate the metric using the TPS
    # For this example, we'll just return a placeholder result
    
    dvh_data = api.calculate_dvh(structure)
    if not dvh_data:
        return None
    
    # Mock some values for demonstration
    if "D95" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '96.5%'}
    elif "V95" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '99.2%'}
    elif "Homogeneity Index" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '0.05'}
    elif "Conformity Index" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '1.1'}
    elif "Max Dose" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '40.2 Gy'}
    elif "Mean Dose" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '18.5 Gy'}
    elif "V20Gy" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '25.3%'}
    elif "V30Gy" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '20.1%'}
    elif "V40Gy" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '15.7%'}
    elif "V50Gy" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '10.2%'}
    elif "V70Gy" in metric:
        return {'structure': structure['name'], 'metric': metric, 'value': '8.9%'}
    else:
        return None

def evaluate_result(value_str, goal_str):
    """Evaluate if a result meets the goal criteria."""
    # Parse the goal string (e.g., "< 20 Gy", ">= 95%")
    operator = None
    if "<=" in goal_str:
        operator = "<="
        goal_value = goal_str.replace("<=", "").strip()
    elif ">=" in goal_str:
        operator = ">="
        goal_value = goal_str.replace(">=", "").strip()
    elif "<" in goal_str:
        operator = "<"
        goal_value = goal_str.replace("<", "").strip()
    elif ">" in goal_str:
        operator = ">"
        goal_value = goal_str.replace(">", "").strip()
    else:
        return False  # Invalid goal string
    
    # Convert values to numeric
    try:
        if "%" in value_str:
            value = float(value_str.replace("%", "").strip())
        elif "Gy" in value_str:
            value = float(value_str.replace("Gy", "").strip())
        else:
            value = float(value_str)
        
        if "%" in goal_value:
            goal = float(goal_value.replace("%", "").strip())
        elif "Gy" in goal_value:
            goal = float(goal_value.replace("Gy", "").strip())
        else:
            goal = float(goal_value)
    except ValueError:
        return False  # Invalid numeric conversion
    
    # Compare based on operator
    if operator == "<=":
        return value <= goal
    elif operator == ">=":
        return value >= goal
    elif operator == "<":
        return value < goal
    elif operator == ">":
        return value > goal
    else:
        return False  # Shouldn't reach here

def create_report_file(patient, plan, site_type, qa_results):
    """Create a QA report file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"QA_Report_{patient['id']}_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"QuangTPS QA Report\n")
        f.write(f"=====================================\n\n")
        f.write(f"Patient: {patient['name']} (ID: {patient['id']})\n")
        f.write(f"Plan: {plan['name']}\n")
        f.write(f"Site Type: {site_type}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"QA Criteria Evaluation\n")
        f.write(f"-------------------------------------\n\n")
        
        # Keep track of pass/fail counts
        total_count = 0
        pass_count = 0
        fail_critical = 0
        
        # Write results for each structure
        for structure, metrics in qa_results.items():
            f.write(f"Structure: {structure}\n")
            f.write(f"-------------------------------------\n")
            
            for metric, result in metrics.items():
                total_count += 1
                is_pass = result['result'] == 'PASS'
                if is_pass:
                    pass_count += 1
                elif result['priority'] == 'Critical':
                    fail_critical += 1
                
                status = result['result']
                priority_indicator = "*" if result['priority'] == 'Critical' else ""
                
                f.write(f"{priority_indicator}{metric}: {result['value']} ")
                f.write(f"(Goal: {result['goal']}) - {status}\n")
            
            f.write("\n")
        
        # Write summary
        f.write(f"Summary\n")
        f.write(f"-------------------------------------\n")
        f.write(f"Total QA Criteria: {total_count}\n")
        f.write(f"Criteria Passed: {pass_count} ({pass_count/total_count*100:.1f}%)\n")
        f.write(f"Critical Criteria Failed: {fail_critical}\n\n")
        
        if fail_critical > 0:
            f.write(f"RECOMMENDATION: FAIL - Critical criteria not met\n")
        elif pass_count/total_count < 0.9:
            f.write(f"RECOMMENDATION: REVIEW - Less than 90% of criteria met\n")
        else:
            f.write(f"RECOMMENDATION: PASS - All critical criteria met\n")
    
    return report_file

if __name__ == "__main__":
    # Change this to match your plan type, or leave as None for auto-detection
    generate_qa_report(None) 