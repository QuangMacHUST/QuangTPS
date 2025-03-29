import sys
import os
import inspect
import importlib
import traceback
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QComboBox, QToolBar, QAction, QFileDialog,
    QSplitter, QTreeWidget, QTreeWidgetItem, QMessageBox, QTabWidget,
    QPlainTextEdit, QDockWidget, QMainWindow
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRegExp, QThread, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor, QTextCharFormat, QSyntaxHighlighter

from quangtps.core.logging import get_logger
from quangtps.core.services import ServiceRegistry
from quangtps.database.patient_db import PatientDB
from quangtps.planning.plan import Plan
from quangtps.core.structures import Structure
from quangtps.planning.prescription import DoseConstraint
from quangtps.planning.beam import Beam
from quangtps.planning.evaluation import PlanEvaluation
from quangtps.evaluation.dvh import DVHAnalysis
from quangtps.core.patient import Patient
from quangtps.planning.dose_calculation import DoseCalculationEngine
from quangtps.planning.optimization import OptimizationEngine

logger = get_logger(__name__)

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Python syntax highlighter for the script editor."""
    
    def __init__(self, document):
        super(PythonSyntaxHighlighter, self).__init__(document)
        
        self.highlighting_rules = []
        
        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(120, 40, 180))
        keyword_format.setFontWeight(QFont.Bold)
        
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
            'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
            'True', 'try', 'while', 'with', 'yield'
        ]
        
        for keyword in keywords:
            pattern = QRegExp(r'\b' + keyword + r'\b')
            rule = (pattern, keyword_format)
            self.highlighting_rules.append(rule)
        
        # Class names
        class_format = QTextCharFormat()
        class_format.setForeground(QColor(0, 120, 0))
        class_format.setFontWeight(QFont.Bold)
        pattern = QRegExp(r'\bclass\b \w+')
        rule = (pattern, class_format)
        self.highlighting_rules.append(rule)
        
        # Function names
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(0, 120, 180))
        function_format.setFontWeight(QFont.Bold)
        pattern = QRegExp(r'\bdef\b \w+')
        rule = (pattern, function_format)
        self.highlighting_rules.append(rule)
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(180, 0, 0))
        pattern = QRegExp(r'".*?"')
        pattern.setMinimal(True)
        rule = (pattern, string_format)
        self.highlighting_rules.append(rule)
        
        pattern = QRegExp(r"'.*?'")
        pattern.setMinimal(True)
        rule = (pattern, string_format)
        self.highlighting_rules.append(rule)
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(128, 128, 128))
        pattern = QRegExp(r'#.*$')
        rule = (pattern, comment_format)
        self.highlighting_rules.append(rule)
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(0, 0, 160))
        pattern = QRegExp(r'\b[0-9]+\b')
        rule = (pattern, number_format)
        self.highlighting_rules.append(rule)
    
    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)


class ApiDocWidget(QWidget):
    """Widget to display API documentation."""
    
    def __init__(self, parent=None):
        super(ApiDocWidget, self).__init__(parent)
        self.setup_ui()
        self.populate_api_docs()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # API tree widget
        self.api_tree = QTreeWidget()
        self.api_tree.setHeaderLabels(["API", "Description"])
        self.api_tree.setColumnWidth(0, 200)
        self.api_tree.itemClicked.connect(self.on_api_item_clicked)
        
        # Documentation view
        self.doc_view = QTextEdit()
        self.doc_view.setReadOnly(True)
        
        # Add to layout
        layout.addWidget(QLabel("Available API Functions:"))
        layout.addWidget(self.api_tree)
        layout.addWidget(QLabel("Documentation:"))
        layout.addWidget(self.doc_view)
    
    def populate_api_docs(self):
        """Populate the API documentation tree."""
        # Patient Management
        patient_item = QTreeWidgetItem(self.api_tree, ["Patient Management", "Functions for managing patients"])
        self.add_api_item(patient_item, "get_patients", "Get list of patients")
        self.add_api_item(patient_item, "get_patient_by_id", "Get patient by ID")
        self.add_api_item(patient_item, "set_current_patient", "Set the current patient")
        self.add_api_item(patient_item, "create_patient", "Create a new patient")
        
        # Plan Management
        plan_item = QTreeWidgetItem(self.api_tree, ["Plan Management", "Functions for managing plans"])
        self.add_api_item(plan_item, "get_plans", "Get plans for current patient")
        self.add_api_item(plan_item, "set_current_plan", "Set the current plan")
        self.add_api_item(plan_item, "create_plan", "Create a new plan")
        self.add_api_item(plan_item, "export_plan", "Export plan to DICOM")
        
        # Structure Management
        structure_item = QTreeWidgetItem(self.api_tree, ["Structure Management", "Functions for managing structures"])
        self.add_api_item(structure_item, "get_structures", "Get structures in current plan")
        self.add_api_item(structure_item, "create_structure", "Create a new structure")
        self.add_api_item(structure_item, "edit_structure", "Edit structure contours")
        
        # Beam Management
        beam_item = QTreeWidgetItem(self.api_tree, ["Beam Management", "Functions for managing beams"])
        self.add_api_item(beam_item, "get_beams", "Get beams in current plan")
        self.add_api_item(beam_item, "create_beam", "Create a new beam")
        self.add_api_item(beam_item, "edit_beam", "Edit beam parameters")
        
        # Dose Calculation
        dose_item = QTreeWidgetItem(self.api_tree, ["Dose Calculation", "Functions for dose calculation"])
        self.add_api_item(dose_item, "calculate_dose", "Calculate dose for current plan")
        self.add_api_item(dose_item, "calculate_dvh", "Calculate DVH for a structure")
        
        # Optimization
        opt_item = QTreeWidgetItem(self.api_tree, ["Optimization", "Functions for plan optimization"])
        self.add_api_item(opt_item, "optimize_plan", "Optimize current plan")
        self.add_api_item(opt_item, "add_objective", "Add optimization objective")
        
        # Evaluation
        eval_item = QTreeWidgetItem(self.api_tree, ["Evaluation", "Functions for plan evaluation"])
        self.add_api_item(eval_item, "evaluate_plan", "Evaluate plan against clinical goals")
        self.add_api_item(eval_item, "get_evaluation_results", "Get evaluation results")
        
        # Expand all items
        self.api_tree.expandAll()
    
    def add_api_item(self, parent, name, description):
        """Add an API item to the tree."""
        return QTreeWidgetItem(parent, [name, description])
    
    def on_api_item_clicked(self, item, column):
        """Display documentation for the selected API item."""
        api_name = item.text(0)
        
        # Only show documentation for leaf items
        if item.childCount() == 0:
            doc_text = self.get_api_doc(api_name)
            self.doc_view.setHtml(doc_text)
    
    def get_api_doc(self, api_name):
        """Get documentation for an API function."""
        docs = {
            "get_patients": """
                <h3>get_patients()</h3>
                <p>Returns a list of all patients in the database.</p>
                <h4>Parameters:</h4>
                <p>None</p>
                <h4>Returns:</h4>
                <p>List of dictionaries, each containing patient information:</p>
                <pre>
                [
                    {
                        'id': 'patient_id',
                        'name': 'Patient Name',
                        'mrn': 'Medical Record Number',
                        'dob': 'Date of Birth'
                    },
                    ...
                ]
                </pre>
                <h4>Example:</h4>
                <pre>
                patients = api.get_patients()
                for patient in patients:
                    print(f"Patient: {patient['name']}, MRN: {patient['mrn']}")
                </pre>
            """,
            
            "get_patient_by_id": """
                <h3>get_patient_by_id(patient_id)</h3>
                <p>Returns a patient by ID.</p>
                <h4>Parameters:</h4>
                <ul>
                    <li><b>patient_id</b> (str): The ID of the patient to retrieve.</li>
                </ul>
                <h4>Returns:</h4>
                <p>Dictionary containing patient information or None if not found.</p>
                <h4>Example:</h4>
                <pre>
                patient = api.get_patient_by_id("12345")
                if patient:
                    print(f"Found patient: {patient['name']}")
                else:
                    print("Patient not found")
                </pre>
            """,
            
            "set_current_patient": """
                <h3>set_current_patient(patient)</h3>
                <p>Sets the current patient for the script context.</p>
                <h4>Parameters:</h4>
                <ul>
                    <li><b>patient</b> (dict): The patient dictionary to set as current.</li>
                </ul>
                <h4>Returns:</h4>
                <p>Boolean indicating success or failure.</p>
                <h4>Example:</h4>
                <pre>
                patients = api.get_patients()
                if patients:
                    success = api.set_current_patient(patients[0])
                    if success:
                        print("Current patient set")
                    else:
                        print("Failed to set current patient")
                </pre>
            """,
            
            # Add more API documentation here as needed...
        }
        
        # Return documentation or default text
        return docs.get(api_name, f"<h3>{api_name}</h3><p>Documentation not available for this function.</p>")


class ScriptExecutionThread(QThread):
    """Thread for executing user scripts."""
    
    output_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    execution_finished = pyqtSignal(bool)
    
    def __init__(self, script_text):
        super(ScriptExecutionThread, self).__init__()
        self.script_text = script_text
        self.api = ScriptingAPI()
    
    def run(self):
        """Run the script in a separate thread."""
        # Create a string IO to redirect stdout and stderr
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_io = io.StringIO()
        stderr_io = io.StringIO()
        
        try:
            # Set up the global namespace for the script
            globals_dict = {
                'api': self.api,
                'print': self._custom_print
            }
            
            # Execute the script with redirected output
            with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
                exec(self.script_text, globals_dict)
            
            # Emit any remaining output
            output = stdout_io.getvalue()
            if output:
                self.output_ready.emit(output)
            
            errors = stderr_io.getvalue()
            if errors:
                self.error_occurred.emit(errors)
            
            self.execution_finished.emit(True)
            
        except Exception as e:
            import traceback
            error_text = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_text)
            self.execution_finished.emit(False)
    
    def _custom_print(self, *args, **kwargs):
        """Custom print function that emits the output signal."""
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture the output of the print statement
        stdout_io = io.StringIO()
        with redirect_stdout(stdout_io):
            print(*args, **kwargs)
        
        output = stdout_io.getvalue()
        self.output_ready.emit(output)


class ScriptingAPI:
    """
    Eclipse-like scripting API providing a clean interface for automating
    planning tasks through scripts.
    """
    
    def __init__(self):
        """Initialize the scripting API and register services."""
        self.patient_db = ServiceRegistry.get_service(PatientDB)
        self.current_patient = None
        self.current_plan = None
        self.dose_engine = DoseCalculationEngine()
        self.optimization_engine = OptimizationEngine()
        
    def get_patients(self) -> List[Dict[str, Any]]:
        """
        Get a list of all patients in the system.
        
        Returns:
            List of patient dictionaries containing id, name, and other metadata.
        """
        if not self.patient_db:
            return []
        
        return self.patient_db.get_all_patients()
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Patient]:
        """
        Get a patient by ID.
        
        Args:
            patient_id: The ID of the patient to retrieve
            
        Returns:
            Patient object or None if not found
        """
        if not self.patient_db:
            return None
        
        return self.patient_db.get_patient(patient_id)
    
    def set_current_patient(self, patient: Union[str, Patient]) -> bool:
        """
        Set the current patient for subsequent operations.
        
        Args:
            patient: Either a patient ID string or a Patient object
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(patient, str):
            self.current_patient = self.get_patient_by_id(patient)
        else:
            self.current_patient = patient
            
        return self.current_patient is not None
    
    def get_plans(self, patient: Optional[Union[str, Patient]] = None) -> List[Plan]:
        """
        Get all plans for a patient.
        
        Args:
            patient: Either a patient ID, Patient object, or None to use current_patient
            
        Returns:
            List of Plan objects
        """
        target_patient = self._resolve_patient(patient)
        if not target_patient:
            return []
        
        return target_patient.plans if hasattr(target_patient, 'plans') else []
    
    def set_current_plan(self, plan: Union[str, Plan]) -> bool:
        """
        Set the current plan for subsequent operations.
        
        Args:
            plan: Either a plan ID string or a Plan object
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_patient:
            return False
            
        if isinstance(plan, str):
            plans = self.get_plans()
            for p in plans:
                if p.id == plan:
                    self.current_plan = p
                    return True
            return False
        else:
            self.current_plan = plan
            return True
    
    def create_plan(self, name: str, description: str = "") -> Optional[Plan]:
        """
        Create a new treatment plan for the current patient.
        
        Args:
            name: Name of the new plan
            description: Optional description
            
        Returns:
            Newly created Plan object or None if failed
        """
        if not self.current_patient:
            return None
            
        # This would call into the plan creation logic
        # For now, just a placeholder
        logger.info(f"Creating plan {name} for patient {self.current_patient.id}")
        
        # Actual implementation would create the plan in the database
        # and return the new plan object
        return None  # Placeholder
    
    def get_structures(self, plan: Optional[Union[str, Plan]] = None) -> List[Structure]:
        """
        Get all structures for a plan.
        
        Args:
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            List of Structure objects
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return []
        
        return target_plan.structures if hasattr(target_plan, 'structures') else []
    
    def calculate_dvh(self, structure: Union[str, Structure], plan: Optional[Union[str, Plan]] = None) -> Dict[str, Any]:
        """
        Calculate the DVH for a specific structure.
        
        Args:
            structure: Either a structure name or Structure object
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            Dictionary containing DVH data
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return {}
            
        target_structure = self._resolve_structure(structure, target_plan)
        if not target_structure:
            return {}
            
        # This would call into DVH calculation logic
        # For now, just a placeholder
        logger.info(f"Calculating DVH for structure {target_structure.name} in plan {target_plan.name}")
        
        # Actual implementation would calculate the DVH and return the data
        return {}  # Placeholder
    
    def add_clinical_goal(self, structure: Union[str, Structure], 
                         goal_type: str, dose: float, volume: float = None,
                         plan: Optional[Union[str, Plan]] = None) -> bool:
        """
        Add a clinical goal to a plan.
        
        Args:
            structure: Either a structure name or Structure object
            goal_type: Type of goal (e.g., "min_dose", "max_dose", "volume_at_dose", "dose_at_volume")
            dose: Dose value in Gy
            volume: Volume percentage (0-100) for volume constraints
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            True if successful, False otherwise
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return False
            
        target_structure = self._resolve_structure(structure, target_plan)
        if not target_structure:
            return False
            
        # This would call into clinical goal creation logic
        # For now, just a placeholder
        logger.info(f"Adding {goal_type} goal for structure {target_structure.name} in plan {target_plan.name}")
        
        # Actual implementation would create the clinical goal
        return True  # Placeholder
    
    def optimize_plan(self, iterations: int = 100, plan: Optional[Union[str, Plan]] = None) -> bool:
        """
        Run optimization on a plan.
        
        Args:
            iterations: Number of iterations to run
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            True if successful, False otherwise
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return False
            
        # This would call into optimization logic
        # For now, just a placeholder
        logger.info(f"Optimizing plan {target_plan.name} for {iterations} iterations")
        
        # Actual implementation would run the optimization
        return True  # Placeholder
    
    def calculate_dose(self, algorithm: str = "collapsed_cone", 
                      resolution: float = 3.0,
                      plan: Optional[Union[str, Plan]] = None) -> bool:
        """
        Calculate dose for a plan.
        
        Args:
            algorithm: Dose calculation algorithm to use
            resolution: Calculation grid resolution in mm
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            True if successful, False otherwise
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return False
            
        # This would call into dose calculation logic
        # For now, just a placeholder
        logger.info(f"Calculating dose for plan {target_plan.name} using {algorithm} at {resolution}mm")
        
        # Actual implementation would calculate the dose
        return True  # Placeholder
    
    def export_plan(self, filename: str, format: str = "dicom",
                  plan: Optional[Union[str, Plan]] = None) -> bool:
        """
        Export a plan to a file.
        
        Args:
            filename: Path to export the plan to
            format: Export format (dicom, xml, etc.)
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            True if successful, False otherwise
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return False
            
        # This would call into plan export logic
        # For now, just a placeholder
        logger.info(f"Exporting plan {target_plan.name} to {filename} in {format} format")
        
        # Actual implementation would export the plan
        return True  # Placeholder
    
    def create_beam(self, name: str, gantry_angle: float, couch_angle: float = 0.0,
                   collimator_angle: float = 0.0, energy: str = "6X",
                   plan: Optional[Union[str, Plan]] = None) -> Optional[Beam]:
        """
        Create a new beam in a plan.
        
        Args:
            name: Name of the beam
            gantry_angle: Gantry angle in degrees
            couch_angle: Couch angle in degrees
            collimator_angle: Collimator angle in degrees
            energy: Beam energy (e.g., "6X", "10X", "6FFF")
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            Newly created Beam object or None if failed
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return None
            
        # This would call into beam creation logic
        # For now, just a placeholder
        logger.info(f"Creating beam {name} at gantry={gantry_angle}, couch={couch_angle} for plan {target_plan.name}")
        
        # Actual implementation would create the beam
        return None  # Placeholder
    
    def evaluate_plan(self, plan: Optional[Union[str, Plan]] = None) -> Dict[str, Any]:
        """
        Evaluate a plan against clinical goals.
        
        Args:
            plan: Either a plan ID, Plan object, or None to use current_plan
            
        Returns:
            Dictionary containing evaluation results
        """
        target_plan = self._resolve_plan(plan)
        if not target_plan:
            return {}
            
        # This would call into plan evaluation logic
        # For now, just a placeholder
        logger.info(f"Evaluating plan {target_plan.name}")
        
        # Actual implementation would evaluate the plan
        return {}  # Placeholder
    
    def get_api_documentation(self) -> Dict[str, Dict[str, str]]:
        """
        Get documentation for all available API functions.
        
        Returns:
            Dictionary mapping function names to their documentation
        """
        docs = {}
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if not name.startswith('_'):
                doc = inspect.getdoc(method) or "No documentation available"
                signature = str(inspect.signature(method))
                docs[name] = {
                    'doc': doc,
                    'signature': signature,
                    'function': method
                }
        return docs
    
    # Helper methods
    def _resolve_patient(self, patient: Optional[Union[str, Patient]]) -> Optional[Patient]:
        """Resolve a patient reference to a Patient object."""
        if patient is None:
            return self.current_patient
            
        if isinstance(patient, str):
            return self.get_patient_by_id(patient)
            
        return patient
    
    def _resolve_plan(self, plan: Optional[Union[str, Plan]]) -> Optional[Plan]:
        """Resolve a plan reference to a Plan object."""
        if plan is None:
            return self.current_plan
            
        if isinstance(plan, str):
            plans = self.get_plans()
            for p in plans:
                if p.id == plan:
                    return p
            return None
            
        return plan
    
    def _resolve_structure(self, structure: Union[str, Structure], plan: Plan) -> Optional[Structure]:
        """Resolve a structure reference to a Structure object."""
        if isinstance(structure, str):
            structures = plan.structures if hasattr(plan, 'structures') else []
            for s in structures:
                if s.name == structure:
                    return s
            return None
            
        return structure


class ScriptEditor(QMainWindow):
    """
    Eclipse-like script editor interface for writing and running planning scripts.
    """
    
    script_executed = pyqtSignal(str, bool)  # Script output, success
    
    def __init__(self, parent=None):
        """Initialize the script editor widget."""
        super().__init__(parent)
        self.parent = parent
        self.api = ScriptingAPI()
        self.setWindowTitle("QuangTPS Script Editor")
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the UI components."""
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        
        # Add toolbar actions
        self.new_action = QAction(QIcon("icons/new.png"), "New Script", self)
        self.new_action.triggered.connect(self.new_script)
        self.toolbar.addAction(self.new_action)
        
        self.open_action = QAction(QIcon("icons/open.png"), "Open Script", self)
        self.open_action.triggered.connect(self.open_script)
        self.toolbar.addAction(self.open_action)
        
        self.save_action = QAction(QIcon("icons/save.png"), "Save Script", self)
        self.save_action.triggered.connect(self.save_script)
        self.toolbar.addAction(self.save_action)
        
        self.toolbar.addSeparator()
        
        self.run_action = QAction(QIcon("icons/run.png"), "Run Script", self)
        self.run_action.triggered.connect(self.run_script)
        self.toolbar.addAction(self.run_action)
        
        self.stop_action = QAction(QIcon("icons/stop.png"), "Stop Execution", self)
        self.stop_action.triggered.connect(self.stop_execution)
        self.stop_action.setEnabled(False)
        self.toolbar.addAction(self.stop_action)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Editor panel
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        
        # Script editor
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Courier New", 10))
        self.highlighter = PythonSyntaxHighlighter(self.editor.document())
        editor_layout.addWidget(self.editor)
        
        # Output panel with tabs
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        
        self.output_tabs = QTabWidget()
        
        # Console output tab
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Courier New", 10))
        self.output_tabs.addTab(self.console_output, "Console")
        
        # API Documentation tab
        self.api_doc_widget = ApiDocWidget()
        self.output_tabs.addTab(self.api_doc_widget, "API Documentation")
        
        output_layout.addWidget(self.output_tabs)
        
        # Add widgets to splitter
        splitter.addWidget(editor_widget)
        splitter.addWidget(output_widget)
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        # Set up script execution thread
        self.execution_thread = None
    
    def new_script(self):
        """Create a new script."""
        if self.editor.document().isModified():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "There are unsaved changes. Do you want to save them?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_script()
            elif reply == QMessageBox.Cancel:
                return
        
        self.editor.clear()
        self.console_output.clear()
    
    def open_script(self):
        """Open a script from file."""
        if self.editor.document().isModified():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "There are unsaved changes. Do you want to save them?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_script()
            elif reply == QMessageBox.Cancel:
                return
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "", "Python Files (*.py)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as file:
                    self.editor.setPlainText(file.read())
                self.console_output.clear()
            except Exception as e:
                QMessageBox.critical(self, "Error Opening File", str(e))
    
    def save_script(self):
        """Save the script to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "", "Python Files (*.py)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as file:
                    file.write(self.editor.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", str(e))
    
    def run_script(self):
        """Run the current script."""
        if self.execution_thread and self.execution_thread.isRunning():
            QMessageBox.warning(
                self, "Script Running",
                "A script is already running. Please stop it first."
            )
            return
        
        script_text = self.editor.toPlainText()
        if not script_text.strip():
            QMessageBox.warning(
                self, "Empty Script",
                "The script is empty. Please enter some code."
            )
            return
        
        # Clear console output
        self.console_output.clear()
        self.console_output.append("Script execution started...\n")
        
        # Switch to console tab
        self.output_tabs.setCurrentIndex(0)
        
        # Disable run button and enable stop button
        self.run_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        
        # Create and start execution thread
        self.execution_thread = ScriptExecutionThread(script_text)
        self.execution_thread.output_ready.connect(self.append_output)
        self.execution_thread.error_occurred.connect(self.append_error)
        self.execution_thread.execution_finished.connect(self.on_execution_finished)
        self.execution_thread.start()
    
    def stop_execution(self):
        """Stop the current script execution."""
        if self.execution_thread and self.execution_thread.isRunning():
            # In Python, there's no clean way to stop a thread from outside
            # We'll just terminate it forcefully
            self.execution_thread.terminate()
            self.on_execution_finished(False)
            self.console_output.append("\nScript execution terminated by user.")
    
    def append_output(self, text):
        """Append text to the console output."""
        self.console_output.append(text)
    
    def append_error(self, text):
        """Append error text to the console output."""
        # Format error text in red
        self.console_output.append(f"<span style='color:red'>{text}</span>")
    
    def on_execution_finished(self, success):
        """Handle script execution finished."""
        # Enable run button and disable stop button
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        
        if success:
            self.console_output.append("\nScript execution completed successfully.")
        else:
            self.console_output.append("\nScript execution failed.")


# Note: Missing import for io module, add this at the top
import io 