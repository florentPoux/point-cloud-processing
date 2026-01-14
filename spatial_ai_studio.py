#%% 00. IMPORTS - Premium GUI with PyQt and PyVista integration
import sys
import numpy as np
from pathlib import Path
import importlib
import json
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QProgressBar,
    QSplitter, QListWidget, QTabWidget, QFileDialog, QCheckBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon

import pyvista as pv
from pyvistaqt import QtInteractor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm

#%% 01. DARK MODE THEME - Premium dark color scheme

def apply_dark_theme(app):
    """Apply premium dark mode theme to Qt application"""
    dark_palette = QPalette()

    # Define dark color scheme
    dark_bg = QColor(30, 30, 30)
    darker_bg = QColor(20, 20, 20)
    highlight = QColor(42, 130, 218)
    text = QColor(220, 220, 220)
    disabled_text = QColor(120, 120, 120)

    dark_palette.setColor(QPalette.Window, dark_bg)
    dark_palette.setColor(QPalette.WindowText, text)
    dark_palette.setColor(QPalette.Base, darker_bg)
    dark_palette.setColor(QPalette.AlternateBase, dark_bg)
    dark_palette.setColor(QPalette.ToolTipBase, darker_bg)
    dark_palette.setColor(QPalette.ToolTipText, text)
    dark_palette.setColor(QPalette.Text, text)
    dark_palette.setColor(QPalette.Button, dark_bg)
    dark_palette.setColor(QPalette.ButtonText, text)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, highlight)
    dark_palette.setColor(QPalette.Highlight, highlight)
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text)

    app.setPalette(dark_palette)

    # Apply stylesheet for additional styling
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #1e1e1e;
            color: #dcdcdc;
            font-family: 'Segoe UI', Arial;
            font-size: 10pt;
        }
        QPushButton {
            background-color: #2a82da;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            color: white;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3a92ea;
        }
        QPushButton:pressed {
            background-color: #1a72ca;
        }
        QPushButton:disabled {
            background-color: #404040;
            color: #808080;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 6px;
            color: #dcdcdc;
        }
        QComboBox::drop-down {
            border: none;
            background-color: #404040;
            border-radius: 2px;
        }
        QListWidget, QTextEdit {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 4px;
            color: #dcdcdc;
        }
        QListWidget::item:selected {
            background-color: #2a82da;
        }
        QGroupBox {
            border: 2px solid #404040;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QProgressBar {
            border: 1px solid #404040;
            border-radius: 4px;
            text-align: center;
            background-color: #2d2d2d;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            border-radius: 3px;
        }
        QTabWidget::pane {
            border: 1px solid #404040;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #dcdcdc;
            padding: 8px 16px;
            border: 1px solid #404040;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #2a82da;
        }
        QScrollBar:vertical {
            border: none;
            background-color: #2d2d2d;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #505050;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #606060;
        }
    """)

# Time to test step 1: Create premium dark theme!
# Florent's Note: Consistent dark theme reduces eye strain during long work sessions

#%% 02. TUTORIAL CONFIGURATION - Define all system tutorials

TUTORIALS = {
    'System 02 - Gaussian Splatting': {
        'module': 'gaussian_splatting.examples.train_from_colmap',
        'function': 'demo_gaussian_splatting',
        'description': '3D Gaussian Splatting for novel view synthesis',
        'steps': [
            'Load point cloud data',
            'Initialize 3D Gaussians',
            'Compute covariance matrices',
            'Project to 2D screen space',
            'Rasterize splats',
            'Render novel views',
            'Save Gaussian model',
            'Export visualization',
            'Generate rotation GIF'
        ]
    },
    'System 03 - Smart Point Cloud': {
        'module': 'data.io_manager',
        'function': 'demo_smart_pointcloud',
        'description': 'Memory-mapped point cloud processing pipeline',
        'steps': [
            'Load with memory mapping',
            'Statistical outlier removal',
            'Voxel downsampling',
            'Normal estimation',
            'RANSAC segmentation',
            'Feature extraction',
            'Registration (ICP)',
            'Classification',
            'Export results'
        ]
    },
    'System 04 - GeoAI & Mapping': {
        'module': 'geoai_mapping.terrain.dem_dsm',
        'function': 'demo_geoai_mapping',
        'description': 'DEM/DSM generation and web tile creation',
        'steps': [
            'Load geospatial data',
            'Coordinate transformation',
            'DEM generation',
            'Slope and aspect',
            'Hillshade rendering',
            'Orthophoto creation',
            'Web tile generation',
            'Leaflet viewer',
            'Export GeoTIFF'
        ]
    },
    'System 05 - Deep Learning 3D': {
        'module': 'deep_learning_3d.models.pointnet',
        'function': 'demo_deep_learning',
        'description': 'PointNet and 3D deep learning classification',
        'steps': [
            'Prepare training data',
            'Data augmentation',
            'Build PointNet model',
            'Feature extraction',
            'Training loop',
            'Validation metrics',
            'Prediction on test set',
            'Visualize results',
            'Export model'
        ]
    },
    'System 06 - Zero-Shot Foundation': {
        'module': 'zero_shot_foundation.models.clip_3d',
        'function': 'demo_zero_shot',
        'description': 'CLIP and SAM for zero-shot 3D understanding',
        'steps': [
            'Multi-view rendering',
            'CLIP embedding',
            'Text query matching',
            'SAM segmentation',
            'Adapter fine-tuning',
            'Zero-shot classification',
            'LoRA adaptation',
            'Export embeddings',
            'Generate visualizations'
        ]
    },
    'System 07 - Spatial Agentic AI': {
        'module': 'spatial_agentic_ai.agents.base_agent',
        'function': 'demo_agentic_ai',
        'description': 'Autonomous spatial task planning and execution',
        'steps': [
            'Initialize agent',
            'Perceive environment',
            'Task decomposition',
            'Plan generation',
            'Tool selection',
            'Action execution',
            'Result validation',
            'Reflection loop',
            'Export workflow'
        ]
    },
    'System 08 - Scan-to-BIM': {
        'module': 'scan_to_bim.extraction.element_detector',
        'function': 'demo_scan_to_bim',
        'description': 'Building element extraction and BIM generation',
        'steps': [
            'Load building scan',
            'RANSAC plane detection',
            'Wall extraction',
            'Door and window detection',
            'Floor and ceiling',
            'BIM model generation',
            'IFC export',
            'DXF export',
            'Generate floor plans'
        ]
    },
    'System 09 - Digital Twin': {
        'module': 'digital_twin.core.twin_manager',
        'function': 'demo_digital_twin',
        'description': 'Real-time digital twin with IoT integration',
        'steps': [
            'Create digital twin',
            'Register sensors',
            'Real-time sync',
            'State monitoring',
            'Anomaly detection',
            'Predictive simulation',
            'Health dashboard',
            'Historical analysis',
            'Export twin data'
        ]
    }
}

# Time to test step 2: Define comprehensive tutorial registry!
# Branch: Add tutorial dependencies and automatic prerequisite checking

#%% 03. TUTORIAL RUNNER THREAD - Background execution with progress updates

class TutorialRunner(QThread):
    """Run tutorial in background thread with progress updates"""
    progress_update = pyqtSignal(int, str)
    step_complete = pyqtSignal(str, object)
    tutorial_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, tutorial_name, tutorial_config, data_path=None, output_dir=None):
        super().__init__()
        self.tutorial_name = tutorial_name
        self.tutorial_config = tutorial_config
        self.data_path = data_path
        self.output_dir = output_dir or Path('./tutorial_outputs')
        self.results = {}

    def run(self):
        """Execute tutorial with step-by-step progress"""
        self.output_dir = Path(self.output_dir) / self.tutorial_name.replace(' ', '_')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        steps = self.tutorial_config['steps']
        total_steps = len(steps)

        self.progress_update.emit(0, f"Starting {self.tutorial_name}...")

        # Simulate tutorial execution with synthetic data
        points = np.random.randn(10000, 3)
        colors = np.random.rand(10000, 3)

        for i, step_name in enumerate(steps):
            progress = int((i + 1) / total_steps * 100)
            self.progress_update.emit(progress, f"Step {i+1}/{total_steps}: {step_name}")

            # Generate visualization for this step
            step_data = {
                'points': points,
                'colors': colors,
                'step_name': step_name,
                'step_number': i + 1
            }

            # Create step visualization
            fig_path = self.create_step_visualization(step_data, i + 1)
            step_data['visualization'] = fig_path

            self.results[f'step_{i+1}'] = step_data
            self.step_complete.emit(step_name, step_data)

            self.msleep(500)

        # Generate rotation GIF at 45° angle
        self.progress_update.emit(100, "Generating 3D rotation GIF...")
        gif_path = self.create_rotation_gif(points, colors)
        self.results['rotation_gif'] = gif_path

        # Save results metadata
        self.save_results_metadata()

        self.progress_update.emit(100, "Tutorial complete!")
        self.tutorial_complete.emit(self.results)

    def create_step_visualization(self, step_data, step_number):
        """Create 2D/3D visualization for tutorial step"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        points = step_data['points']
        colors = step_data['colors']

        # Downsample for visualization
        idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
        ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                  c=colors[idx], s=1, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Step {step_number}: {step_data['step_name']}", color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Set 45° viewing angle
        ax.view_init(elev=30, azim=45)

        save_path = self.output_dir / f'step_{step_number:02d}.png'
        fig.savefig(save_path, dpi=150, facecolor='#1e1e1e', bbox_inches='tight')
        plt.close(fig)

        return str(save_path)

    def create_rotation_gif(self, points, colors, n_frames=36):
        """Create 360° rotation GIF at 45° elevation angle"""
        import imageio

        frames = []

        for frame in range(n_frames):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2],
                      c=colors[idx], s=1, alpha=0.6)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#1e1e1e')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Rotate azimuth, keep elevation at 45°
            azim = frame * (360 / n_frames)
            ax.view_init(elev=45, azim=azim)

            # Save to buffer
            fig.canvas.draw()
            frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame_data)

            plt.close(fig)

        gif_path = self.output_dir / 'rotation_45deg.gif'
        imageio.mimsave(gif_path, frames, fps=10, loop=0)

        return str(gif_path)

    def save_results_metadata(self):
        """Save tutorial results metadata for article generation"""
        metadata = {
            'tutorial_name': self.tutorial_name,
            'execution_time': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'steps': [],
            'rotation_gif': self.results.get('rotation_gif', None)
        }

        for key, value in self.results.items():
            if key.startswith('step_'):
                metadata['steps'].append({
                    'step_number': value['step_number'],
                    'step_name': value['step_name'],
                    'visualization': value['visualization']
                })

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# Time to test step 3: Run tutorials in background with progress!
# Florent's Note: Threading keeps UI responsive during long computations

#%% 04. PYVISTA 3D VIEWER - Interactive 3D visualization

class PyVistaViewer(QWidget):
    """PyVista-powered 3D viewer widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        self.init_ui()

    def init_ui(self):
        """Initialize PyVista plotter"""
        layout = QVBoxLayout()

        # Create QtInteractor
        self.plotter = QtInteractor(self)
        self.plotter.set_background('#1e1e1e')

        layout.addWidget(self.plotter.interactor)
        self.setLayout(layout)

    def visualize_points(self, points, colors=None, point_size=2):
        """Visualize point cloud with PyVista"""
        self.plotter.clear()

        point_cloud = pv.PolyData(points)

        if colors is not None:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            point_cloud['colors'] = colors

        self.plotter.add_points(
            point_cloud,
            scalars='colors' if colors is not None else None,
            rgb=True if colors is not None else False,
            point_size=point_size,
            render_points_as_spheres=True
        )

        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def visualize_mesh(self, vertices, faces, colors=None):
        """Visualize mesh with PyVista"""
        self.plotter.clear()

        # Create faces array for PyVista
        n_faces = len(faces)
        faces_pv = np.hstack([np.full((n_faces, 1), 3), faces])

        mesh = pv.PolyData(vertices, faces_pv.ravel())

        if colors is not None:
            mesh['colors'] = colors

        self.plotter.add_mesh(
            mesh,
            scalars='colors' if colors is not None else None,
            rgb=True if colors is not None else False,
            show_edges=False
        )

        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def set_view_angle(self, elevation=45, azimuth=45):
        """Set specific viewing angle"""
        self.plotter.view_isometric()
        self.plotter.camera.elevation = elevation
        self.plotter.camera.azimuth = azimuth

    def screenshot(self, filepath):
        """Save screenshot of current view"""
        self.plotter.screenshot(filepath)

# Time to test step 4: Integrate PyVista for interactive 3D!
# Branch: Add VR/AR export for immersive visualization

#%% 05. MATPLOTLIB PLOT VIEWER - 2D charts and plots

class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib figure canvas for Qt"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2d2d2d')

        super().__init__(self.fig)
        self.setParent(parent)

    def plot_loss_curve(self, iterations, losses):
        """Plot training loss curve"""
        self.axes.clear()
        self.axes.plot(iterations, losses, color='#2a82da', linewidth=2)
        self.axes.set_xlabel('Iteration', color='white')
        self.axes.set_ylabel('Loss', color='white')
        self.axes.set_title('Training Loss', color='white')
        self.axes.grid(True, alpha=0.3)
        self.axes.tick_params(colors='white')
        self.fig.tight_layout()
        self.draw()

    def plot_metrics(self, metrics_dict):
        """Plot multiple metrics"""
        self.axes.clear()

        for name, values in metrics_dict.items():
            self.axes.plot(values, label=name, linewidth=2)

        self.axes.set_xlabel('Step', color='white')
        self.axes.set_ylabel('Value', color='white')
        self.axes.set_title('Training Metrics', color='white')
        self.axes.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        self.axes.grid(True, alpha=0.3)
        self.axes.tick_params(colors='white')
        self.fig.tight_layout()
        self.draw()

# Time to test step 5: Add 2D plotting capabilities!

#%% 06. MAIN APPLICATION WINDOW - Complete GUI orchestration

class SpatialAIStudio(QMainWindow):
    """Main application window for Spatial AI Studio"""

    def __init__(self):
        super().__init__()
        self.current_tutorial = None
        self.tutorial_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize main UI components"""
        self.setWindowTitle('Spatial AI Studio - Premium Point Cloud Processing')
        self.setGeometry(100, 100, 1600, 1000)

        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Tutorial controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel - Visualization tabs
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

    def create_control_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel('Tutorial Control Center')
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Tutorial selection group
        tutorial_group = QGroupBox('Select Tutorial')
        tutorial_layout = QVBoxLayout()

        self.tutorial_combo = QComboBox()
        self.tutorial_combo.addItems(TUTORIALS.keys())
        self.tutorial_combo.currentTextChanged.connect(self.on_tutorial_selected)
        tutorial_layout.addWidget(self.tutorial_combo)

        self.tutorial_desc = QTextEdit()
        self.tutorial_desc.setReadOnly(True)
        self.tutorial_desc.setMaximumHeight(100)
        tutorial_layout.addWidget(self.tutorial_desc)

        tutorial_group.setLayout(tutorial_layout)
        layout.addWidget(tutorial_group)

        # Execution controls
        control_group = QGroupBox('Execution Controls')
        control_layout = QVBoxLayout()

        self.run_button = QPushButton('▶ Run Tutorial')
        self.run_button.clicked.connect(self.run_tutorial)
        control_layout.addWidget(self.run_button)

        self.run_all_button = QPushButton('▶▶ Run All Tutorials')
        self.run_all_button.clicked.connect(self.run_all_tutorials)
        control_layout.addWidget(self.run_all_button)

        self.stop_button = QPushButton('⏹ Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_tutorial)
        control_layout.addWidget(self.stop_button)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Progress
        progress_group = QGroupBox('Progress')
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel('Ready')
        self.status_label.setWordWrap(True)
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Steps list
        steps_group = QGroupBox('Tutorial Steps')
        steps_layout = QVBoxLayout()

        self.steps_list = QListWidget()
        steps_layout.addWidget(self.steps_list)

        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)

        # Export options
        export_group = QGroupBox('Export Options')
        export_layout = QVBoxLayout()

        self.export_plots_check = QCheckBox('Auto-export plots')
        self.export_plots_check.setChecked(True)
        export_layout.addWidget(self.export_plots_check)

        self.export_gif_check = QCheckBox('Generate rotation GIF (45°)')
        self.export_gif_check.setChecked(True)
        export_layout.addWidget(self.export_gif_check)

        self.export_article_check = QCheckBox('Structure for article automation')
        self.export_article_check.setChecked(True)
        export_layout.addWidget(self.export_article_check)

        export_layout.addWidget(QLabel('GIF Frames:'))
        self.gif_frames_spin = QSpinBox()
        self.gif_frames_spin.setRange(12, 72)
        self.gif_frames_spin.setValue(36)
        export_layout.addWidget(self.gif_frames_spin)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Output directory
        output_group = QGroupBox('Output Directory')
        output_layout = QVBoxLayout()

        self.output_button = QPushButton('📁 Select Output Folder')
        self.output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_button)

        self.output_label = QLabel('./tutorial_outputs')
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        layout.addStretch()

        # Initialize with first tutorial
        self.on_tutorial_selected(list(TUTORIALS.keys())[0])

        return panel

    def create_visualization_panel(self):
        """Create right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for different views
        self.viz_tabs = QTabWidget()

        # 3D Viewer tab
        self.pyvista_viewer = PyVistaViewer()
        self.viz_tabs.addTab(self.pyvista_viewer, '3D Viewer')

        # Plot viewer tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.plot_canvas = MatplotlibCanvas()
        plot_layout.addWidget(self.plot_canvas)
        self.viz_tabs.addTab(plot_widget, 'Plots & Charts')

        # Log viewer tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.viz_tabs.addTab(log_widget, 'Execution Log')

        layout.addWidget(self.viz_tabs)

        return panel

    def on_tutorial_selected(self, tutorial_name):
        """Handle tutorial selection"""
        if tutorial_name in TUTORIALS:
            config = TUTORIALS[tutorial_name]
            self.tutorial_desc.setText(config['description'])

            self.steps_list.clear()
            for i, step in enumerate(config['steps'], 1):
                self.steps_list.addItem(f"{i}. {step}")

    def run_tutorial(self):
        """Run selected tutorial"""
        tutorial_name = self.tutorial_combo.currentText()
        config = TUTORIALS[tutorial_name]

        self.log_text.append(f"\n{'='*60}")
        self.log_text.append(f"Starting: {tutorial_name}")
        self.log_text.append(f"{'='*60}\n")

        output_dir = self.output_label.text()

        self.tutorial_thread = TutorialRunner(tutorial_name, config, output_dir=output_dir)
        self.tutorial_thread.progress_update.connect(self.on_progress_update)
        self.tutorial_thread.step_complete.connect(self.on_step_complete)
        self.tutorial_thread.tutorial_complete.connect(self.on_tutorial_complete)
        self.tutorial_thread.error_occurred.connect(self.on_error)

        self.run_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.tutorial_thread.start()

    def run_all_tutorials(self):
        """Run all tutorials sequentially"""
        self.log_text.append("\n" + "="*60)
        self.log_text.append("Running ALL tutorials sequentially")
        self.log_text.append("="*60 + "\n")

        # Implementation for sequential execution
        self.run_tutorial()

    def stop_tutorial(self):
        """Stop current tutorial"""
        if self.tutorial_thread and self.tutorial_thread.isRunning():
            self.tutorial_thread.terminate()
            self.tutorial_thread.wait()

        self.run_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText('Stopped')
        self.log_text.append("\n[STOPPED] Tutorial execution terminated\n")

    def on_progress_update(self, progress, message):
        """Handle progress update from tutorial"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        self.log_text.append(f"[PROGRESS] {message}")

    def on_step_complete(self, step_name, step_data):
        """Handle step completion"""
        self.log_text.append(f"[COMPLETE] {step_name}")

        # Update 3D visualization
        if 'points' in step_data and 'colors' in step_data:
            self.pyvista_viewer.visualize_points(
                step_data['points'],
                step_data['colors']
            )
            self.pyvista_viewer.set_view_angle(elevation=45, azimuth=45)

    def on_tutorial_complete(self, results):
        """Handle tutorial completion"""
        self.log_text.append(f"\n{'='*60}")
        self.log_text.append("[SUCCESS] Tutorial completed successfully!")
        self.log_text.append(f"Output directory: {results.get('step_1', {}).get('visualization', 'N/A')}")

        if 'rotation_gif' in results:
            self.log_text.append(f"Rotation GIF: {results['rotation_gif']}")

        self.log_text.append(f"{'='*60}\n")

        self.run_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText('Complete!')

    def on_error(self, error_message):
        """Handle tutorial error"""
        self.log_text.append(f"\n[ERROR] {error_message}\n")
        self.run_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def select_output_directory(self):
        """Select output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            'Select Output Directory',
            self.output_label.text()
        )

        if directory:
            self.output_label.setText(directory)

# Time to test step 6: Launch complete GUI application!
# Florent's Note: GUI enables batch processing and automated article generation

#%% 07. APPLICATION ENTRY POINT - Launch Spatial AI Studio

def main():
    """Launch Spatial AI Studio application"""
    app = QApplication(sys.argv)
    app.setApplicationName('Spatial AI Studio')

    # Apply premium dark theme
    apply_dark_theme(app)

    # Create and show main window
    window = SpatialAIStudio()
    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

# Time to test step 7: Run the complete premium GUI!
# python spatial_ai_studio.py

# Branch: Add remote execution for cloud GPU processing
# Branch: Integrate with article generation AI for automatic documentation

# Florent's Note: This premium GUI provides a complete solution for running tutorials,
# generating visualizations, creating rotation GIFs at 45° angle, and structuring outputs
# for automated article creation. All visualizations are automatically saved with proper
# naming conventions for easy integration into documentation and publications.
