"""
Raccoon Painter - Enhanced painting application (JSON Direct Support)
Now works directly with data.json files
"""

import sys
import os

# Fix Qt conflicts if they exist
os.environ.pop('QT_PLUGIN_PATH', None)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QColorDialog, QMessageBox,
    QRadioButton, QButtonGroup, QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QBrush, QCursor
from PIL import Image
import numpy as np
import random
import tempfile

# Import animation window
try:
    from animation_system import AnimationWindow
except ImportError:
    print("Warning: animation_system.py not found")
    print("Animation features will be disabled.")
    AnimationWindow = None

# Constants
DEFAULT_CANVAS_WIDTH = 800
DEFAULT_CANVAS_HEIGHT = 600
DEFAULT_BRUSH_SIZE = 15
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 50
MAX_HISTORY_SIZE = 50
CONTOUR_THRESHOLD = 50
DISPLAY_SCALE_FACTOR = 0.95

# Color palette
QUICK_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#FFC0CB", "#A52A2A", "#808080", "#FFFFFF"
]


class PaintCanvas(QWidget):
    """Canvas for drawing with contour protection and resize preservation"""

    canvas_changed = pyqtSignal()

    def __init__(self, width=DEFAULT_CANVAS_WIDTH, height=DEFAULT_CANVAS_HEIGHT):
        super().__init__()
        self._initialize_images(width, height)
        self._initialize_drawing_state()
        self._initialize_ui()
        self.save_to_history()

    def _initialize_images(self, width, height):
        """Initialize image-related attributes"""
        self.original_image = None
        self.base_image_original = None
        self.base_image_path = None

        self.display_image = QImage(width, height, QImage.Format_ARGB32)
        self.display_image.fill(Qt.white)

    def _initialize_drawing_state(self):
        """Initialize drawing state attributes"""
        # History
        self.history = []
        self.history_index = -1
        self.max_history = MAX_HISTORY_SIZE

        # Tools
        self.current_tool = "brush"
        self.brush_color = QColor(0, 0, 0)
        self.brush_size = DEFAULT_BRUSH_SIZE
        self.glitter_mode = False

        # Contour protection
        self.protect_contours = True
        self.contour_threshold = CONTOUR_THRESHOLD

        # Drawing state
        self.drawing = False
        self.last_point = QPoint()

    def _initialize_ui(self):
        """Initialize UI settings"""
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def set_original_size(self, width, height):
        """Sets the size of the original image"""
        self.original_image = QImage(width, height, QImage.Format_ARGB32)
        self.original_image.fill(Qt.white)
        self.update_display_image()

    def update_display_image(self):
        """Updates the display image from the original"""
        if self.original_image is None:
            return

        canvas_w = self.width()
        canvas_h = self.height()

        # Clear display
        self.display_image = QImage(canvas_w, canvas_h, QImage.Format_ARGB32)
        self.display_image.fill(Qt.white)

        # Scale original to canvas
        scale = self._calculate_display_scale()
        scaled_image = self._create_scaled_image(scale)

        # Center and draw image
        self._draw_centered_image(scaled_image)
        self.update()

    def _calculate_display_scale(self):
        """Calculate the scale factor for display"""
        canvas_w = self.width()
        canvas_h = self.height()
        orig_w = self.original_image.width()
        orig_h = self.original_image.height()

        # Prevent division by zero
        if orig_w == 0 or orig_h == 0:
            return 1.0

        return min(canvas_w / orig_w, canvas_h / orig_h) * DISPLAY_SCALE_FACTOR

    def _create_scaled_image(self, scale):
        """Create a scaled version of the original image"""
        orig_w = self.original_image.width()
        orig_h = self.original_image.height()

        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        return self.original_image.scaled(
            new_w, new_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

    def _draw_centered_image(self, scaled_image):
        """Draw the scaled image centered on the display"""
        canvas_w = self.width()
        canvas_h = self.height()
        img_w = scaled_image.width()
        img_h = scaled_image.height()

        x = (canvas_w - img_w) // 2
        y = (canvas_h - img_h) // 2

        painter = QPainter(self.display_image)
        painter.drawImage(x, y, scaled_image)
        # painter.end() called automatically when painter goes out of scope

    def paintEvent(self, event):
        """Render canvas"""
        if self.display_image.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.drawImage(0, 0, self.display_image)
        # painter.end() is called automatically when painter goes out of scope

    def map_to_original_coords(self, screen_point):
        """Transform screen coordinates to original image coordinates"""
        if self.original_image is None:
            return None

        canvas_w = self.width()
        canvas_h = self.height()
        orig_w = self.original_image.width()
        orig_h = self.original_image.height()

        # Calculate display scale and position
        scale = self._calculate_display_scale()
        display_w = max(1, int(orig_w * scale))
        display_h = max(1, int(orig_h * scale))
        display_x = (canvas_w - display_w) // 2
        display_y = (canvas_h - display_h) // 2

        # Coordinates relative to image
        rel_x = screen_point.x() - display_x
        rel_y = screen_point.y() - display_y

        # Check if inside image
        if not self._is_point_in_bounds(rel_x, rel_y, display_w, display_h):
            return None

        # Scale back to original coordinates
        scale_back = orig_w / display_w if display_w > 0 else 1.0
        orig_x = int(rel_x * scale_back)
        orig_y = int(rel_y * scale_back)

        # Ensure within boundaries
        orig_x = max(0, min(orig_x, orig_w - 1))
        orig_y = max(0, min(orig_y, orig_h - 1))

        return QPoint(orig_x, orig_y)

    def _is_point_in_bounds(self, x, y, width, height):
        """Check if a point is within bounds"""
        return 0 <= x < width and 0 <= y < height

    def resizeEvent(self, event):
        """Handle window resize - preserves coloring"""
        self.update_display_image()
        super().resizeEvent(event)

    def load_base_image(self, image_path):
        """Load base image (contours)"""
        self.base_image_path = image_path

        # Load original
        self.base_image_original = Image.open(image_path).convert("RGBA")

        # Set original image size
        orig_w, orig_h = self.base_image_original.size
        self.set_original_size(orig_w, orig_h)

        # Apply image to original
        self._apply_base_to_original()

        # Update display
        self.update_display_image()
        self.save_to_history()

    def _apply_base_to_original(self):
        """Apply base image to original image"""
        if self.base_image_original is None or self.original_image is None:
            return

        orig_w, orig_h = self.base_image_original.size
        pil_image = self.base_image_original.copy()
        data = pil_image.tobytes("raw", "RGBA")

        # QImage requires bytes per line to be a multiple of 4
        bytes_per_line = orig_w * 4
        base_qimage = QImage(data, orig_w, orig_h, bytes_per_line, QImage.Format_ARGB32)

        painter = QPainter(self.original_image)
        painter.drawImage(0, 0, base_qimage)
        # painter.end() called automatically when painter goes out of scope

    def mousePressEvent(self, event):
        """Handle mouse press event"""
        if event.button() == Qt.LeftButton:
            orig_point = self.map_to_original_coords(event.pos())
            if orig_point:
                if self.current_tool == "fill":
                    # Fill tool - single click fill
                    self._fill_at_point(orig_point)
                    self.update_display_image()
                    self.save_to_history()
                else:
                    # Other tools - start drawing
                    self.drawing = True
                    self.last_point = orig_point
                    self._apply_tool(orig_point)

    def mouseMoveEvent(self, event):
        """Handle mouse move event"""
        if self.drawing and event.buttons() & Qt.LeftButton:
            orig_point = self.map_to_original_coords(event.pos())
            if orig_point:
                self._draw_line(self.last_point, orig_point)
                self.last_point = orig_point

    def mouseReleaseEvent(self, event):
        """Handle mouse release event"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.save_to_history()

    def _apply_tool(self, point):
        """Apply current tool at given point"""
        if self.current_tool == "brush" or self.glitter_mode:
            self._paint_at_point(point)
        elif self.current_tool == "eraser":
            self._erase_at_point(point)
        self.update_display_image()

    def _paint_at_point(self, point):
        """Paint at a specific point"""
        if not self._should_protect_contour(point):
            painter = QPainter(self.original_image)
            painter.setRenderHint(QPainter.Antialiasing)

            if self.glitter_mode:
                self._apply_glitter_effect(painter, point)
            else:
                self._apply_brush_stroke(painter, point)
            # painter.end() called automatically

    def _apply_glitter_effect(self, painter, center):
        """Apply glitter effect at a point"""
        num_particles = 15

        for _ in range(num_particles):
            offset_x = random.randint(-self.brush_size, self.brush_size)
            offset_y = random.randint(-self.brush_size, self.brush_size)
            size = random.randint(1, 3)

            particle_color = QColor(self.brush_color)
            particle_color.setAlpha(random.randint(100, 255))

            painter.setPen(QPen(particle_color, size, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoint(center.x() + offset_x, center.y() + offset_y)

    def _apply_brush_stroke(self, painter, point):
        """Apply normal brush stroke at a point"""
        painter.setPen(QPen(self.brush_color, self.brush_size,
                           Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(point)

    def _erase_at_point(self, point):
        """Erase at a specific point"""
        if not self._should_protect_contour(point):
            painter = QPainter(self.original_image)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(QPen(Qt.transparent, self.brush_size,
                              Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPoint(point)
            # painter.end() called automatically

    def _draw_line(self, start_point, end_point):
        """Draw a line between two points"""
        if self.current_tool == "brush" or self.glitter_mode:
            if not self._should_protect_contour(end_point):
                painter = QPainter(self.original_image)
                painter.setRenderHint(QPainter.Antialiasing)

                if self.glitter_mode:
                    # Calculate distance for glitter spacing
                    distance = self._distance(start_point, end_point)
                    steps = max(1, int(distance / 5))
                    for i in range(steps + 1):
                        t = i / steps if steps > 0 else 0
                        x = int(start_point.x() + t * (end_point.x() - start_point.x()))
                        y = int(start_point.y() + t * (end_point.y() - start_point.y()))
                        self._apply_glitter_effect(painter, QPoint(x, y))
                else:
                    painter.setPen(QPen(self.brush_color, self.brush_size,
                                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.drawLine(start_point, end_point)
                # painter.end() called automatically

        elif self.current_tool == "eraser":
            if not self._should_protect_contour(end_point):
                painter = QPainter(self.original_image)
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.setPen(QPen(Qt.transparent, self.brush_size,
                                  Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(start_point, end_point)
                # painter.end() called automatically

        self.update_display_image()

    def _distance(self, p1, p2):
        """Calculate distance between two points"""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        return (dx * dx + dy * dy) ** 0.5

    def _should_protect_contour(self, point):
        """Check if contour should be protected at point"""
        if not self.protect_contours or self.base_image_original is None:
            return False

        x, y = point.x(), point.y()

        # Check bounds
        if x < 0 or y < 0 or x >= self.base_image_original.width or y >= self.base_image_original.height:
            return False

        # Get pixel from base image
        try:
            pixel = self.base_image_original.getpixel((x, y))

            # Check if it's a dark contour (black or dark gray line)
            if isinstance(pixel, tuple) and len(pixel) >= 3:
                r, g, b = pixel[:3]
                brightness = (r + g + b) / 3
                return brightness < self.contour_threshold
        except:
            return False

        return False

    def _fill_at_point(self, point):
        """Fill enclosed area at point using optimized flood fill"""
        if self.original_image is None:
            return

        x, y = point.x(), point.y()
        width = self.original_image.width()
        height = self.original_image.height()

        # Check bounds
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        # Check if clicking on protected contour
        if self._should_protect_contour(point):
            return

        # Get target color at click point
        target_color = self.original_image.pixelColor(x, y)
        fill_color = self.brush_color

        # Don't fill if same color (within tolerance)
        if self._colors_similar(target_color, fill_color, 5):
            return

        # Convert QImage to bytes for faster access
        ptr = self.original_image.bits()
        ptr.setsize(self.original_image.byteCount())

        # Create numpy array view (zero-copy)
        import numpy as np
        img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        # Target and fill colors as numpy arrays
        target_rgb = np.array([target_color.blue(), target_color.green(), target_color.red()], dtype=np.uint8)
        fill_rgb = np.array([fill_color.blue(), fill_color.green(), fill_color.red(), 255], dtype=np.uint8)

        # Tolerance for matching
        tolerance = 30

        # Flood fill using stack
        stack = [(x, y)]
        filled = np.zeros((height, width), dtype=bool)

        pixels_filled = 0
        max_pixels = 500000

        while stack and pixels_filled < max_pixels:
            cx, cy = stack.pop()

            # Skip if out of bounds or already filled
            if cx < 0 or cy < 0 or cx >= width or cy >= height:
                continue
            if filled[cy, cx]:
                continue

            # Check contour protection
            if self._should_protect_contour(QPoint(cx, cy)):
                continue

            # Get current pixel color (BGR format in Qt)
            current_rgb = img_array[cy, cx, :3]

            # Check if color matches target (with tolerance)
            color_diff = np.abs(current_rgb.astype(np.int16) - target_rgb.astype(np.int16))
            if np.any(color_diff > tolerance):
                continue

            # Fill this pixel
            img_array[cy, cx] = fill_rgb
            filled[cy, cx] = True
            pixels_filled += 1

            # Add neighbors to stack (4-way connectivity)
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))

        # The array is modified in-place, so the QImage is already updated
        # Just make sure to mark it as changed
        self.original_image = self.original_image.copy()

    def _colors_similar(self, color1, color2, tolerance=10):
        """Check if two colors are similar within tolerance"""
        r_diff = abs(color1.red() - color2.red())
        g_diff = abs(color1.green() - color2.green())
        b_diff = abs(color1.blue() - color2.blue())

        return (r_diff <= tolerance and
                g_diff <= tolerance and
                b_diff <= tolerance)

    # Tool setters
    def set_tool(self, tool):
        """Set the current drawing tool"""
        self.current_tool = tool
        self.glitter_mode = (tool == "glitter")

        # Change cursor based on tool
        if tool == "fill":
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def set_color(self, color):
        """Set the brush color"""
        self.brush_color = color

    def set_brush_size(self, size):
        """Set the brush size"""
        self.brush_size = size

    def set_contour_protection(self, enabled):
        """Enable or disable contour protection"""
        self.protect_contours = enabled

    # History management
    def save_to_history(self):
        """Save current state to history"""
        if self.original_image:
            # Remove future history if we're not at the end
            if self.history_index < len(self.history) - 1:
                self.history = self.history[:self.history_index + 1]

            # Add new state
            self.history.append(self.original_image.copy())

            # Limit history size
            if len(self.history) > self.max_history:
                self.history.pop(0)
            else:
                self.history_index += 1

            self.canvas_changed.emit()

    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.original_image = self.history[self.history_index].copy()
            self.update_display_image()
            self.canvas_changed.emit()

    def redo(self):
        """Redo last undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.original_image = self.history[self.history_index].copy()
            self.update_display_image()
            self.canvas_changed.emit()

    def clear_canvas(self):
        """Clear the canvas to white"""
        if self.original_image:
            self.original_image.fill(Qt.white)

            # Reapply base image if exists
            if self.base_image_original:
                self._apply_base_to_original()

            self.update_display_image()
            self.save_to_history()

    def save_image(self, filepath):
        """Save the current image to file"""
        if self.original_image:
            success = self.original_image.save(filepath)
            if not success:
                raise Exception("Failed to save image to the specified path")
        else:
            raise Exception("No image to save")


class PainterApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.has_unsaved_changes = False
        self._setup_window()
        self._setup_ui()
        self._setup_status_bar()

    def _setup_window(self):
        """Configure main window"""
        self.setWindowTitle("Raccoon Painter - JSON Direct Support")
        self.setGeometry(100, 100, 1200, 800)

    def _setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Canvas
        self.canvas = PaintCanvas()
        self.canvas.canvas_changed.connect(self._on_canvas_changed)
        main_layout.addWidget(self.canvas, stretch=1)

        # Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

    def _setup_status_bar(self):
        """Setup status bar"""
        self.statusBar().showMessage("Ready. Load an image to start!")

    def _on_canvas_changed(self):
        """Handle canvas change signal"""
        self.has_unsaved_changes = True
        if not self.windowTitle().endswith(" *"):
            self.setWindowTitle(self.windowTitle() + " *")

    def _create_control_panel(self):
        """Create the control panel with all tools and options"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        panel.setMinimumWidth(250)

        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # Add all control groups
        layout.addWidget(self._create_file_group())
        layout.addWidget(self._create_tools_group())
        layout.addWidget(self._create_color_group())
        layout.addWidget(self._create_brush_size_group())
        layout.addWidget(self._create_actions_group())
        layout.addWidget(self._create_save_group())
        layout.addWidget(self._create_animation_group())

        layout.addStretch()

        return panel

    def _create_file_group(self):
        """Create file operations group"""
        group = QGroupBox("File")
        layout = QVBoxLayout()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)

        group.setLayout(layout)
        return group

    def _create_tools_group(self):
        """Create tools selection group"""
        group = QGroupBox("Tools")
        layout = QVBoxLayout()

        # Tool buttons
        tools_data = [
            ("Brush", "brush"),
            ("Fill Bucket", "fill"),
            ("Eraser", "eraser"),
            ("Glitter Brush", "glitter")
        ]

        btn_group = QButtonGroup()
        for i, (label, tool_name) in enumerate(tools_data):
            radio = QRadioButton(label)
            radio.setChecked(i == 0)
            radio.clicked.connect(lambda checked, t=tool_name: self.canvas.set_tool(t))
            btn_group.addButton(radio)
            layout.addWidget(radio)

        # Contour protection
        contour_check = QCheckBox("Protect Contours")
        contour_check.setChecked(True)
        contour_check.toggled.connect(self.canvas.set_contour_protection)
        layout.addWidget(contour_check)

        group.setLayout(layout)
        return group

    def _create_color_group(self):
        """Create color selection group"""
        group = QGroupBox("Color")
        layout = QVBoxLayout()

        # Color picker button
        self.color_btn = QPushButton("Choose Color")
        self.color_btn.setStyleSheet("background-color: black; color: white;")
        self.color_btn.clicked.connect(self.choose_color)
        layout.addWidget(self.color_btn)

        # Quick color palette
        palette_label = QLabel("Quick Palette:")
        layout.addWidget(palette_label)

        palette_layout = QHBoxLayout()
        for i, color in enumerate(QUICK_COLORS):
            btn = QPushButton()
            btn.setFixedSize(30, 30)
            btn.setStyleSheet(f"background-color: {color};")
            btn.clicked.connect(lambda checked, c=color: self.set_quick_color(c))
            palette_layout.addWidget(btn)

            if (i + 1) % 6 == 0 and i < len(QUICK_COLORS) - 1:
                layout.addLayout(palette_layout)
                palette_layout = QHBoxLayout()

        if palette_layout.count() > 0:
            layout.addLayout(palette_layout)

        group.setLayout(layout)
        return group

    def _create_brush_size_group(self):
        """Create brush size control group"""
        group = QGroupBox("Brush Size")
        layout = QVBoxLayout()

        self.size_label = QLabel(f"Size: {DEFAULT_BRUSH_SIZE} px")
        layout.addWidget(self.size_label)

        size_slider = QSlider(Qt.Horizontal)
        size_slider.setMinimum(MIN_BRUSH_SIZE)
        size_slider.setMaximum(MAX_BRUSH_SIZE)
        size_slider.setValue(DEFAULT_BRUSH_SIZE)
        size_slider.valueChanged.connect(self.change_brush_size)
        layout.addWidget(size_slider)

        group.setLayout(layout)
        return group

    def _create_actions_group(self):
        """Create actions group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.canvas.undo)
        layout.addWidget(undo_btn)

        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(self.canvas.redo)
        layout.addWidget(redo_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_with_confirm)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    def _create_save_group(self):
        """Create save group"""
        group = QGroupBox("Save")
        layout = QVBoxLayout()

        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self.save_image)
        layout.addWidget(save_btn)

        group.setLayout(layout)
        return group

    def _create_animation_group(self):
        """Create animation group"""
        group = QGroupBox("Animation")
        layout = QVBoxLayout()

        animate_btn = QPushButton("Create Animation")
        animate_btn.clicked.connect(self.create_animation)
        layout.addWidget(animate_btn)

        info = QLabel("Color the image (frame 0) then create animation using meta.json and transform.json ")
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        group.setLayout(layout)
        return group

    # Action handlers
    def load_image(self):
        """Load an image file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image (Frame 0)",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if filepath:
            try:
                self.canvas.load_base_image(filepath)
                self.statusBar().showMessage(f"Loaded: {filepath}")
                self.has_unsaved_changes = False
                self.setWindowTitle("Raccoon Painter - JSON Direct Support")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def choose_color(self):
        """Open color dialog"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_color(color)
            self._update_color_button(color)
            self.statusBar().showMessage(f"Color: {color.name()}")

    def set_quick_color(self, hex_color):
        """Set color from quick palette"""
        color = QColor(hex_color)
        self.canvas.set_color(color)
        self._update_color_button(color)
        self.statusBar().showMessage(f"Color: {hex_color}")

    def _update_color_button(self, color):
        """Update color button appearance"""
        text_color = 'white' if color.lightness() < 128 else 'black'
        self.color_btn.setStyleSheet(
            f"background-color: {color.name()}; color: {text_color};"
        )

    def change_brush_size(self, value):
        """Handle brush size change"""
        self.canvas.set_brush_size(value)
        self.size_label.setText(f"Size: {value} px")

    def clear_with_confirm(self):
        """Clear canvas with confirmation dialog"""
        reply = QMessageBox.question(
            self,
            "Confirmation",
            "Are you sure you want to clear the canvas?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.canvas.clear_canvas()
            self.statusBar().showMessage("Canvas cleared")

    def save_image(self):
        """Save the current image"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "painted_frame.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )

        if filepath:
            try:
                self.canvas.save_image(filepath)
                self.statusBar().showMessage(f"Saved: {filepath}")
                self.has_unsaved_changes = False
                self.setWindowTitle("Raccoon Painter - JSON Direct Support")
                QMessageBox.information(self, "Success", f"Image saved:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def create_animation(self):
        """Create animation from coloring - works directly with data.json"""
        if AnimationWindow is None:
            QMessageBox.critical(
                self,
                "Error",
                "Failed to import 'animation_system.py'.\n"
                "Animation features are disabled."
            )
            return

        # Check if there's a colored image
        if self.canvas.original_image is None or self.canvas.base_image_path is None:
            QMessageBox.warning(
                self,
                "Warning",
                "First load and color an image (Frame 0)!"
            )
            return

        # Temporarily save colored image
        temp_dir = tempfile.gettempdir()
        temp_painted = os.path.join(temp_dir, "painted_frame_temp.png")

        try:
            self.canvas.save_image(temp_painted)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save temporary file:\n{str(e)}")
            return

        # Launch animation window (it will ask for data.json directly)
        try:
            self.anim_window = AnimationWindow(
                painted_image_path=temp_painted
            )
            self.anim_window.show()
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to create animation:\n{str(e)}")

    def closeEvent(self, event):
        """Handle window closing"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Close without saving?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style for better appearance
    app.setStyle('Fusion')

    window = PainterApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()