import sys
import os
import cv2
import numpy as np
import glob
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QSlider, QSpinBox, QFrame, QColorDialog, 
                             QProgressBar, QStyle, QGroupBox, QSizePolicy, 
                             QScrollArea, QListWidget, QListWidgetItem, QSplitter) # <--- Додано QListWidgetItem та QSplitter
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QImage, QPixmap, QColor, QCursor, QPainter

# --- WORKER: Пакетна обробка ---
class BatchReplaceWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    
    def __init__(self, file_paths, replacements):
        super().__init__()
        self.files = file_paths
        self.replacements = replacements

    def run(self):
        count = len(self.files)
        if count == 0:
            self.finished.emit("No files to process.")
            return

        # Підготовка списку замін (RGB -> BGR для OpenCV)
        todo_list = []
        for src, dst in self.replacements:
            src_bgr = np.array([src[2], src[1], src[0]], dtype=np.uint8)
            dst_bgr = np.array([dst[2], dst[1], dst[0]], dtype=np.uint8)
            todo_list.append((src_bgr, dst_bgr))

        changed_files_count = 0

        for i, path in enumerate(self.files):
            try:
                img = cv2.imread(path)
                if img is not None:
                    img_modified = False
                    for src_bgr, dst_bgr in todo_list:
                        # Створюємо маску для конкретного кольору
                        mask = cv2.inRange(img, src_bgr, src_bgr)
                        if cv2.countNonZero(mask) > 0:
                            img[mask > 0] = dst_bgr
                            img_modified = True
                    
                    if img_modified:
                        cv2.imwrite(path, img)
                        changed_files_count += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

            self.progress.emit(int(((i + 1) / count) * 100))

        self.finished.emit(f"Done! Updated {changed_files_count} files.")

# --- UI: Картинка (Внутрішній віджет) ---
class ImageCanvas(QLabel):
    # Сигнал: (x, y) у координатах оригінального зображення
    pixel_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setScaledContents(True)
        self.setMouseTracking(False)
        self.original_w = 0
        self.original_h = 0

    def set_dims(self, w, h):
        self.original_w = w
        self.original_h = h

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Отримуємо розміри віджета на екрані
            lbl_w = self.width()
            lbl_h = self.height()
            
            if lbl_w > 0 and lbl_h > 0 and self.original_w > 0:
                # Пропорція
                ratio_x = self.original_w / lbl_w
                ratio_y = self.original_h / lbl_h
                
                real_x = int(event.x() * ratio_x)
                real_y = int(event.y() * ratio_y)
                
                # Захист меж
                real_x = max(0, min(real_x, self.original_w - 1))
                real_y = max(0, min(real_y, self.original_h - 1))
                
                self.pixel_clicked.emit(real_x, real_y)
        
        super().mousePressEvent(event)

# --- UI: ScrollArea (Зум і Панорамування) ---
class ZoomViewer(QScrollArea):
    image_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #202020;")
        self.setAlignment(Qt.AlignCenter)
        
        self.canvas = ImageCanvas()
        self.canvas.pixel_clicked.connect(self.image_clicked)
        self.setWidget(self.canvas)
        
        self.qimage = None
        self.zoom_factor = 1.0
        
        # Для перетягування (Pan)
        self.panning = False
        self.last_mouse_pos = QPoint()

    def set_image(self, qimg):
        self.qimage = qimg
        self.canvas.setPixmap(QPixmap.fromImage(qimg))
        self.canvas.set_dims(qimg.width(), qimg.height())
        self.canvas.adjustSize()
        self.update_zoom()

    def update_zoom(self):
        if not self.qimage: return
        new_w = int(self.qimage.width() * self.zoom_factor)
        new_h = int(self.qimage.height() * self.zoom_factor)
        self.canvas.setFixedSize(new_w, new_h)

    def fit_view(self):
        """Вписати картинку у вікно"""
        if not self.qimage: return
        viewport_w = self.viewport().width()
        viewport_h = self.viewport().height()
        
        if self.qimage.width() > 0 and self.qimage.height() > 0:
            ratio_w = viewport_w / self.qimage.width()
            ratio_h = viewport_h / self.qimage.height()
            self.zoom_factor = min(ratio_w, ratio_h) * 0.95
            self.update_zoom()

    def wheelEvent(self, event):
        if not self.qimage: return
        # Зум колесом
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.zoom_factor = max(0.05, min(self.zoom_factor * factor, 50.0))
        self.update_zoom()

    def mousePressEvent(self, event):
        # Права кнопка для перетягування
        if event.button() == Qt.RightButton:
            self.panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


# --- MAIN APP ---
# --- MAIN APP ---
class ColorReplaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Color Replacer (Adjustable UI)")
        self.resize(1400, 900)
        
        self.files = []
        self.current_idx = 0
        self.current_img_cv = None # BGR Image
        
        self.target_color = QColor(0, 255, 0) # Зелений
        self.replacements = [] # [(src_rgb, dst_rgb), ...]
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5) # Тонші відступи
        main_layout.setSpacing(5)

        # === 1. TOP BAR ===
        top_bar = QHBoxLayout()
        # ... (код верхньої панелі той самий) ...
        btn_open = QPushButton(" Open Folder")
        btn_open.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        btn_open.setStyleSheet("padding: 5px 15px; font-weight: bold;")
        btn_open.clicked.connect(self.open_folder)
        top_bar.addWidget(btn_open)
        
        self.lbl_info = QLabel("No folder loaded")
        self.lbl_info.setStyleSheet("color: #AAA; margin-left: 10px;")
        top_bar.addWidget(self.lbl_info)
        
        top_bar.addStretch()
        
        btn_fit = QPushButton("Fit Image")
        btn_fit.clicked.connect(lambda: self.viewer.fit_view())
        top_bar.addWidget(btn_fit)
        
        main_layout.addLayout(top_bar)

        # === 2. VERTICAL SPLITTER (Main Content / Bottom Nav) ===
        # Ми використовуємо вертикальний сплітер, щоб можна было тягати лінію між картинкою і навбаром
        v_splitter = QSplitter(Qt.Vertical)
        
        # --- TOP PART OF V_SPLITTER (Horizontal Splitter: Controls | Image) ---
        h_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setMinimumWidth(250)
        l_layout = QVBoxLayout(left_panel)
        
        # ... (Код лівої панелі той самий) ...
        g_target = QGroupBox("1. Target Color")
        gt_layout = QVBoxLayout(g_target)
        h_col = QHBoxLayout()
        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(60, 40)
        self.update_preview()
        h_col.addWidget(self.lbl_preview)
        v_btns = QVBoxLayout()
        btn_pal = QPushButton("Palette...")
        btn_pal.clicked.connect(self.pick_palette)
        v_btns.addWidget(btn_pal)
        self.btn_pipette = QPushButton("Pipette")
        self.btn_pipette.setCheckable(True)
        self.btn_pipette.clicked.connect(self.toggle_pipette)
        v_btns.addWidget(self.btn_pipette)
        h_col.addLayout(v_btns)
        gt_layout.addLayout(h_col)
        l_layout.addWidget(g_target)
        
        lbl_hint = QLabel("<b>2. Click on Image to Replace</b>")
        lbl_hint.setStyleSheet("color: #CCC; margin: 5px 0;")
        l_layout.addWidget(lbl_hint)
        
        self.list_wdg = QListWidget()
        l_layout.addWidget(self.list_wdg)
        
        btn_undo = QPushButton("Undo Last")
        btn_undo.clicked.connect(self.undo_last)
        l_layout.addWidget(btn_undo)
        
        self.btn_apply = QPushButton("APPLY TO ALL FILES")
        self.btn_apply.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 10px;")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.run_batch)
        l_layout.addWidget(self.btn_apply)
        
        h_splitter.addWidget(left_panel)
        
        # Viewer
        self.viewer = ZoomViewer()
        self.viewer.image_clicked.connect(self.on_pixel_click)
        h_splitter.addWidget(self.viewer)
        
        h_splitter.setStretchFactor(1, 1) # Viewer takes more space
        
        # Додаємо верхню частину (горизонтальний сплітер) у вертикальний сплітер
        v_splitter.addWidget(h_splitter)

        # --- BOTTOM PART OF V_SPLITTER (Navigation) ---
        bottom_container = QWidget()
        b_layout = QVBoxLayout(bottom_container)
        b_layout.setContentsMargins(0,0,0,0)
        
        nav_bar = QFrame()
        nav_bar.setFrameShape(QFrame.StyledPanel)
        nav_bar.setStyleSheet("background-color: #2A2A2A;")
        nav_l = QHBoxLayout(nav_bar)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.change_frame)
        
        self.spin = QSpinBox()
        self.spin.setEnabled(False)
        self.spin.valueChanged.connect(self.change_frame)
        
        nav_l.addWidget(QLabel("Frame:"))
        nav_l.addWidget(self.spin)
        nav_l.addWidget(self.slider)
        
        b_layout.addWidget(nav_bar)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar {height: 6px; border: none;} QProgressBar::chunk {background-color: #0078D7;}")
        b_layout.addWidget(self.progress)
        
        v_splitter.addWidget(bottom_container)
        
        # Налаштовуємо початкові розміри вертикального сплітера
        # Верхня частина (контент) отримує більше місця, нижня (навігація) - мінімум
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        # Можна задати початковий розмір у пікселях, якщо треба
        # v_splitter.setSizes([800, 100]) 

        main_layout.addWidget(v_splitter)

    # --- LOGIC ---

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not d: return
        
        path = os.path.join(d, "colored_*.png")
        self.files = sorted(glob.glob(path))
        
        if not self.files:
            QMessageBox.warning(self, "Error", "No 'colored_*.png' files found!")
            return
            
        self.lbl_info.setText(f"{len(self.files)} frames | {d}")
        
        self.slider.setRange(0, len(self.files)-1)
        self.spin.setRange(0, len(self.files)-1)
        self.slider.setEnabled(True)
        self.spin.setEnabled(True)
        self.btn_apply.setEnabled(True)
        
        self.current_idx = 0
        self.slider.setValue(0)
        self.load_frame(0)
        self.viewer.fit_view()

    def change_frame(self, idx):
        self.current_idx = idx
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(idx)
        self.spin.setValue(idx)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        self.load_frame(idx)

    def load_frame(self, idx):
        if idx < 0 or idx >= len(self.files): return
        path = self.files[idx]
        
        # Load OpenCV BGR
        self.current_img_cv = cv2.imread(path)
        if self.current_img_cv is None: return
        
        # Apply pending visual changes
        self.apply_visuals()
        self.show_image()

    def show_image(self):
        if self.current_img_cv is None: return
        # CV2 (BGR) -> Qt (RGB)
        rgb = cv2.cvtColor(self.current_img_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_line, QImage.Format_RGB888)
        self.viewer.set_image(qimg)

    def apply_visuals(self):
        """Applies replacements to the current CV image in memory."""
        for src_rgb, dst_rgb in self.replacements:
            src_bgr = np.array([src_rgb[2], src_rgb[1], src_rgb[0]], dtype=np.uint8)
            dst_bgr = np.array([dst_rgb[2], dst_rgb[1], dst_rgb[0]], dtype=np.uint8)
            
            mask = cv2.inRange(self.current_img_cv, src_bgr, src_bgr)
            if cv2.countNonZero(mask) > 0:
                self.current_img_cv[mask > 0] = dst_bgr

    # --- COLOR TOOLS ---

    def update_preview(self):
        self.lbl_preview.setStyleSheet(f"background: {self.target_color.name()}; border: 1px solid white;")

    def pick_palette(self):
        # Вимкнути піпетку, якщо була
        if self.btn_pipette.isChecked():
            self.toggle_pipette()
            
        c = QColorDialog.getColor(self.target_color, self)
        if c.isValid():
            self.target_color = c
            self.update_preview()

    def toggle_pipette(self):
        active = self.btn_pipette.isChecked()
        if active:
            self.viewer.setCursor(Qt.CrossCursor)
            self.btn_pipette.setStyleSheet("background-color: orange; color: black; font-weight: bold;")
        else:
            self.viewer.setCursor(Qt.ArrowCursor)
            self.btn_pipette.setStyleSheet("")

    def on_pixel_click(self, x, y):
        if self.current_img_cv is None: return
        
        try:
            # OpenCV is y, x
            b, g, r = self.current_img_cv[y, x]
            clicked_col = QColor(r, g, b)
        except: return

        # 1. PIPETTE MODE (Picking Target)
        if self.btn_pipette.isChecked():
            self.target_color = clicked_col
            self.update_preview()
            self.btn_pipette.setChecked(False) # Auto-off
            self.toggle_pipette()
            return

        # 2. PAINT MODE (Replacement)
        if clicked_col == self.target_color: return # Same color
        
        src_rgb = (clicked_col.red(), clicked_col.green(), clicked_col.blue())
        dst_rgb = (self.target_color.red(), self.target_color.green(), self.target_color.blue())
        
        # Check duplicates
        if (src_rgb, dst_rgb) in self.replacements: return
        
        # Add rule
        self.replacements.append((src_rgb, dst_rgb))
        self.list_wdg.addItem(f"RGB{src_rgb} -> RGB{dst_rgb}")
        self.list_wdg.scrollToBottom()
        
        # Update view
        self.apply_visuals()
        self.show_image()

    def undo_last(self):
        if not self.replacements: return
        self.replacements.pop()
        self.list_wdg.takeItem(self.list_wdg.count() - 1)
        # Reload clean frame and re-apply remaining replacements
        self.load_frame(self.current_idx)

    # --- BATCH RUN ---

    def run_batch(self):
        if not self.replacements:
            QMessageBox.warning(self, "Info", "No replacements defined.")
            return
            
        ans = QMessageBox.question(self, "Confirm", 
                                   f"Apply {len(self.replacements)} rules to {len(self.files)} files?\nIrreversible!",
                                   QMessageBox.Yes | QMessageBox.No)
        if ans == QMessageBox.No: return
        
        self.setEnabled(False)
        self.progress.setValue(0)
        
        self.worker = BatchReplaceWorker(self.files, self.replacements)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, msg):
        self.setEnabled(True)
        self.replacements.clear()
        self.list_wdg.clear()
        QMessageBox.information(self, "Finished", msg)
        self.load_frame(self.current_idx)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark Theme
    p = app.palette()
    p.setColor(p.Window, QColor(53, 53, 53))
    p.setColor(p.WindowText, Qt.white)
    p.setColor(p.Base, QColor(25, 25, 25))
    p.setColor(p.AlternateBase, QColor(53, 53, 53))
    p.setColor(p.ToolTipBase, Qt.white)
    p.setColor(p.ToolTipText, Qt.white)
    p.setColor(p.Text, Qt.white)
    p.setColor(p.Button, QColor(53, 53, 53))
    p.setColor(p.ButtonText, Qt.white)
    p.setColor(p.BrightText, Qt.red)
    p.setColor(p.Link, QColor(42, 130, 218))
    p.setColor(p.Highlight, QColor(42, 130, 218))
    p.setColor(p.HighlightedText, Qt.black)
    app.setPalette(p)

    w = ColorReplaceApp()
    w.show()
    sys.exit(app.exec_())