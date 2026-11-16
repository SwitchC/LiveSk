import sys
import json
import os
import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QListWidget, QListWidgetItem, QInputDialog, 
                             QMessageBox, QSplitter, QFrame, QLineEdit, 
                             QColorDialog, QSpinBox, QStyle, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QFontMetrics, QIcon

from AtlasManager import AtlasManager

# --- IMAGE WIDGET ---
class ImageLabel(QLabel):
    clicked = pyqtSignal(int) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.parts_map = {} 
        
        # Mode: 'layers' or 'groups'
        self.mode = 'layers'
        
        # Layer Data
        self.layer_assignments = {} # part_id -> layer_index
        self.layer_colors = {}      # layer_index -> QColor
        
        # Group Data
        self.highlighted_ids = set() # Set of IDs in current group
        self.group_color = QColor(0, 255, 0) # Green for selected items
        
        self.pixmap_original = None
        self.global_w = 1000
        self.global_h = 1000
        self.font_id = QFont("Arial", 10, QFont.Bold)

    def set_frame_data(self, parts, manager, global_w, global_h):
        self.global_w = global_w
        self.global_h = global_h
        self.parts_map = {}
        
        for part in parts:
            pid = part['id']
            mid = part['mask_id']
            bbox = part.get('bbox')
            if not bbox or len(bbox) != 4: continue
            
            mask_pil = manager.get_mask(mid)
            if mask_pil:
                mask_np = np.array(mask_pil.resize((bbox[2], bbox[3]), Image.NEAREST))
                self.parts_map[pid] = {'bbox': bbox, 'mask': mask_np}
        
        self.update_view()

    def set_mode(self, mode):
        self.mode = mode
        self.update_view()

    def update_view(self):
        img = QImage(self.global_w, self.global_h, QImage.Format_ARGB32)
        img.fill(Qt.white)
        painter = QPainter(img)
        painter.setFont(self.font_id)

        sorted_pids = list(self.parts_map.keys())
        
        # --- PASS 1: COLORS ---
        for pid in sorted_pids:
            data = self.parts_map[pid]
            x, y, w, h = data['bbox']
            mask_data = data['mask']
            
            # Determine Color based on Mode
            color = QColor(220, 220, 220) # Default Gray
            
            if self.mode == 'layers':
                l_idx = self.layer_assignments.get(pid, 0)
                color = self.layer_colors.get(l_idx, QColor(200, 200, 200))
            elif self.mode == 'groups':
                if pid in self.highlighted_ids:
                    color = self.group_color # Active Group Color
                else:
                    color = QColor(200, 200, 200) # Inactive

            # Draw Part
            c_r, c_g, c_b = color.red(), color.green(), color.blue()
            img_np = np.zeros((h, w, 4), dtype=np.uint8)
            img_np[:, :, 0] = c_b 
            img_np[:, :, 1] = c_g 
            img_np[:, :, 2] = c_r 
            img_np[:, :, 3] = (mask_data > 10) * 255 
            
            qimg_part = QImage(img_np.data, w, h, w * 4, QImage.Format_ARGB32)
            painter.drawImage(x, y, qimg_part)

        # --- PASS 2: TEXT ---
        for pid in sorted_pids:
            data = self.parts_map[pid]
            x, y, w, h = data['bbox']
            text = str(pid)
            
            fm = QFontMetrics(self.font_id)
            text_w = fm.width(text)
            text_h = fm.height()
            tx = x + (w - text_w) // 2
            ty = y + (h + text_h) // 2 - fm.descent()

            # Halo
            painter.setPen(QPen(Qt.white, 3))
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,1), (0,-1), (1,0), (-1,0)]:
                 painter.drawText(tx + dx, ty + dy, text)
            
            painter.setPen(QPen(Qt.black))
            painter.drawText(tx, ty, text)

        painter.end()
        self.pixmap_original = QPixmap.fromImage(img)
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.pixmap_original:
            scaled = self.pixmap_original.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap():
            # Hit test logic
            w_w, h_w = self.width(), self.height()
            pix = self.pixmap()
            ox = (w_w - pix.width()) / 2
            oy = (h_w - pix.height()) / 2
            
            if pix.width() == 0: return

            real_x = int((event.x() - ox) * (self.pixmap_original.width() / pix.width()))
            real_y = int((event.y() - oy) * (self.pixmap_original.width() / pix.width()))
            
            candidates = []
            for pid, data in self.parts_map.items():
                bx, by, bw, bh = data['bbox']
                if bx <= real_x < bx + bw and by <= real_y < by + bh:
                    local_x = real_x - bx
                    local_y = real_y - by
                    if 0 <= local_x < bw and 0 <= local_y < bh:
                        if data['mask'][local_y, local_x] > 10:
                            candidates.append((bw * bh, pid))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                self.clicked.emit(candidates[0][1])

    def resizeEvent(self, event):
        self.update_scaled_pixmap()
        super().resizeEvent(event)

# --- MAIN WINDOW ---

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JSON Editor: Layers & Groups")
        self.resize(1400, 900)
        
        self.json_data = {}
        self.frames_list = []
        self.json_path = ""
        
        # DATA STORAGE
        self.layers_data = [] # List of lists
        self.groups_data = {} # Dict {group_name: set(ids)}
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === LEFT PANEL ===
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(350)
        
        # 1. Global Controls
        top_ctrls = QFrame()
        top_ctrls.setFrameShape(QFrame.StyledPanel)
        top_l = QVBoxLayout(top_ctrls)
        
        btn_load = QPushButton(" Load Meta JSON")
        btn_load.clicked.connect(self.load_json)
        top_l.addWidget(btn_load)

        fr_layout = QHBoxLayout()
        fr_layout.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setEnabled(False)
        self.frame_spin.valueChanged.connect(self.load_frame)
        fr_layout.addWidget(self.frame_spin)
        top_l.addLayout(fr_layout)
        left_layout.addWidget(top_ctrls)

        # 2. TABS (Layers vs Groups)
        self.tabs = QTabWidget()
        # ВАЖЛИВО: Не підключаємо сигнал тут, бо image_display ще не створено
        
        # --- TAB 1: LAYERS ---
        self.tab_layers = QWidget()
        l_layout = QVBoxLayout(self.tab_layers)
        
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Key:"))
        self.layer_key_input = QLineEdit("layers")
        key_layout.addWidget(self.layer_key_input)
        l_layout.addLayout(key_layout)

        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QListWidget.InternalMove)
        self.layer_list.itemClicked.connect(self.on_layer_selected)
        l_layout.addWidget(QLabel("Layer Hierarchy (Bottom to Top):"))
        l_layout.addWidget(self.layer_list)

        l_btns = QHBoxLayout()
        btn_l_add = QPushButton("Add")
        btn_l_add.clicked.connect(self.add_layer)
        btn_l_col = QPushButton("Color")
        btn_l_col.clicked.connect(self.change_layer_color)
        btn_l_del = QPushButton("Del")
        btn_l_del.clicked.connect(self.delete_layer)
        l_btns.addWidget(btn_l_add)
        l_btns.addWidget(btn_l_col)
        l_btns.addWidget(btn_l_del)
        l_layout.addLayout(l_btns)
        
        self.tabs.addTab(self.tab_layers, "Layers Mode")

        # --- TAB 2: GROUPS ---
        self.tab_groups = QWidget()
        g_layout = QVBoxLayout(self.tab_groups)
        
        g_layout.addWidget(QLabel("Defined Groups (Lists):"))
        self.group_list = QListWidget()
        self.group_list.itemClicked.connect(self.on_group_selected)
        g_layout.addWidget(self.group_list)
        
        g_btns = QHBoxLayout()
        btn_g_add = QPushButton("New Group")
        btn_g_add.clicked.connect(self.add_group)
        btn_g_del = QPushButton("Delete Group")
        btn_g_del.clicked.connect(self.delete_group)
        g_btns.addWidget(btn_g_add)
        g_btns.addWidget(btn_g_del)
        g_layout.addLayout(g_btns)
        
        self.tabs.addTab(self.tab_groups, "Groups Mode")
        
        left_layout.addWidget(self.tabs)
        left_layout.addStretch()

        # 3. Save
        btn_save = QPushButton(" SAVE ALL CHANGES")
        btn_save.setStyleSheet("background-color: #2E7D32; color: white; padding: 10px; font-weight: bold;")
        btn_save.clicked.connect(self.save_json)
        left_layout.addWidget(btn_save)

        # === RIGHT PANEL ===
        self.image_display = ImageLabel()
        self.image_display.setStyleSheet("background-color: #404040;")
        self.image_display.clicked.connect(self.on_part_clicked)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.image_display, 1)

        # ВАЖЛИВО: Підключаємо сигнал зміни вкладки ТІЛЬКИ ТЕПЕР, коли self.image_display існує
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.reset_ui_state()

    def reset_ui_state(self):
        self.layer_list.clear()
        self.group_list.clear()
        self.layers_data = []
        self.groups_data = {}
        # Default Base Layer
        self.add_layer_item("Layer 0 (Default)", QColor(180, 180, 180))
        self.image_display.layer_colors = {0: QColor(180, 180, 180)}
        self.image_display.layer_assignments = {}

    # --- LOGIC: JSON LOAD ---
    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Meta JSON", "", "JSON (*.json)")
        if not path: return
        self.json_path = path
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            
            atlas_dir = os.path.dirname(path)
            self.manager = AtlasManager(self.json_data, atlas_dir)
            if self.manager.atlas_image is None:
                QMessageBox.critical(self, "Error", "Atlas image not found.")
                return

            self.frames_list = self.json_data.get('frames', [])
            if self.frames_list:
                self.frame_spin.setEnabled(True)
                self.frame_spin.setMaximum(len(self.frames_list)-1)
                self.frame_spin.setValue(0)
                self.load_frame(0)
            
            # Load Data
            self.load_layers_from_json()
            self.load_groups_from_json()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_frame(self, val):
        if val >= len(self.frames_list): return
        parts = self.frames_list[val].get('parts', [])
        g_bbox = self.json_data.get('global_bbox', {'width': 1000, 'height': 1000})
        self.image_display.set_frame_data(parts, self.manager, g_bbox['width'], g_bbox['height'])
        
        # Ensure new parts have default layer 0
        for p in parts:
            if p['id'] not in self.image_display.layer_assignments:
                self.image_display.layer_assignments[p['id']] = 0
        self.image_display.update_view()

    # --- LOGIC: TABS & MODES ---
    def on_tab_changed(self, index):
        # Додаткова перевірка, щоб уникнути помилок при старті
        if not hasattr(self, 'image_display'): return

        if index == 0:
            self.image_display.set_mode('layers')
        else:
            self.image_display.set_mode('groups')
            self.on_group_selected(self.group_list.currentItem())

    # --- LOGIC: LAYERS ---
    def load_layers_from_json(self):
        key = self.layer_key_input.text()
        self.layer_list.clear()
        self.layers_data = []
        self.image_display.layer_assignments = {}
        
        # Default
        self.add_layer_item("Layer 0", QColor(180, 180, 180))
        
        if key in self.json_data:
            loaded_layers = self.json_data[key]
            self.layer_list.clear()
            self.layers_data = []
            
            for i, pids in enumerate(loaded_layers):
                col = QColor.fromHsv((i * 45) % 360, 150, 230)
                if i == 0: col = QColor(180, 180, 180)
                self.add_layer_item(f"Layer {i}", col)
                for pid in pids:
                    self.image_display.layer_assignments[pid] = i
        self.update_layer_colors_map()
        self.image_display.update_view()

    def add_layer_item(self, name, color):
        item = QListWidgetItem(name)
        pix = QPixmap(16, 16); pix.fill(color)
        item.setIcon(QIcon(pix))
        item.setData(Qt.UserRole, color)
        self.layer_list.addItem(item)
        self.layer_list.setCurrentItem(item)

    def add_layer(self):
        idx = self.layer_list.count()
        self.add_layer_item(f"Layer {idx}", QColor.fromHsv(np.random.randint(0,360), 180, 220))
        self.update_layer_colors_map()

    def change_layer_color(self):
        item = self.layer_list.currentItem()
        if item:
            c = QColorDialog.getColor(item.data(Qt.UserRole), self)
            if c.isValid():
                item.setData(Qt.UserRole, c)
                pix = QPixmap(16, 16); pix.fill(c)
                item.setIcon(QIcon(pix))
                self.update_layer_colors_map()
                self.image_display.update_view()

    def delete_layer(self):
        r = self.layer_list.currentRow()
        if r > 0:
            self.layer_list.takeItem(r)
            # Remap parts to 0
            for pid, l in self.image_display.layer_assignments.items():
                if l == r: self.image_display.layer_assignments[pid] = 0
                elif l > r: self.image_display.layer_assignments[pid] -= 1
            self.update_layer_colors_map()
            self.image_display.update_view()

    def update_layer_colors_map(self):
        cols = {}
        for i in range(self.layer_list.count()):
            cols[i] = self.layer_list.item(i).data(Qt.UserRole)
        self.image_display.layer_colors = cols

    def on_layer_selected(self, item):
        pass 

    # --- LOGIC: GROUPS ---
    def load_groups_from_json(self):
        # Очищаємо попередні дані
        self.group_list.clear()
        self.groups_data = {}

        # 1. Скануємо JSON на наявність будь-яких валідних груп
        ignore_keys = {"uv_map", "frames", "global_bbox", "atlas", "layers"}
        for k, v in self.json_data.items():
            if k not in ignore_keys and isinstance(v, list) and all(isinstance(x, int) for x in v):
                # Це виглядає як група ID
                self.groups_data[k] = set(v)
                self.group_list.addItem(k)

        # 2. Обов'язково додаємо 'penalty_scale' та 'non_drawable_part_ids', якщо їх ще немає
        mandatory_groups = ["penalty_scale", "non_drawable_part_ids"]
        for group_name in mandatory_groups:
            if group_name not in self.groups_data:
                # Якщо група була в JSON, але не пройшла перевірку на "int list" (наприклад, була пустою або містила щось інше), 
                # спробуємо завантажити її як є, або створимо нову порожню.
                if group_name in self.json_data and isinstance(self.json_data[group_name], list):
                    self.groups_data[group_name] = set(self.json_data[group_name])
                else:
                    self.groups_data[group_name] = set()
                
                self.group_list.addItem(group_name)

    def add_group(self):
        text, ok = QInputDialog.getText(self, "New Group", "Enter JSON Key for group (e.g. 'non_drawable_part_ids'):")
        if ok and text:
            if text in self.groups_data:
                QMessageBox.warning(self, "Exists", "Group already loaded.")
                return
            
            ids_set = set()
            if text in self.json_data and isinstance(self.json_data[text], list):
                 ids_set = set(self.json_data[text])
                 QMessageBox.information(self, "Found", f"Loaded {len(ids_set)} items from existing JSON data.")
            
            self.groups_data[text] = ids_set
            self.group_list.addItem(text)
            self.group_list.setCurrentRow(self.group_list.count()-1)

    def delete_group(self):
        row = self.group_list.currentRow()
        if row >= 0:
            name = self.group_list.item(row).text()
            del self.groups_data[name]
            self.group_list.takeItem(row)
            self.image_display.highlighted_ids = set()
            self.image_display.update_view()

    def on_group_selected(self, item):
        if not item: return
        name = item.text()
        self.image_display.highlighted_ids = self.groups_data.get(name, set())
        self.image_display.update_view()

    # --- LOGIC: CLICK HANDLER ---
    def on_part_clicked(self, part_id):
        if self.tabs.currentIndex() == 0: 
            # LAYER MODE
            row = self.layer_list.currentRow()
            if row >= 0:
                self.image_display.layer_assignments[part_id] = row
                self.image_display.update_view()
        else:
            # GROUP MODE
            item = self.group_list.currentItem()
            if not item: 
                QMessageBox.warning(self, "Select Group", "Please create or select a group in the list first.")
                return
            name = item.text()
            
            if part_id in self.groups_data[name]:
                self.groups_data[name].remove(part_id)
            else:
                self.groups_data[name].add(part_id)
            
            self.image_display.highlighted_ids = self.groups_data[name]
            self.image_display.update_view()

    # --- SAVE ---
    def save_json(self):
        if not self.json_path: return
        
        # 1. Save Layers
        l_key = self.layer_key_input.text()
        num_l = self.layer_list.count()
        out_layers = [[] for _ in range(num_l)]
        for pid, l_idx in self.image_display.layer_assignments.items():
            if 0 <= l_idx < num_l:
                out_layers[l_idx].append(pid)
        for l in out_layers: l.sort()
        self.json_data[l_key] = out_layers
        
        # 2. Save Groups
        for g_name, g_ids in self.groups_data.items():
            self.json_data[g_name] = sorted(list(g_ids))
            
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2)
            QMessageBox.information(self, "Saved", "All data saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())