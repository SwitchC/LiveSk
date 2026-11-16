"""
Animation System - Bridge between Painter and Texture Animator
Provides user-friendly interface for creating and viewing animations
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QFileDialog, QMessageBox, QTextEdit, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget


class AnimatorThread(QThread):
    """Thread to run the texture animator without freezing UI"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, meta_json_path, transform_json_path, first_frame_path, output_dir):
        super().__init__()
        self.meta_json_path = meta_json_path
        self.transform_json_path = transform_json_path
        self.first_frame_path = first_frame_path
        self.output_dir = output_dir

    def run(self):
        """Run the animator in background"""
        try:
            self.progress.emit("Starting animation process...")

            # Import and run animator
            from texture_animator import TextureAnimator

            animator = TextureAnimator(
                data_path=self.meta_json_path,
                first_frame_path=self.first_frame_path,
                output_dir=self.output_dir,
                transform_path=self.transform_json_path
            )

            self.progress.emit("Extracting textures from painted image...")
            animator.extract_textures_from_first_frame()

            self.progress.emit("Applying textures to all frames...")
            animator.animate()

            self.progress.emit("Animation complete!")

            # Check if video was created
            video_path = Path(self.output_dir) / "animation.mp4"
            if video_path.exists():
                self.finished.emit(True, str(video_path))
            else:
                self.finished.emit(False, "Video file was not created")

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            self.progress.emit(error_msg)
            self.finished.emit(False, error_msg)


class VideoPlayerWindow(QDialog):
    """Video player window to view the created animation"""

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.setWindowTitle("Animation Player - Raccoon Painter")
        self.setGeometry(200, 200, 800, 650)

        self._setup_ui()
        self._setup_media_player()

    def _setup_ui(self):
        """Setup the video player UI"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🎬 Your Animation is Ready!")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        layout.addWidget(self.video_widget)

        # Controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play_pause)
        controls_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop)
        controls_layout.addWidget(self.stop_btn)

        self.replay_btn = QPushButton("🔄 Replay")
        self.replay_btn.clicked.connect(self.replay)
        controls_layout.addWidget(self.replay_btn)

        controls_layout.addStretch()

        self.open_folder_btn = QPushButton("📁 Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        controls_layout.addWidget(self.open_folder_btn)

        layout.addLayout(controls_layout)

        # Info
        info = QLabel(f"Video: {Path(self.video_path).name}")
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

    def _setup_media_player(self):
        """Setup the media player"""
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # Load video
        media_url = QUrl.fromLocalFile(str(self.video_path))
        self.media_player.setMedia(QMediaContent(media_url))

        # Connect state changed signal
        self.media_player.stateChanged.connect(self.on_state_changed)

        # Auto-play
        QTimer.singleShot(500, self.play)

    def play_pause(self):
        """Toggle play/pause"""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def play(self):
        """Play video"""
        self.media_player.play()

    def stop(self):
        """Stop video"""
        self.media_player.stop()

    def replay(self):
        """Replay from beginning"""
        self.media_player.setPosition(0)
        self.media_player.play()

    def on_state_changed(self, state):
        """Update button text based on state"""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("⏸ Pause")
        else:
            self.play_btn.setText("▶ Play")

    def open_output_folder(self):
        """Open the output folder in file explorer"""
        output_dir = Path(self.video_path).parent

        # Cross-platform way to open folder
        if sys.platform == 'win32':
            os.startfile(output_dir)
        elif sys.platform == 'darwin':
            subprocess.run(['open', output_dir])
        else:
            subprocess.run(['xdg-open', output_dir])

    def closeEvent(self, event):
        """Cleanup when closing"""
        self.media_player.stop()
        event.accept()


class AnimationWindow(QDialog):
    """Main animation creation window"""

    def __init__(self, painted_image_path=None, parent=None):
        super().__init__(parent)
        self.painted_image_path = painted_image_path
        self.meta_json_path = None
        self.transform_json_path = None
        self.output_dir = None
        self.animator_thread = None

        self.setWindowTitle("Animation Creator - Raccoon Painter")
        self.setGeometry(150, 150, 700, 500)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the animation window UI"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🎨 Create Animation from Your Painting")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Instructions
        instructions = QGroupBox("📋 Instructions")
        instr_layout = QVBoxLayout()
        instr_text = QLabel(
            "1. You've already colored Frame 0 (the first frame)\n"
            "2. Select your meta.json file\n"
            "3. Select your transform.json file\n"
            "4. Click 'Create Animation' and wait\n"
            "5. Watch your animation come to life!"
        )
        instr_text.setWordWrap(True)
        instr_layout.addWidget(instr_text)
        instructions.setLayout(instr_layout)
        layout.addWidget(instructions)

        # File selection
        files_group = QGroupBox("📁 Required Files")
        files_layout = QVBoxLayout()

        # Data JSON
        meta_layout = QHBoxLayout()
        meta_layout.addWidget(QLabel("meta.json:"))
        self.meta_label = QLabel("Not selected")
        self.meta_label.setStyleSheet("color: #999;")
        meta_layout.addWidget(self.meta_label, stretch=1)
        meta_btn = QPushButton("Browse...")
        meta_btn.clicked.connect(self.select_meta_json)
        meta_layout.addWidget(meta_btn)
        files_layout.addLayout(meta_layout)
        
        # Transform JSON
        transform_layout = QHBoxLayout()
        transform_layout.addWidget(QLabel("transform.json:"))
        self.transform_label = QLabel("Not selected")
        self.transform_label.setStyleSheet("color: #999;")
        transform_layout.addWidget(self.transform_label, stretch=1)
        transform_btn = QPushButton("Browse...")
        transform_btn.clicked.connect(self.select_transform_json)
        transform_layout.addWidget(transform_btn)
        files_layout.addLayout(transform_layout)

        # Painted image
        painted_layout = QHBoxLayout()
        painted_layout.addWidget(QLabel("Painted Frame 0:"))
        self.painted_label = QLabel(Path(self.painted_image_path).name if self.painted_image_path else "Not set")
        self.painted_label.setStyleSheet("color: #080;")
        painted_layout.addWidget(self.painted_label, stretch=1)
        files_layout.addLayout(painted_layout)

        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_label = QLabel("Will be auto-generated")
        self.output_label.setStyleSheet("color: #999;")
        output_layout.addWidget(self.output_label, stretch=1)
        output_btn = QPushButton("Change...")
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(output_btn)
        files_layout.addLayout(output_layout)

        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        # Progress area
        progress_group = QGroupBox("⚙️ Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setPlaceholderText("Animation logs will appear here...")
        progress_layout.addWidget(self.log_text)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.create_btn = QPushButton("🎬 Create Animation")
        self.create_btn.setMinimumHeight(40)
        self.create_btn.clicked.connect(self.start_animation)
        button_layout.addWidget(self.create_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def select_meta_json(self):
        """Select meta.json file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select meta.json File",
            "",
            "JSON Files (*.json)"
        )

        if filepath:
            self.meta_json_path = filepath
            self.meta_label.setText(Path(filepath).name)
            self.meta_label.setStyleSheet("color: #080;")
            self.log_text.append(f"✓ Loaded: {Path(filepath).name}")

    def select_transform_json(self):
        """Select transform.json file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select transform.json File",
            "",
            "JSON Files (*.json)"
        )

        if filepath:
            self.transform_json_path = filepath
            self.transform_label.setText(Path(filepath).name)
            self.transform_label.setStyleSheet("color: #080;")
            self.log_text.append(f"✓ Loaded: {Path(filepath).name}")

    def select_output_dir(self):
        """Select output directory"""
        dirpath = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )

        if dirpath:
            self.output_dir = dirpath
            self.output_label.setText(str(Path(dirpath).name))
            self.output_label.setStyleSheet("color: #080;")
            self.log_text.append(f"✓ Output folder: {dirpath}")

    def start_animation(self):
        """Start the animation creation process"""
        # Validate inputs
        if not self.meta_json_path:
            QMessageBox.warning(
                self,
                "Missing File",
                "Please select the meta.json file first!"
            )
            return
        if not self.transform_json_path:
            QMessageBox.warning(
                self,
                "Missing File",
                "Please select the transform.json file first!"
            )
            return
        if not self.painted_image_path or not Path(self.painted_image_path).exists():
            QMessageBox.warning(
                self,
                "Missing File",
                "Painted image not found!"
            )
            return

        # Set default output directory if not set
        if not self.output_dir:
            self.output_dir = str(Path(tempfile.gettempdir()) / "raccoon_animation_output")
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.output_label.setText(Path(self.output_dir).name)

        # Disable controls
        self.create_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log_text.append("🚀 Starting animation creation...")

        # Start animator thread
        self.animator_thread = AnimatorThread(
            self.meta_json_path,
            self.transform_json_path,
            self.painted_image_path,
            self.output_dir
        )

        self.animator_thread.progress.connect(self.on_progress)
        self.animator_thread.finished.connect(self.on_finished)
        self.animator_thread.start()

    def on_progress(self, message):
        """Handle progress updates"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def on_finished(self, success, message):
        """Handle animation completion"""
        self.progress_bar.setVisible(False)
        self.create_btn.setEnabled(True)

        if success:
            self.log_text.append("\n✅ Animation created successfully!")

            # Show success message
            reply = QMessageBox.question(
                self,
                "Success!",
                "Animation created successfully!\n\nWould you like to watch it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                # Open video player
                self.video_player = VideoPlayerWindow(message, self)
                self.video_player.show()

        else:
            self.log_text.append(f"\n❌ Animation failed:\n{message}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create animation:\n\n{message}"
            )