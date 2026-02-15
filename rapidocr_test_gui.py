"""
PySide6 RapidOCR Test Application
Tests Korean PP-OCRv5 model with images
"""
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from rapidocr import RapidOCR


class OCRWorker(QThread):
    """Worker thread for OCR processing"""
    progress = Signal(int, int, str)  # current, total, filename
    result = Signal(str, list, float)  # filename, results, elapsed_time
    finished_all = Signal()
    error = Signal(str)

    def __init__(self, engine: RapidOCR, image_paths: List[str], draw_boxes: bool = True, contrast_factor: float = 0.5):
        super().__init__()
        self.engine = engine
        self.image_paths = image_paths
        self.draw_boxes = draw_boxes
        self.contrast_factor = contrast_factor
        self._is_running = True
        
    def run(self):
        try:
            for idx, img_path in enumerate(self.image_paths):
                if not self._is_running:
                    break

                self.progress.emit(idx + 1, len(self.image_paths), os.path.basename(img_path))

                start_time = time.time()

                # Load image and apply contrast adjustment
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Apply contrast enhancement
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.contrast_factor)

                # Convert PIL to numpy array for OCR engine
                img_array = np.array(img)

                output = self.engine(img_array)
                if output.txts:
                    result = [[box, txt, score] for box, txt, score in zip(output.boxes, output.txts, output.scores)]
                else:
                    result = []
                elapse = output.elapse
                elapsed = time.time() - start_time

                if result is None:
                    result = []

                self.result.emit(img_path, result, elapsed)

            self.finished_all.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class ImageViewer(QWidget):
    """Custom image viewer with zoom and pan support"""
    def __init__(self):
        super().__init__()
        self._init_ui()
        self.current_pixmap: Optional[QPixmap] = None
        self.original_image: Optional[Image.Image] = None
        self.ocr_results: List = []
        self.show_boxes = True
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #2d2d2d;")
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
        # Info label
        self.info_label = QLabel("No image loaded")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.info_label)
        
    def load_image(self, image_path: str):
        """Load an image from file"""
        try:
            self.original_image = Image.open(image_path)
            self.display_image(with_boxes=False)
            self.info_label.setText(f"Image: {os.path.basename(image_path)} ({self.original_image.size[0]}x{self.original_image.size[1]})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"Error loading image: {str(e)}")
            
    def display_image(self, with_boxes: bool = False):
        """Display the current image"""
        if self.original_image is None:
            return
            
        img = self.original_image.copy()
        
        # Draw bounding boxes if enabled and results exist
        if with_boxes and self.ocr_results and self.show_boxes:
            img = self._draw_boxes_on_image(img, self.ocr_results)
            
        # Convert PIL to QPixmap
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        height, width, channels = img_array.shape
        
        # Convert to QImage
        bytes_per_line = channels * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.current_pixmap = QPixmap.fromImage(q_image)
        self._update_display()
        
    def _draw_boxes_on_image(self, img: Image.Image, results: List) -> Image.Image:
        """Draw bounding boxes and text on image"""
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
        ]
        
        for idx, res in enumerate(results):
            if len(res) >= 2:
                box = res[0]
                text = res[1]
                confidence = res[2] if len(res) > 2 else 0.0
                
                color = colors[idx % len(colors)]
                
                # Convert numpy arrays to lists
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                
                # Draw box
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    # Handle different box formats
                    # Check if it's polygon format by checking if first element is also a list/tuple
                    first_elem = box[0]
                    if hasattr(first_elem, 'tolist'):
                        first_elem = first_elem.tolist()
                    
                    if isinstance(first_elem, (list, tuple)):
                        # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        points = []
                        for p in box[:4]:
                            if hasattr(p, 'tolist'):
                                p = p.tolist()
                            points.append((int(p[0]), int(p[1])))
                        draw.polygon(points, outline=color, width=2)
                        # Draw text near first point
                        text_x, text_y = points[0]
                    else:
                        # Bounding box format: [x, y, w, h] or [x1, y1, x2, y2]
                        if len(box) == 4:
                            x1, y1, x2, y2 = map(int, box)
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            text_x, text_y = x1, y1 - 20
                        else:
                            continue
                            
                    # Draw text
                    text_to_draw = f"{text[:30]}{'...' if len(text) > 30 else ''} ({confidence:.2f})"
                    draw.text((text_x, max(0, text_y)), text_to_draw, fill=color, font=font)
                    
        return img
        
    def _update_display(self):
        """Update the display with current pixmap"""
        if self.current_pixmap:
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(
                self.scroll_area.size() - QSize(20, 20),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
    def set_ocr_results(self, results: List):
        """Set OCR results and redraw with boxes"""
        self.ocr_results = results
        self.display_image(with_boxes=self.show_boxes)
        
    def toggle_boxes(self):
        """Toggle bounding box display"""
        self.show_boxes = not self.show_boxes
        self.display_image(with_boxes=self.show_boxes)
        return self.show_boxes
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class RapidOCRTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 RapidOCR Test - Korean PP-OCRv5")
        self.setMinimumSize(1200, 800)
        
        self.ocr_engine: Optional[RapidOCR] = None
        self.image_files: List[str] = []
        self.current_image_index = -1
        self.ocr_results: dict = {}  # Store results for each image
        self.worker: Optional[OCRWorker] = None
        
        self._init_ui()
        self._init_ocr_engine()
        self._load_images()
        
    def _init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image list
        left_layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(250)
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.image_list)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self._show_previous_image)
        button_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._show_next_image)
        button_layout.addWidget(self.next_btn)
        
        left_layout.addLayout(button_layout)
        
        # OCR buttons
        ocr_btn_layout = QHBoxLayout()
        
        self.ocr_btn = QPushButton("📝 OCR Current")
        self.ocr_btn.clicked.connect(self._ocr_current_image)
        self.ocr_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        ocr_btn_layout.addWidget(self.ocr_btn)
        
        self.batch_btn = QPushButton("🔄 Batch OCR All")
        self.batch_btn.clicked.connect(self._batch_ocr)
        self.batch_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        ocr_btn_layout.addWidget(self.batch_btn)
        
        left_layout.addLayout(ocr_btn_layout)
        
        # Toggle boxes button
        self.toggle_boxes_btn = QPushButton("👁 Toggle Boxes")
        self.toggle_boxes_btn.clicked.connect(self._toggle_boxes)
        self.toggle_boxes_btn.setCheckable(True)
        self.toggle_boxes_btn.setChecked(True)
        left_layout.addWidget(self.toggle_boxes_btn)

        # Contrast adjustment
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_spinbox = QDoubleSpinBox()
        self.contrast_spinbox.setRange(0.0, 2.0)
        self.contrast_spinbox.setValue(1.5)
        self.contrast_spinbox.setSingleStep(0.1)
        self.contrast_spinbox.setDecimals(1)
        self.contrast_spinbox.setToolTip("Contrast factor: 0 (gray) to 1 (original) to 2 (high contrast)")
        contrast_layout.addWidget(self.contrast_spinbox)
        left_layout.addLayout(contrast_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e0e0e0; border-radius: 3px;")
        left_layout.addWidget(self.status_label)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Image viewer and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        right_layout.addWidget(self.image_viewer, stretch=2)
        
        # Results panel
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(10, 10, 10, 10)
        
        results_layout.addWidget(QLabel("OCR Results:"))
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(150)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #444;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        results_layout.addWidget(self.results_text)
        
        right_layout.addWidget(results_frame, stretch=1)
        
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        
    def _init_ocr_engine(self):
        """Initialize the OCR engine with custom Korean model"""
        try:
            # Get paths
            base_dir = Path(__file__).parent
            model_path = base_dir / "savedir" / "model.onnx"
            keys_path = base_dir / "savedir" / "keys.txt"
            det_model_path = base_dir / "savedir" / "detection" / "det.onnx"
            
            if not model_path.exists():
                QMessageBox.critical(self, "Error", f"Model not found: {model_path}")
                return
                
            if not keys_path.exists():
                QMessageBox.critical(self, "Error", f"Keys file not found: {keys_path}")
                return

            if not det_model_path.exists():
                QMessageBox.critical(self, "Error", f"Detection model not found: {det_model_path}")
                return
            
            self.status_label.setText("Loading OCR engine...")
            QApplication.processEvents()
            
            # Initialize RapidOCR with custom model
            self.ocr_engine = RapidOCR(
                params={
                    'Det.model_path': str(det_model_path),
                    'Global.use_det': True,
                    'Rec.model_path': str(model_path),
                    'Rec.rec_keys_path': str(keys_path),
                    'Global.use_cls': True,
                    'Global.use_rec': True,
                }
            )
            
            self.status_label.setText(f"OCR Engine Ready | Det: {det_model_path.name} | Rec: {model_path.name}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to initialize OCR engine:\n{str(e)}")
            self.status_label.setText("OCR Engine Failed")
            
    def _load_images(self):
        """Load images from the images directory"""
        images_dir = Path(__file__).parent / "images"
        
        if not images_dir.exists():
            QMessageBox.warning(self, "Warning", f"Images directory not found: {images_dir}")
            return
            
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.image_files = sorted([
            str(f) for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        # Populate list
        self.image_list.clear()
        for img_path in self.image_files:
            item = QListWidgetItem(os.path.basename(img_path))
            item.setData(Qt.UserRole, img_path)
            self.image_list.addItem(item)
            
        if self.image_files:
            self.image_list.setCurrentRow(0)
            self.status_label.setText(f"Loaded {len(self.image_files)} images")
        else:
            self.status_label.setText("No images found")
            
    def _on_image_selected(self, index: int):
        """Handle image selection from list"""
        if index < 0 or index >= len(self.image_files):
            return
            
        self.current_image_index = index
        img_path = self.image_files[index]
        
        self.image_viewer.load_image(img_path)
        
        # Display cached results if available
        if img_path in self.ocr_results:
            results, elapsed = self.ocr_results[img_path]
            self.image_viewer.set_ocr_results(results)
            self._display_results(img_path, results, elapsed)
        else:
            self.results_text.clear()
            self.results_text.setText("No OCR results. Click 'OCR Current' to process.")
            self.image_viewer.set_ocr_results([])
            
        self._update_navigation_buttons()
        
    def _show_previous_image(self):
        """Show previous image"""
        if self.current_image_index > 0:
            self.image_list.setCurrentRow(self.current_image_index - 1)
            
    def _show_next_image(self):
        """Show next image"""
        if self.current_image_index < len(self.image_files) - 1:
            self.image_list.setCurrentRow(self.current_image_index + 1)
            
    def _update_navigation_buttons(self):
        """Update navigation button states"""
        self.prev_btn.setEnabled(self.current_image_index > 0)
        self.next_btn.setEnabled(self.current_image_index < len(self.image_files) - 1)
        
    def _ocr_current_image(self):
        """Run OCR on the current image"""
        if self.current_image_index < 0 or not self.ocr_engine:
            return
            
        img_path = self.image_files[self.current_image_index]
        self._run_ocr([img_path])
        
    def _batch_ocr(self):
        """Run OCR on all images"""
        if not self.image_files or not self.ocr_engine:
            return
            
        reply = QMessageBox.question(
            self, "Batch OCR",
            f"Process all {len(self.image_files)} images?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._run_ocr(self.image_files)
            
    def _run_ocr(self, image_paths: List[str]):
        """Run OCR on a list of images"""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "OCR is already running")
            return
            
        self.ocr_btn.setEnabled(False)
        self.batch_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(image_paths))
        self.progress_bar.setValue(0)
        
        contrast_factor = self.contrast_spinbox.value()
        self.worker = OCRWorker(self.ocr_engine, image_paths, draw_boxes=True, contrast_factor=contrast_factor)
        self.worker.progress.connect(self._on_ocr_progress)
        self.worker.result.connect(self._on_ocr_result)
        self.worker.finished_all.connect(self._on_ocr_finished)
        self.worker.error.connect(self._on_ocr_error)
        self.worker.start()
        
    def _on_ocr_progress(self, current: int, total: int, filename: str):
        """Handle OCR progress update"""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Processing {current}/{total}: {filename}")
        
    def _on_ocr_result(self, filename: str, results: List, elapsed: float):
        """Handle OCR result for one image"""
        self.ocr_results[filename] = (results, elapsed)
        
        # If this is the currently displayed image, update the view
        if self.current_image_index >= 0:
            current_path = self.image_files[self.current_image_index]
            if current_path == filename:
                self.image_viewer.set_ocr_results(results)
                self._display_results(filename, results, elapsed)
                
        # Update list item to show it's been processed
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == filename:
                item.setText(f"✓ {os.path.basename(filename)}")
                break
                
    def _on_ocr_finished(self):
        """Handle OCR completion"""
        self.ocr_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("OCR Complete")
        
        # Count results
        processed = len(self.ocr_results)
        self.status_label.setText(f"Processed {processed} images")
        
    def _on_ocr_error(self, error_msg: str):
        """Handle OCR error"""
        self.ocr_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "OCR Error", error_msg)
        self.status_label.setText("OCR Failed")
        
    def _display_results(self, filename: str, results: List, elapsed: float):
        """Display OCR results in text area"""
        text = f"File: {os.path.basename(filename)}\n"
        text += f"Time: {elapsed:.3f}s\n"
        text += f"Detections: {len(results)}\n"
        text += "=" * 50 + "\n\n"
        
        if results:
            for idx, res in enumerate(results, 1):
                if len(res) >= 2:
                    box = res[0]
                    detected_text = res[1]
                    confidence = res[2] if len(res) > 2 else 0.0
                    
                    text += f"[{idx}] Confidence: {confidence:.3f}\n"
                    text += f"    Text: {detected_text}\n"
                    if isinstance(box, (list, tuple)):
                        text += f"    Box: {box}\n"
                    text += "\n"
        else:
            text += "No text detected\n"
            
        self.results_text.setText(text)
        
    def _toggle_boxes(self):
        """Toggle bounding box display"""
        showing = self.image_viewer.toggle_boxes()
        self.toggle_boxes_btn.setChecked(showing)
        self.toggle_boxes_btn.setText("👁 Hide Boxes" if showing else "👁 Show Boxes")
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


def main():
    # Custom exception hook to print full tracebacks instead of swallowing errors
    def exception_hook(exctype, value, traceback_obj):
        import traceback
        traceback.print_exception(exctype, value, traceback_obj)
        sys.__excepthook__(exctype, value, traceback_obj)
        sys.exit(1)

    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = RapidOCRTestApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
