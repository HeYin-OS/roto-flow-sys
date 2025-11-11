import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QMainWindow, QGroupBox, QLabel, QPushButton, QSlider, QFrame
from PySide6.QtCore import QObject, Signal, Slot, QPoint

from windows.canvas_widget import CanvasWidget
from utils.edge_snapping import EdgeSnappingConfig, local_snapping


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # =================
        # member variables
        # =================

        # width of tool area in main window
        self.tool_space_expansion_size = 250

        self.canvas = CanvasWidget(self)

        # ===== frame controls =====

        # line box for frame control area
        self.frame_bound_box = QGroupBox(self)

        # number label for showing current frame number
        self.frame_number_label = QLabel(self)

        # button for frame backwards
        self.frame_backward_button = QPushButton(self)

        # button for frame forwards
        self.frame_forward_button = QPushButton(self)

        # slider for dragging frames
        self.frame_slider = QSlider(self)

        # button for frame 1 & 2 selection
        self.frame1_select_button = QPushButton(self)
        self.frame2_select_button = QPushButton(self)

        # ===== draw controls =====

        # line box for draw control area
        self.draw_bound_box = QGroupBox(self)

        # button for switching to insert
        self.switch_insert_button = QPushButton(self)

        # button for switching to move
        self.switch_move_button = QPushButton(self)

        # button for deleting recent draws
        self.delete_recent_button = QPushButton(self)

        # button for clearing current canvas
        self.clear_canvas_button = QPushButton(self)

        # button for clear all strokes on all frames
        self.all_clear_button = QPushButton(self)

        # ===== display controls =====

        # line box for display control area
        self.display_bound_box = QGroupBox(self)

        # button for switching visibility of direction line
        self.switch_direction_line_button = QPushButton(self)

        # button for switching visibility of end point
        self.switch_end_point_button = QPushButton(self)

        # ===== optimization controls =====

        # line box for optimization control area
        self.optimize_bound_box = QGroupBox(self)

        # button for conducting optimization
        self.optimize_button = QPushButton(self)

        # number label for showing frame 1 & 2
        self.frame1_label = QLabel(self)
        self.frame2_label = QLabel(self)

        # =================
        # init procedures
        # =================
        self.setWindowTitle("Main Sketch Window")
        self.setFixedSize(self.canvas.test_video.width + self.tool_space_expansion_size, self.canvas.test_video.height)
        self.moveToCenter()
        self.setMouseTracking(True)
        self.initComponentLayouts()
        self.initSignalAndSlots()

    # ==================================================
    # Slot Functions
    # ==================================================

    @Slot()
    def onPredictCurves(self):
        idx_start = int(self.frame1_label.text()) - 1
        idx_end = int(self.frame2_label.text()) - 1  # minus 1 for index
        print(f"\n[Prediction] Ready for prediction: frame {idx_start} to frame {idx_end}")

        for idx_frame in range(idx_start + 1, idx_end + 1):
            for curve_real in self.canvas.fitted_curves_on_each_frame[idx_frame - 1]:
                assert len(curve_real) > 0

                print(f"---- Frame {idx_frame}:")
                curve_real_list = []
                curve_predict = []
                curve_only_optical_flow = []

                for point_ready in curve_real:
                    x_real = point_ready.x()
                    y_real = point_ready.y()
                    curve_real_list.append([y_real, x_real])

                    x_predict = x_real + self.canvas.test_video.optical_flow_cache[idx_frame - 1, y_real, x_real, 0]
                    y_predict = y_real + self.canvas.test_video.optical_flow_cache[idx_frame - 1, y_real, x_real, 1]

                    curve_predict.append([int(y_predict), int(x_predict)])
                    curve_only_optical_flow.append(QPoint(int(x_predict), int(y_predict)))

                # TODO: edge snapping "curve_predict" here

                curve_predict_np_yx = np.array(curve_predict)

                curve_real_np_yx = np.array(curve_real_list)

                avg_dist = np.linalg.norm(curve_predict_np_yx - curve_real_np_yx, axis=1).mean()

                print(f"avg move distance by optical flow: {avg_dist} (std: 3.5)")

                if avg_dist > 3.5:
                    curve_predict_np_yx = curve_real_np_yx

                candidates_yx = self.canvas.test_video.candidate_kd_trees.query_batch(
                    i_group=idx_frame,
                    centers=curve_predict_np_yx,
                    radius=EdgeSnappingConfig.r_s
                )

                local_snapped_stroke_np_xy = local_snapping(stroke=curve_predict_np_yx,
                                                            image_tensor_rgb=self.canvas.test_video.tensor_format[idx_frame],
                                                            candidate_points=candidates_yx)

                fitted_stroke_xy = [QPoint(int(xy[0]), int(xy[1])) for xy in local_snapped_stroke_np_xy]

                self.canvas.original_curves_on_each_frame[idx_frame].append(curve_real)
                self.canvas.optical_flow_curves_on_each_frame[idx_frame].append(curve_only_optical_flow)
                self.canvas.fitted_curves_on_each_frame[idx_frame].append(fitted_stroke_xy)

        self.canvas.reDraw()

    # ===== frame controls =====
    @Slot()
    def onFrameBackward(self):
        if self.canvas.index_current_frame > 0:
            self.canvas.index_current_frame -= 1
            self.frame_number_label.setText(str(self.canvas.index_current_frame + 1))
            self.frame_slider.setValue(self.canvas.index_current_frame + 1)
            self.canvas.reDraw()

    @Slot()
    def onFrameForward(self):
        if self.canvas.index_current_frame < self.canvas.test_video.frame_num - 1:
            self.canvas.index_current_frame += 1
            self.frame_number_label.setText(str(self.canvas.index_current_frame + 1))
            self.frame_slider.setValue(self.canvas.index_current_frame + 1)
            self.canvas.reDraw()

    @Slot()
    def onFrameSliderChange(self):
        self.canvas.index_current_frame = self.frame_slider.value() - 1
        self.canvas.reDraw()
        self.frame_number_label.setText(f"{self.canvas.index_current_frame + 1}")

    @Slot()
    def onSelectFrameOne(self):
        self.frame1_label.setText(str(self.canvas.index_current_frame + 1))

    @Slot()
    def onSelectFrameTwo(self):
        self.frame2_label.setText(str(self.canvas.index_current_frame + 1))

    @Slot()
    def onWipeAllCanvases(self):
        self.canvas.index_current_frame = 0

        self.canvas.index_insert_on_each_frame = np.zeros(self.canvas.test_video.frame_num, dtype=int)

        self.canvas.index_drawing_order_on_each_frame = np.zeros(self.canvas.test_video.frame_num, dtype=int)

        self.canvas.fitted_curves_on_each_frame = [[] for _ in range(self.canvas.test_video.frame_num)]
        self.canvas.optical_flow_curves_on_each_frame = [[] for _ in range(self.canvas.test_video.frame_num)]
        self.canvas.original_curves_on_each_frame = [[] for _ in range(self.canvas.test_video.frame_num)]
        self.canvas.curve_temp = []

        self.canvas.weights_history = []

        self.canvas.reDraw()

    # TODO: implement other slot functions

    # ==================================================
    # Init Functions
    # ==================================================

    def initSignalAndSlots(self):
        self.frame_backward_button.clicked.connect(self.onFrameBackward)
        self.frame_slider.valueChanged.connect(self.onFrameSliderChange)
        self.frame_forward_button.clicked.connect(self.onFrameForward)
        self.optimize_button.clicked.connect(self.onPredictCurves)
        self.frame1_select_button.clicked.connect(self.onSelectFrameOne)
        self.frame2_select_button.clicked.connect(self.onSelectFrameTwo)
        self.all_clear_button.clicked.connect(self.onWipeAllCanvases)

    def initComponentLayouts(self):
        # ===== canvas area =====
        self.canvas.setGeometry(0, 0,
                                self.canvas.test_video.width, self.canvas.test_video.height)

        # ===== frame controls =====
        self.frame_bound_box.setTitle("Frame")
        self.frame_bound_box.setStyleSheet("QGroupBox {"
                                           "    color: black;"
                                           "    border: 1px solid black;"
                                           "    margin-top: 1.5ex;"
                                           "}"
                                           "QGroupBox::title {"
                                           "    subcontrol-origin: margin;"
                                           "    left: 7px;"
                                           "}")
        self.frame_bound_box.setGeometry(self.canvas.test_video.width + 10, 0,
                                         230, 100)

        self.frame_number_label.setText("1")
        self.frame_number_label.setGeometry(self.frame_bound_box.x() + 10, self.frame_bound_box.y() + 20,
                                            20, 30)

        self.frame_backward_button.setText("◀")
        self.frame_backward_button.setGeometry(self.frame_number_label.x() + self.frame_number_label.width() + 10,
                                               self.frame_number_label.y(),
                                               30, 30)

        self.frame_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setGeometry(self.frame_backward_button.x() + self.frame_backward_button.width() + 10,
                                      self.frame_number_label.y(),
                                      100, 30)
        self.frame_slider.setRange(1, self.canvas.test_video.frame_num)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.frame_forward_button.setText("▶")
        self.frame_forward_button.setGeometry(self.frame_slider.x() + self.frame_slider.width() + 10,
                                              self.frame_slider.y(),
                                              30, 30)

        self.frame1_select_button.setText("As 1st Frame")
        self.frame1_select_button.setGeometry(self.frame_number_label.x(),
                                              self.frame_number_label.y() + self.frame_number_label.height() + 10,
                                              100, 30)

        self.frame2_select_button.setText("As 2nd Frame")
        self.frame2_select_button.setGeometry(self.frame1_select_button.x() + self.frame1_select_button.width() + 10,
                                              self.frame1_select_button.y(),
                                              100, 30)

        # ===== draw controls =====
        self.draw_bound_box.setTitle("Edit")
        self.draw_bound_box.setStyleSheet("QGroupBox {"
                                          "    color: black;"
                                          "    border: 1px solid black;"
                                          "    margin-top: 1.5ex;"
                                          "}"
                                          "QGroupBox::title {"
                                          "    subcontrol-origin: margin;"
                                          "    left: 7px;"
                                          "}")
        self.draw_bound_box.setGeometry(self.frame_bound_box.x(), self.frame_bound_box.y() + self.frame_bound_box.height() + 10,
                                        230, 140)

        self.switch_insert_button.setText("Draw (On)")
        self.switch_insert_button.setGeometry(self.draw_bound_box.x() + 10, self.draw_bound_box.y() + 20,
                                              100, 30)

        self.switch_move_button.setText("Move (Off)")
        self.switch_move_button.setGeometry(self.switch_insert_button.x() + self.switch_insert_button.width() + 10,
                                            self.switch_insert_button.y(),
                                            100, 30)

        self.delete_recent_button.setText("Delete Recent")
        self.delete_recent_button.setGeometry(self.switch_insert_button.x(),
                                              self.switch_insert_button.y() + self.switch_insert_button.height() + 10,
                                              100, 30)

        self.clear_canvas_button.setText("Clear Canvas")
        self.clear_canvas_button.setGeometry(self.delete_recent_button.x() + self.delete_recent_button.width() + 10,
                                             self.delete_recent_button.y(),
                                             100, 30)

        self.all_clear_button.setText("Wipe All Canvases")
        self.all_clear_button.setGeometry(self.delete_recent_button.x(),
                                          self.delete_recent_button.y() + self.delete_recent_button.height() + 10,
                                          150, 30)

        # ===== display controls =====
        self.display_bound_box.setTitle("Display")
        self.display_bound_box.setStyleSheet("QGroupBox {"
                                             "    color: black;"
                                             "    border: 1px solid black;"
                                             "    margin-top: 1.5ex;"
                                             "}"
                                             "QGroupBox::title {"
                                             "    subcontrol-origin: margin;"
                                             "    left: 7px;"
                                             "}")
        self.display_bound_box.setGeometry(self.draw_bound_box.x(), self.draw_bound_box.y() + self.draw_bound_box.height() + 10,
                                           230, 100)

        self.switch_direction_line_button.setText("Toggle Direction Line(s)")
        self.switch_direction_line_button.setGeometry(self.display_bound_box.x() + 10, self.display_bound_box.y() + 20,
                                                      150, 30)

        self.switch_end_point_button.setText("Toggle End Point(s)")
        self.switch_end_point_button.setGeometry(self.switch_direction_line_button.x(),
                                                 self.switch_direction_line_button.y() + self.switch_direction_line_button.height() + 10,
                                                 150, 30)

        # ===== optimization controls =====
        self.optimize_bound_box.setTitle("Optimize")
        self.optimize_bound_box.setStyleSheet("QGroupBox {"
                                              "    color: green;"
                                              "    border: 1px solid green;"
                                              "    margin-top: 1.5ex;"
                                              "}"
                                              "QGroupBox::title {"
                                              "    subcontrol-origin: margin;"
                                              "    left: 7px;"
                                              "}")
        self.optimize_bound_box.setGeometry(self.display_bound_box.x(),
                                            self.display_bound_box.y() + self.display_bound_box.height() + 10,
                                            230, 60)

        self.optimize_button.setText("Optimize")
        self.optimize_button.setGeometry(self.optimize_bound_box.x() + 10,
                                         self.optimize_bound_box.y() + 20,
                                         100, 30)

        self.frame1_label.setText("1")
        self.frame1_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.frame1_label.setFrameShape(QFrame.Shape.Panel)
        self.frame1_label.setFrameShadow(QFrame.Shadow.Plain)
        self.frame1_label.setLineWidth(1)
        self.frame1_label.setGeometry(self.optimize_button.x() + self.optimize_button.width() + 10, self.optimize_button.y(),
                                      40, 30)

        self.frame2_label.setText(f"{self.canvas.test_video.frame_num}")
        self.frame2_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.frame2_label.setFrameShape(QFrame.Shape.Panel)
        self.frame2_label.setFrameShadow(QFrame.Shadow.Plain)
        self.frame2_label.setLineWidth(1)
        self.frame2_label.setGeometry(self.frame1_label.x() + self.frame1_label.width() + 20, self.frame1_label.y(),
                                      40, 30)

    # move this window to the center of screen
    def moveToCenter(self):
        qr = self.geometry()
        center_point = self.screen().availableGeometry().center()
        qr.moveCenter(center_point)
        self.move(qr.topLeft())
