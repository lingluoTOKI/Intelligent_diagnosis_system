    def init_ui(self):
        # 主布局 - 使用QSplitter实现可调整的两部分布局
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        main_splitter.setStretchFactor(0, 6)  # 左侧视觉区占比 6
        main_splitter.setStretchFactor(1, 4)  # 右侧分析区占比 4
        main_splitter.setMinimumSize(1000, 600)

        # ========================================================
        # 左侧容器 - 视觉与操作区
        # ========================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(20, 20, 10, 20)

        # 标题
        title_label = QLabel("AI 眼科疾病智诊系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.accent_color}; letter-spacing: 3px; margin-bottom: 10px;")
        left_layout.addWidget(title_label)

        # --- 1. 图像展示区 (采用 QTabWidget 节省空间) ---
        self.image_tab_widget = QTabWidget()
        self.image_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid #3b4252; border-radius: 8px; background-color: {self.secondary_bg}; }}
            QTabBar::tab {{ background-color: {self.primary_color}; color: #81A1C1; padding: 10px 25px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; font-size: 14px; margin-right: 2px; }}
            QTabBar::tab:selected {{ background-color: {self.accent_color}; color: white; }}
        """)

        # 标签页 A：本地图像分析
        local_tab = QWidget()
        local_layout = QHBoxLayout(local_tab)
        local_layout.setSpacing(15)

        # 原始图像卡片
        self.original_image_label = QLabel("等待加载图像...")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 300)
        self.original_image_label.setMaximumSize(600, 600)
        self.original_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_image_label.setScaledContents(False)
        self.original_image_label.setStyleSheet(f"background-color: #1a1e24; border-radius: 6px; border: 1px solid #2c323c;")
        local_layout.addWidget(self.original_image_label)

        # 检测结果卡片
        self.detected_image_label = QLabel("等待检测结果...")
        self.detected_image_label.setAlignment(Qt.AlignCenter)
        self.detected_image_label.setMinimumSize(300, 300)
        self.detected_image_label.setMaximumSize(600, 600)
        self.detected_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.detected_image_label.setScaledContents(False)
        self.detected_image_label.setStyleSheet(f"background-color: #1a1e24; border-radius: 6px; border: 1px solid {self.highlight_color};")
        local_layout.addWidget(self.detected_image_label)
        self.image_tab_widget.addTab(local_tab, "🖼️ 本地图像分析")

        # 标签页 B：开发板流媒体
        board_tab = QWidget()
        board_layout = QVBoxLayout(board_tab)

        self.camera_preview_label = QLabel("摄像头未连接\n点击连接开始预览")
        self.camera_preview_label.setAlignment(Qt.AlignCenter)
        self.camera_preview_label.setMinimumSize(200, 150)
        self.camera_preview_label.setMaximumSize(350, 250)
        self.camera_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_preview_label.setScaledContents(False)
        self.camera_preview_label.setStyleSheet("background-color: #1a1e24; border-radius: 6px; color: #616E88;")
        board_layout.addWidget(self.camera_preview_label)

        # 摄像头控制条
        cam_ctrl_layout = QHBoxLayout()
        self.board_camera_status = QLabel("🔴 未连接")
        self.board_camera_status.setStyleSheet("font-weight: bold; color: #E53E3E;")
        self.connect_camera_button = QPushButton("🔗 连接开发板")
        self.capture_from_camera_button = QPushButton("📸 截取并诊断")
        self.capture_from_camera_button.setEnabled(False)

        cam_btn_style = f"QPushButton {{ background-color: {self.success_color}; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold; }} QPushButton:hover {{ background-color: #2F855A; }} QPushButton:disabled {{ background-color: #4A5568; color: #A0AEC0; }}"
        for btn in [self.connect_camera_button, self.capture_from_camera_button]:
            btn.setStyleSheet(cam_btn_style)
            btn.setCursor(QCursor(Qt.PointingHandCursor))

        self.connect_camera_button.clicked.connect(self.toggle_camera_connection)
        self.capture_from_camera_button.clicked.connect(self.capture_from_board_camera)

        cam_ctrl_layout.addWidget(self.board_camera_status)
        cam_ctrl_layout.addStretch()
        cam_ctrl_layout.addWidget(self.connect_camera_button)
        cam_ctrl_layout.addWidget(self.capture_from_camera_button)
        board_layout.addLayout(cam_ctrl_layout)
        self.image_tab_widget.addTab(board_tab, "📱 硬件实时视窗")

        left_layout.addWidget(self.image_tab_widget)

        # --- 2. 按钮控制面板 (逻辑分组) ---
        btn_panel = QWidget()
        btn_panel.setStyleSheet(f"background-color: {self.secondary_bg}; border-radius: 8px; border-top: 3px solid {self.accent_color};")
        btn_layout_v = QVBoxLayout(btn_panel)
        btn_layout_v.setContentsMargins(15, 15, 15, 15)
        btn_layout_v.setSpacing(15)

        # 核心工作流按钮
        main_flow_layout = QHBoxLayout()
        main_flow_layout.setSpacing(10)
        self.model_button = QPushButton("1. 🔁 加载模型")
        self.image_button = QPushButton("2. 🖼️ 加载图像")
        self.detect_button = QPushButton("3. 🔍 开始检测")
        self.results_button = QPushButton("4. 📊 查看报告")

        main_btn_style = f"QPushButton {{ background-color: {self.accent_color}; color: white; padding: 12px; border-radius: 6px; font-weight: bold; font-size: 13px; }} QPushButton:hover {{ background-color: #0097B2; }} QPushButton:disabled {{ background-color: #3b4252; color: #7b88a1; }}"
        for btn in [self.model_button, self.image_button, self.detect_button, self.results_button]:
            btn.setStyleSheet(main_btn_style)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setMinimumHeight(45)
            main_flow_layout.addWidget(btn)

        self.model_button.clicked.connect(lambda: self.load_model(None))
        self.image_button.clicked.connect(self.load_image)
        self.image_button.setEnabled(False)
        self.detect_button.clicked.connect(self.detect_image)
        self.detect_button.setEnabled(False)
        self.results_button.clicked.connect(self.show_results)
        self.results_button.setEnabled(False)

        # 扩展工具按钮
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(10)
        self.batch_button = QPushButton("📁 批量处理")
        self.history_button = QPushButton("📜 历史记录")
        self.board_interaction_button = QPushButton("📱 开发板交互")
        self.voice_server_button = QPushButton("🎤 语音服务")

        tool_style = f"QPushButton {{ background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px; border-radius: 4px; }} QPushButton:hover {{ background-color: #4C566A; }}"
        for btn in [self.batch_button, self.history_button, self.board_interaction_button, self.voice_server_button]:
            btn.setStyleSheet(tool_style)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            tools_layout.addWidget(btn)

        self.batch_button.clicked.connect(self.batch_process)
        self.batch_button.setEnabled(False)
        self.history_button.clicked.connect(self.show_history)
        self.board_interaction_button.clicked.connect(self.show_board_interaction)
        self.voice_server_button.clicked.connect(self.toggle_voice_server)

        btn_layout_v.addLayout(main_flow_layout)
        btn_layout_v.addLayout(tools_layout)
        left_layout.addWidget(btn_panel)

        # ========================================================
        # 右侧容器 - AI 分析与设置区 (同样采用 Tab 化)
        # ========================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 20, 20, 20)

        self.right_tab_widget = QTabWidget()
        self.right_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid #3b4252; border-radius: 8px; background-color: {self.secondary_bg}; }}
            QTabBar::tab {{ background-color: {self.primary_color}; color: #81A1C1; padding: 12px 20px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }}
            QTabBar::tab:selected {{ background-color: {self.highlight_color}; color: white; }}
        """)

        # --- Tab 1: AI 诊疗建议 (主视图) ---
        advice_tab = QWidget()
        advice_layout = QVBoxLayout(advice_tab)

        # 全屏/操作工具栏
        advice_tool_layout = QHBoxLayout()
        self.advice_button = QPushButton("🔄 生成当前报告")
        self.advice_button.setStyleSheet(f"QPushButton {{ background-color: {self.highlight_color}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; }} QPushButton:disabled {{ background-color: #3b4252; }}")
        self.advice_button.clicked.connect(self.show_ai_advice)
        self.advice_button.setEnabled(False)
        self.advice_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.fullscreen_advice_btn = QPushButton("🗖 全屏阅览")
        self.fullscreen_advice_btn.setStyleSheet(f"QPushButton {{ background-color: transparent; border: 1px solid {self.highlight_color}; color: {self.highlight_color}; padding: 6px 12px; border-radius: 4px; }} QPushButton:hover {{ background-color: rgba(128, 90, 213, 0.2); }}")
        self.fullscreen_advice_btn.clicked.connect(self.show_fullscreen_advice)
        self.fullscreen_advice_btn.setCursor(QCursor(Qt.PointingHandCursor))

        advice_tool_layout.addWidget(self.advice_button)
        advice_tool_layout.addStretch()
        advice_tool_layout.addWidget(self.fullscreen_advice_btn)

        # 建议展示文本框
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setMinimumHeight(300)
        self.advice_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.advice_text.setFont(QFont("Microsoft YaHei", 14))
        self.advice_text.setStyleSheet(f"background-color: {self.primary_color}; border: none; padding: 10px; color: {self.text_color}; font-size: 14px;")
        self.advice_text.setHtml(f"<div style='text-align:center; margin-top:50px;'><h2 style='color:{self.highlight_color};'>DeepSeek 诊疗引擎</h2><p style='color:#616E88;'>请先进行疾病检测, 然后点击生成报告获取专业建议.</p></div>")

        advice_layout.addLayout(advice_tool_layout)
        advice_layout.addWidget(self.advice_text)
        self.right_tab_widget.addTab(advice_tab, "🩺 诊疗建议")

        # --- Tab 2: 自定义对话 ---
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setSpacing(10)

        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("在此描述其他症状或向AI提问...")
        self.chat_input.setMaximumHeight(100)
        self.chat_input.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; border-radius: 4px; padding: 10px;")

        voice_ctrl_layout = QHBoxLayout()
        self.voice_chat_enabled = QCheckBox("启用语音回复")
        self.voice_chat_enabled.setStyleSheet(f"color: {self.text_color};")
        self.voice_chat_enabled.stateChanged.connect(self.toggle_voice_chat)

        self.voice_input_button = QPushButton("🎤 按住说话")
        self.voice_input_button.setStyleSheet(f"background-color: {self.accent_color}; color: white; padding: 6px 15px; border-radius: 15px;")
        self.voice_input_button.clicked.connect(self.start_voice_input)
        self.voice_input_button.setEnabled(False)
        self.voice_input_button.setCursor(QCursor(Qt.PointingHandCursor))

        voice_ctrl_layout.addWidget(self.voice_chat_enabled)
        voice_ctrl_layout.addWidget(self.voice_input_button)
        voice_ctrl_layout.addStretch()

        chat_btns_layout = QHBoxLayout()
        self.send_chat_button = QPushButton("发送提问")
        self.send_chat_button.setStyleSheet(f"background-color: {self.highlight_color}; color: white; padding: 8px 20px; border-radius: 4px; font-weight: bold;")
        self.send_chat_button.clicked.connect(self.send_chat_message)
        self.send_chat_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.clear_chat_button = QPushButton("清除记录")
        self.clear_chat_button.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px 20px; border-radius: 4px;")
        self.clear_chat_button.clicked.connect(self.clear_chat_history)
        self.clear_chat_button.setCursor(QCursor(Qt.PointingHandCursor))

        chat_btns_layout.addStretch()
        chat_btns_layout.addWidget(self.clear_chat_button)
        chat_btns_layout.addWidget(self.send_chat_button)

        self.ai_progress_bar = QProgressBar()
        self.ai_progress_bar.setVisible(False)

        chat_layout.addWidget(self.chat_input)
        chat_layout.addLayout(voice_ctrl_layout)
        chat_layout.addWidget(self.ai_progress_bar)
        chat_layout.addLayout(chat_btns_layout)
        chat_layout.addStretch()
        self.right_tab_widget.addTab(chat_tab, "💬 医疗问答")

        # --- Tab 3: 系统设置 (收纳低频配置) ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setSpacing(15)

        group_style = f"QGroupBox {{ border: 1px solid #4C566A; border-radius: 6px; margin-top: 15px; padding: 15px; }} QGroupBox::title {{ color: {self.accent_color}; top: -10px; left: 10px; }}"

        api_group = QGroupBox("DeepSeek 引擎配置")
        api_group.setStyleSheet(group_style)
        api_layout_v = QVBoxLayout(api_group)

        self.use_api_checkbox = QCheckBox("启用云端 AI 推理")
        self.use_api_checkbox.setChecked(True)
        self.use_api_checkbox.setStyleSheet(f"color: {self.text_color}; font-weight: bold;")
        self.use_api_checkbox.toggled.connect(self.toggle_api_usage)

        key_input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("输入 DeepSeek API Key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; padding: 8px; border-radius: 4px; color: {self.text_color};")

        self.toggle_password_button = QPushButton("👁")
        self.toggle_password_button.setFixedWidth(35)
        self.toggle_password_button.clicked.connect(self.toggle_password_visibility)

        self.save_api_key_button = QPushButton("保存配置")
        self.save_api_key_button.setStyleSheet(f"background-color: {self.success_color}; color: white; padding: 8px 15px; border-radius: 4px;")
        self.save_api_key_button.clicked.connect(self.save_api_key)
        self.save_api_key_button.setCursor(QCursor(Qt.PointingHandCursor))

        key_input_layout.addWidget(self.api_key_input)
        key_input_layout.addWidget(self.toggle_password_button)
        key_input_layout.addWidget(self.save_api_key_button)

        self.network_test_button = QPushButton("测试 API 连通性")
        self.network_test_button.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px 15px; border-radius: 4px;")
        self.network_test_button.clicked.connect(self.test_network_and_show_result)
        self.network_test_button.setCursor(QCursor(Qt.PointingHandCursor))

        api_layout_v.addWidget(self.use_api_checkbox)
        api_layout_v.addLayout(key_input_layout)
        api_layout_v.addWidget(self.network_test_button)

        # 语音辅助配置
        mic_group = QGroupBox("外设测试")
        mic_group.setStyleSheet(group_style)
        mic_layout = QVBoxLayout(mic_group)
        self.mic_test_button = QPushButton("测试麦克风")
        self.mic_test_button.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px 15px; border-radius: 4px;")
        self.mic_test_button.clicked.connect(self.test_microphone)
        self.mic_test_button.setCursor(QCursor(Qt.PointingHandCursor))
        mic_layout.addWidget(self.mic_test_button)

        # 语音时长滑块
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(3, 30)
        self.duration_slider.setValue(10)
        self.duration_value_label = QLabel("10秒")
        self.duration_value_label.setStyleSheet(f"color: {self.text_color};")
        self.duration_slider.valueChanged.connect(self.update_duration_value)

        dur_layout = QHBoxLayout()
        dur_label = QLabel("语音最长识别时间:")
        dur_label.setStyleSheet(f"color: {self.text_color};")
        dur_layout.addWidget(dur_label)
        dur_layout.addWidget(self.duration_slider)
        dur_layout.addWidget(self.duration_value_label)
        mic_layout.addLayout(dur_layout)

        settings_layout.addWidget(api_group)
        settings_layout.addWidget(mic_group)
        settings_layout.addStretch()
        self.right_tab_widget.addTab(settings_tab, "⚙️ 系统设置")

        right_layout.addWidget(self.right_tab_widget)

        # 添加到主分离器
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 700])  # 调整初始大小比例

        # 语音管理器将在延迟加载中初始化, 这里不做任何操作
        # 避免阻塞主线程的Vosk模型加载
