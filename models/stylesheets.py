MAIN_STYLE = """
 QWidget {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        QTextEdit {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            selection-background-color: #3a3a3a;
        }
        QPushButton {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #303030;
        }
        QPushButton:checkable {
            border: 2px solid #555555;
        }
        QPushButton:checked {
            background-color: #2d5a2d;
            border: 2px solid #4a9e4a;
            color: #ffffff;
            font-weight: bold;
        }
        QPushButton:checked:hover {
            background-color: #346a34;
        }
        QPushButton:!checked {
            background-color: #404040;
            border: 2px solid #555555;
        }
        QPushButton:!checked:hover {
            background-color: #505050;
        }
        QListWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            alternate-background-color: #252525;
        }
        QListWidget::item {
            padding: 3px;
            color: #ffffff;
        }
        QListWidget::item:selected {
            background-color: #3a3a3a;
            color: #ffffff;
        }
        QListWidget::item:hover {
            color: #ffffff;
        }
        QTreeWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            alternate-background-color: #252525;
        }
        QTreeWidget::item {
            padding: 3px;
            color: #ffffff;
        }
        QTreeWidget::item:selected {
            background-color: #3a3a3a;
            color: #ffffff;
        }
        QTreeWidget::item:hover {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        QTreeWidget::branch {
            background-color: #1e1e1e;
        }
        QTreeWidget::branch:has-siblings:!adjoins-item {
            border-image: none;
        }
        QTreeWidget::branch:has-siblings:adjoins-item {
            border-image: none;
        }
        QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
            border-image: none;
        }
        QTreeWidget::branch:closed:has-children:has-siblings,
        QTreeWidget::branch:has-children:!has-siblings:closed {
            background: #1e1e1e;
            border-image: none;
        }
        QTreeWidget::branch:open:has-children:has-siblings,
        QTreeWidget::branch:open:has-children:!has-siblings {
            background: #1e1e1e;
            border-image: none;
        }
        QTreeWidget::indicator {
            width: 13px;
            height: 13px;
        }
        QTreeWidget::indicator:unchecked {
            background-color: #2a2a2a;
            border: 1px solid #555555;
        }
        QTreeWidget::indicator:checked {
            background-color: #4a9eff;
            border: 1px solid #3a8eef;
            image: none;
        }
        QTreeWidget::indicator:checked::after {
            content: "✓";
        }
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #cccccc;
        }
        QProgressBar {
            border: 1px solid #555555;
            background-color: #1e1e1e;
            text-align: center;
            color: #ffffff;
        }
        QProgressBar::chunk {
            background-color: #505050;
        }
        QLabel {
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #2a2a2a;
        }
        QTabBar::tab {
            background-color: #303030;
            color: #ffffff;
            border: 1px solid #555555;
            border-bottom: none;
            padding: 6px 12px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #404040;
            border-bottom: 1px solid #404040;
        }
        QTabBar::tab:hover {
            background-color: #3a3a3a;
        }
        QTabBar::close-button {
            image: url(none);
            subcontrol-position: right;
        }
        QTabBar::close-button:hover {
            background-color: #555555;
        }
        QComboBox {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
            min-height: 20px;
        }
        QComboBox:hover {
            background-color: #505050;
        }
        QComboBox:on {
            background-color: #303030;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
            margin-right: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #2a2a2a;
            color: #ffffff;
            selection-background-color: #3a3a3a;
            border: 1px solid #555555;
        }
        QSplitter::handle {
            background-color: #555555;
            height: 3px;
        }
        QSplitter::handle:hover {
            background-color: #777777;
        }
        QSplitter::handle:pressed {
            background-color: #999999;
        }
        QSpinBox {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
        }
        QSpinBox:hover {
            background-color: #505050;
        }
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: #505050;
            border: 1px solid #555555;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background-color: #606060;
        }
"""

PROMPT_DIALOG = """color: #888888; font-style: italic;"""


SEARCH_BAR = """  SearchWidget {
                background-color: palette(window);
                border: 1px solid palette(mid);
                border-radius: 3px;
            }"""

EDIT_ACCEPT_BUTTON = """background-color: #28a745; color: white; padding: 5px 15px;"""
EDIT_REJECT_BUTTON = """background-color: #dc3545; color: white; padding: 5px 15px;"""
ACCEPT_REJECT_BOX = """background-color: #3c3c3c; border-radius: 3px;"""
PLANNING_MODE = (
    """background-color: #2a4a2a; color: white; padding: 5px; font-weight: bold;"""
)
SUPPLEMENTAL_PROMPTS = """
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #094771;
            }"""

SYSTEM_PROMPTS = """
            QMenu {
                background-color: #0b2b2b;
                color: #ffffff;
                border: -1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #094769;
            }"""
