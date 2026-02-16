"""
Story Builder Application - Refactored MVC Version

Main entry point for the application.
Run this file to start the story builder application.
"""

import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class SplashScreen(QtWidgets.QSplashScreen):
    """Loading splash screen to show progress during imports."""

    def __init__(self):
        # Create a simple pixmap for the splash screen
        pixmap = QtGui.QPixmap(500, 200)
        pixmap.fill(QtGui.QColor(45, 45, 45))

        super().__init__(pixmap)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
        )

        # Draw text on the pixmap
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor(255, 255, 255))

        # Title
        title_font = QtGui.QFont("Arial", 18, QtGui.QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "Story Builder")

        # Loading message
        msg_font = QtGui.QFont("Arial", 10)
        painter.setFont(msg_font)
        rect = pixmap.rect()
        rect.setTop(rect.top() + 60)
        painter.drawText(rect, QtCore.Qt.AlignCenter, "Initializing application...")

        painter.end()

        self.setPixmap(pixmap)
        self.show()
        QtWidgets.QApplication.processEvents()

    def update_message(self, message):
        """Update the loading message."""
        pixmap = self.pixmap()
        painter = QtGui.QPainter(pixmap)

        # Clear previous message area
        painter.fillRect(0, 80, 500, 120, QtGui.QColor(45, 45, 45))

        # Draw new message
        painter.setPen(QtGui.QColor(200, 200, 200))
        msg_font = QtGui.QFont("Arial", 9)
        painter.setFont(msg_font)

        rect = pixmap.rect()
        rect.setTop(rect.top() + 80)
        painter.drawText(rect, QtCore.Qt.AlignCenter, message)

        painter.end()

        self.setPixmap(pixmap)
        self.repaint()
        QtWidgets.QApplication.processEvents()


def main_with_splash():
    """Main entry point with splash screen."""
    app = QtWidgets.QApplication(sys.argv)

    # Register QTextCursor for queued signal/slot connections
    try:
        QtCore.qRegisterMetaType(QtGui.QTextCursor)
    except Exception:
        pass

    # Show splash screen
    splash = SplashScreen()

    # Import heavy dependencies with progress updates
    try:
        splash.update_message(
            "Loading language models...\n(This may take a moment on first run)"
        )
        from controllers.main_controller import main

        splash.update_message("Starting application...")

        # Close splash and start main application
        splash.finish(None)

        # Run the main application
        main()

    except Exception as e:
        splash.close()
        QtWidgets.QMessageBox.critical(
            None, "Startup Error", f"Failed to start application:\n\n{str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main_with_splash()
