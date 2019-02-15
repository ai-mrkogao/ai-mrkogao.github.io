---
title: "QuantRobot"
date: 2019-02-15
classes: wide
use_math: true
tags: quant robot qt gui pyqt algorobot
category: python_api
---

## Algo Quant robot GUI Implementation

```python
import PyQt5
from PyQt5 import QtCore, QtGui, uic
from PyQt5 import QAxContainer
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QMainWindow, QDialog, QMessageBox, QProgressBar)
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *


Ui_MainWindow, QtBaseClass_MainWindow = uic.loadUiType(UI_DIR+"xxxx.ui")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("")
        ## plugins are robot class
        self.plugins = CPluginManager.plugin_loader() 

class CPluginManager:
    plugins = None
    @classmethod
    def plugin_loader(cls):
        ...
        for f in os.listdir(path):
            fname, ext = os.path.splitext(f)
            if ext == '.py':
                mod = __import__(fname)
                robot = mod.robot_loader()
                if robot != None:
                    result[robot.Name] = robot
        ...

        return result

if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    window = MainWindow()
    window.show()

    QTimer().singleShot(3, window.OnQApplicationStarted)    


## robot.py 
def robot_loader():
    UUID = uuid.uuid4().hex
    robot = Robot(Name=ROBOT_NAME, UUID=UUID)
    return robot


class CPortStock(object):
    ...

class CRobot(object):
    ...


class Robov1(CRobot):
    ...
    def instance(self):
        UUID = uuid.uuid4().hex
        return Robov1(Name=ROBOT_NAME, UUID=UUID)
    ...
    
```

## Pandas GUI
```python
df['컬럼'] = df['업종코드'] + " : " + df['업종명']
df = df.sort_values(['업종코드', '업종명'], ascending=[True, True])
```


## Xing API
```python
#Inblock,outblock

self.MYNAME = self.__class__.__name__
self.INBLOCK = "%sInBlock" % self.MYNAME

```

