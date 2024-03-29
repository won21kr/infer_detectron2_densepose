from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion
from infer_detectron2_densepose.infer_detectron2_densepose_process import DensePoseParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class DensePoseWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = DensePoseParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # cuda parameter
        cuda_label = QLabel("CUDA")
        self.cuda_ckeck = QCheckBox()
        self.cuda_ckeck.setChecked(True)

        # proba parameter
        proba_label = QLabel("Threshold :")
       
        self.proba_spinbox = QDoubleSpinBox()
        self.proba_spinbox.setValue(0.8)
        self.proba_spinbox.setSingleStep(0.1)
        self.proba_spinbox.setMaximum(1)
        if self.parameters.proba != 0.8:
            self.proba_spinbox.setValue(self.parameters.proba)

        self.grid_layout.setColumnStretch(0, 0)
        self.grid_layout.addWidget(self.cuda_ckeck, 0, 0)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.addWidget(cuda_label, 0, 1)
        self.grid_layout.addWidget(proba_label, 1, 0)
        self.grid_layout.addWidget(self.proba_spinbox, 1, 1)
        self.grid_layout.setColumnStretch(2, 2)

        # Set widget layout
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.setLayout(layout_ptr)

        if not self.parameters.cuda:
            self.cuda_ckeck.setChecked(False)

    def onApply(self):
        # Apply button clicked slot
        if self.cuda_ckeck.isChecked():
            self.parameters.cuda = True
        else:
            self.parameters.cuda = False

        self.parameters.proba = self.proba_spinbox.value()
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class DensePoseWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_detectron2_densepose"

    def create(self, param):
        # Create widget object
        return DensePoseWidget(param, None)
