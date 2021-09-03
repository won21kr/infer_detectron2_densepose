from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_DensePose(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from Detectron2_DensePose.Detectron2_DensePose_process import Detectron2_DensePoseProcessFactory
        # Instantiate process object
        return Detectron2_DensePoseProcessFactory()

    def getWidgetFactory(self):
        from Detectron2_DensePose.Detectron2_DensePose_widget import Detectron2_DensePoseWidgetFactory
        # Instantiate associated widget object
        return Detectron2_DensePoseWidgetFactory()
