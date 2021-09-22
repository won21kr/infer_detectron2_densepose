from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_detectron2_densepose.infer_detectron2_densepose_process import DensePoseFactory
        # Instantiate process object
        return DensePoseFactory()

    def getWidgetFactory(self):
        from infer_detectron2_densepose.infer_detectron2_densepose_widget import DensePoseWidgetFactory
        # Instantiate associated widget object
        return DensePoseWidgetFactory()
