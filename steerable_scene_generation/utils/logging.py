import logging


class ExcludeContainsFilter(logging.Filter):
    def __init__(self, string):
        super().__init__()
        self.string = string

    def filter(self, record):
        # Exclude messages starting with the specified prefix.
        return not self.string in record.getMessage()


def filter_drake_vtk_warning() -> None:
    exclude_filter = ExcludeContainsFilter(
        "In vtkGLTFDocumentLoaderInternals.cxx, line 1353: vtkGLTFDocumentLoader"
    )
    exclude_filter = ExcludeContainsFilter(
        "KHR_texture_basisu is used in this model, but not supported by this loader"
    )

    logger = logging.getLogger("drake")
    logger.addFilter(exclude_filter)


def filter_drake_obj_warning() -> None:
    exclude_filter = ExcludeContainsFilter("The OBJ file's material requested")
    logger = logging.getLogger("drake")
    logger.addFilter(exclude_filter)
