from pwcp import PreprocessorHooks, PycType

from . import config
from .parser import to_pure_python
from .version import __version__


class YabiHooks(PreprocessorHooks):
    def __init__(self):
        super().__init__("yabi")

    def process_source(
        self, source: str, filename: str, state: None
    ) -> tuple[str, None]:
        return to_pure_python(source), None

    def create_pyc_data(self, data: None, pyc_type: PycType) -> dict:
        return {
            "version": __version__,
            "preprocessing": config.ENABLE_PREPROCESSING,
        }

    def validate_pyc_data(self, pyc: dict, pyc_type: PycType) -> bool:
        return (
            pyc["version"] == __version__
            and pyc["preprocessing"] == config.ENABLE_PREPROCESSING
        )
