import tempfile

from fastapi.staticfiles import StaticFiles
from pyinstrument import Profiler
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from .server import APP
from .config import settings


class DownloadStaticFiles(StaticFiles):
    """
    Wrapper on StaticFiles to serve files that are automatically
    downloaded using the Content-Disposition header
    """

    def file_response(*args, **kwargs):
        resp = StaticFiles.file_response(*args, **kwargs)
        resp.headers["Content-Disposition"] = "attachment"
        return resp


# Create directory to store request profile data
PROFILE_DIRECTORY = tempfile.mkdtemp()

captured_profiles = []


# Middleware to capture profile data and save it
# to files
@APP.middleware("http")
async def profiler_middleware(request, call_next):
    if settings.profiler:
        # Run the request
        profiler = Profiler(async_mode="enabled")
        profiler.start()
        call_result = await call_next(request)
        profiler.stop()
        profiler.open_in_browser()
        with open("profiles/profile.speedscope.json", "w") as f:
            f.write(profiler.output(renderer=SpeedscopeRenderer()))
        return call_result


# Serve prof files under /profiles/
APP.mount(
    "/profiles",
    DownloadStaticFiles(directory=PROFILE_DIRECTORY),
    name="profiles",
)


@APP.get("/profiles")
async def profiles_list():
    """Get all profiles captured in the current session"""
    return captured_profiles
