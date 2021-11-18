from pathlib import Path
import tempfile
import time
import uuid

from fastapi.staticfiles import StaticFiles
import yappi

from .server import APP


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
    # Run the request
    with yappi.run():
        call_result = await call_next(request)

    # Generate ID for this request
    request_id = str(uuid.uuid1())

    # Save profile
    stats = yappi.get_func_stats()
    stats.save(f"{PROFILE_DIRECTORY}/{request_id}.prof", type="pstat")

    download_link = (
        f"{request.url.scheme}://{request.url.netloc}/profiles/{request_id}.prof"
    )

    # Save profile meta-info
    captured_profiles.append(
        {
            "id": request_id,
            "timestamp": time.time(),
            "path": request.url.path,
            "download_link": download_link,
        }
    )

    # Continue
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
