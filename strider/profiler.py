from pathlib import Path
import time

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import yappi

from .server import APP

# Create directory to store request profile data
PROFILE_DIRECTORY = "/profiles"
Path(PROFILE_DIRECTORY).mkdir(exist_ok = True)

# Middleware to capture profile data and save it
# to files
@APP.middleware("http")
async def profiler_middleware(request, call_next):
    # Run the request
    with yappi.run():
        call_result = await call_next(request)

    # Save profile
    stats = yappi.get_func_stats()
    stats.save(f"{PROFILE_DIRECTORY}/req-{time.time()}.prof", type='pstat')

    # Continue
    return call_result

# Serve prof files under /profiles/
APP.mount("/profiles",
          StaticFiles(directory=PROFILE_DIRECTORY),
          name="profiles",
          )

@APP.get("/profiles", response_class=HTMLResponse)
async def profiles_list():
    """ Helper page that lists available profiles to download """

    profile_links = [
        f"<li><a href='/profiles/{p.name}' download>{p.stem}</a></li>"
        for p in Path(PROFILE_DIRECTORY).iterdir()
    ]

    return f"""
    <html>
        <head>
            <title>Strider Profiles</title>
        </head>
        <body>
            <h1>Available request profiles for download:</h1>
            <ul>
                {"".join(profile_links)}
            </ul>
        </body>
    </html>
    """
