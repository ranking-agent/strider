"""Run fetcher and prioritizer."""
import asyncio
import logging
import logging.config

import uvloop
import yaml

from strider.fetcher import Fetcher
from strider.util import setup_logging

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


async def start():
    """Start fetcher and prioritizer."""
    fetcher = Fetcher(max_jobs=5)
    await fetcher.setup()
    await fetcher.run()

    # prioritizer = Prioritizer()
    # await prioritizer.setup()
    # await prioritizer.run()

    print('Ready.')


def main():
    """Run workers."""
    setup_logging()

    # start event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
    loop.run_forever()


if __name__ == "__main__":
    main()
