"""Run fetcher and prioritizer."""
import asyncio
import logging
import logging.config

import uvloop
import yaml

from strider.fetcher import Fetcher
# from strider.prioritizer import Prioritizer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


async def start():
    """Start fetcher and prioritizer."""
    fetcher = Fetcher()
    await fetcher.setup()
    await fetcher.run()

    # prioritizer = Prioritizer()
    # await prioritizer.setup()
    # await prioritizer.run()

    print('Ready.')


if __name__ == "__main__":
    with open('logging_setup.yml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    logging.config.dictConfig(config)

    # start event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start())
    loop.run_forever()
