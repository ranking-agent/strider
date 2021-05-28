from datetime import timedelta
from typing import Optional

from pydantic import \
    BaseSettings, FilePath, RedisDsn, AnyUrl


class Settings(BaseSettings):
    openapi_server_url: Optional[AnyUrl]
    kpregistry_url: AnyUrl = "http://kp-registry.renci.org"
    omnicorp_url: AnyUrl = "http://robokop.renci.org:3210"
    biolink_url: AnyUrl = "https://bl-lookup-sri.renci.org"
    normalizer_url: AnyUrl = "https://nodenormalization-sri.renci.org"

    redis_url: RedisDsn = "redis://localhost"
    store_results_for: timedelta = timedelta(days=7)

    class Config:
        env_file = ".env"


settings = Settings()
