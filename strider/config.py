from typing import Optional

from pydantic import BaseSettings, AnyUrl


class Settings(BaseSettings):
    openapi_server_url: Optional[AnyUrl]
    openapi_server_maturity: str = "development"
    openapi_server_location: str = "RENCI"
    kpregistry_url: AnyUrl = "https://kp-registry.renci.org"
    omnicorp_url: AnyUrl = "http://robokop.renci.org:3210"
    biolink_url: AnyUrl = "https://bl-lookup-sri.renci.org"
    normalizer_url: AnyUrl = "https://nodenormalization-sri.renci.org"
    max_process_time: int = 3400

    profiler: bool = False
    use_cache: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
