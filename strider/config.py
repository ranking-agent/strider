from typing import Optional

from pydantic import BaseSettings, AnyUrl


class Settings(BaseSettings):
    openapi_server_url: Optional[AnyUrl]
    openapi_server_maturity: str = "development"
    openapi_server_location: str = "RENCI"
    kp_trapi_version: str = "1.4.0"
    omnicorp_url: AnyUrl = "http://robokop.renci.org:3210"
    biolink_url: AnyUrl = "https://bl-lookup-sri.renci.org"
    normalizer_url: AnyUrl = "https://nodenormalization-sri.renci.org"
    max_process_time: int = 3400
    kp_timeout: int = 600
    information_content_threshold: int = 75
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_expiration: int = 1209600  # two weeks
    redis_password: str = "supersecretpassword"

    jaeger_enabled: str = "False"
    jaeger_host: str = "jaeger"
    jaeger_port: int = 6831

    profiler: bool = False
    use_cache: bool = True
    offline_mode: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
