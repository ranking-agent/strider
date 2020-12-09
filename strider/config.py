from datetime import timedelta
from pydantic import \
    BaseSettings, FilePath, RedisDsn, AnyUrl


class Settings(BaseSettings):
    kpregistry_url: AnyUrl = "http://kp-registry:4983"
    omnicorp_url: AnyUrl = "http://robokop.renci.org:3210"
    biolink_url: AnyUrl = "https://bl-lookup-sri.renci.org"
    normalizer_url: AnyUrl = "https://nodenormalization-sri.renci.org"

    redis_url: RedisDsn = "redis://redis"
    prefixes_path: FilePath = "strider/prefixes.json"
    store_results_for: timedelta = timedelta(days=7)


settings = Settings()
