import redis

r = redis.Redis(host='redis', port=6379, db=0)

class RedisHash():
    def __init__(self, key: str):
        self.key = key
    def merge(self, new_values):
        r.hset(self.key, mapping=new_values)
    def get(self):
        return r.hget(self.key)
    def get_all(self):
        return r.hgetall(self.key)

class RedisList():
    def __init__(self, key: str):
        self.key = key
    def append(self, value):
        r.lpush(self.key, value)
    def list(self):
        return r.lrange(self.key, 0, -1)

class RedisGraph():
    def __init__(self, key: str):
        self.key = key
        self.nodes = RedisHash(key + ':nodes')
        self.edges = RedisHash(key + ':edges')
    def dict():
        return dict(
                nodes = self.nodes.get_all(),
                edges = self.edges.get_all(),
                )
