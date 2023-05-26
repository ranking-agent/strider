"""Mock Redis."""
import fakeredis.aioredis as fakeredis
import gzip
import json


default_kps = {
    "kp0": {
        "url": "http://kp0/query",
        "infores": "infores:kp0",
        "maturity": "development",
        "operations": [
            {
                "subject_category": "biolink:ChemicalSubstance",
                "predicate": "biolink:treats",
                "object_category": "biolink:Disease",
            },
        ],
        "details": {
            "preferred_prefixes": {
                "biolink:Disease": ["MONDO", "DOID"],
                "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
                "biolink:PhenotypicFeature": ["HP"],
            },
        },
    },
    "kp1": {
        "url": "http://kp1/query",
        "infores": "infores:kp1",
        "maturity": "development",
        "operations": [
            {
                "subject_category": "biolink:Drug",
                "predicate": "biolink:treats",
                "object_category": "biolink:Disease",
            },
            {
                "subject_category": "biolink:Disease",
                "predicate": "biolink:correlated_with",
                "object_category": "biolink:Drug",
            },
            {
                "subject_category": "biolink:Disease",
                "predicate": "biolink:related_to",
                "object_category": "biolink:Disease",
            },
            {
                "subject_category": "biolink:ChemicalSubstance",
                "predicate": "biolink:treats",
                "object_category": "biolink:Disease",
            },
            {
                "subject_category": "biolink:ChemicalSubstance",
                "predicate": "biolink:genetically_interacts_with",
                "object_category": "biolink:Disease",
            },
            {
                "subject_category": "biolink:ChemicalSubstance",
                "predicate": "biolink:treats",
                "object_category": "biolink:PhenotypicFeature",
            },
            {
                "subject_category": "biolink:Disease",
                "predicate": "biolink:has_phenotype",
                "object_category": "biolink:PhenotypicFeature",
            },
        ],
        "details": {
            "preferred_prefixes": {
                "biolink:Disease": ["MONDO", "DOID"],
                "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
                "biolink:PhenotypicFeature": ["HP"],
            },
        },
    },
    "kp2": {
        "url": "http://kp2/query",
        "infores": "infores:kp2",
        "maturity": "development",
        "operations": [
            {
                "subject_category": "biolink:Disease",
                "predicate": "biolink:has_phenotype",
                "object_category": "biolink:Gene",
            },
            {
                "subject_category": "biolink:Gene",
                "predicate": "biolink:related_to",
                "object_category": "biolink:Gene",
            },
            {
                "subject_category": "biolink:Gene",
                "predicate": "biolink:related_to",
                "object_category": "biolink:Disease",
            },
            {
                "subject_category": "biolink:ChemicalSubstance",
                "predicate": "biolink:ameliorates",
                "object_category": "biolink:Gene",
            },
        ],
        "details": {
            "preferred_prefixes": {
                "biolink:Disease": ["MONDO", "DOID"],
                "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
                "biolink:PhenotypicFeature": ["HP"],
            },
        },
    },
    "kp3": {
        "url": "http://kp3/query",
        "infores": "infores:kp3",
        "maturity": "development",
        "operations": [
            {
                "subject_category": "biolink:Vitamin",
                "predicate": "biolink:treats",
                "object_category": "biolink:Disease",
            },
        ],
        "details": {
            "preferred_prefixes": {
                "biolink:Disease": ["MONDO", "DOID"],
                "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
                "biolink:PhenotypicFeature": ["HP"],
            },
        },
    },
    "kp4": {
        "url": "http://kp4/query",
        "infores": "infores:kp4",
        "maturity": "development",
        "operations": [
            {
                "subject_category": "biolink:ChemicalSubstance",
                "predicate": "biolink:not_a_real_predicate",
                "object_category": "biolink:Disease",
            },
        ],
        "details": {
            "preferred_prefixes": {
                "biolink:Disease": ["MONDO", "DOID"],
                "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
                "biolink:PhenotypicFeature": ["HP"],
            },
        },
    },
}


async def redisMock(connection_pool=None):
    # Here's where I got documentation for how to do async fakeredis:
    # https://github.com/cunla/fakeredis-py/issues/66#issuecomment-1316045893
    redis = await fakeredis.FakeRedis()
    await redis.set("kps", gzip.compress(json.dumps(default_kps).encode()))
    # set up mock function
    return redis
