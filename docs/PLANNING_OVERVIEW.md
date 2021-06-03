## Query Planning

Before a query can be executed, a plan needs to be created. This plan describes which edges will be traversed in what order and what KPs will be contacted for each edge. Here is an example of a looped query graph and associated plan:

![Example QG](example_qgraph.png)

```json
{
    "n0-n0n1-n1": [{
        "name": "kp3",
        "url": "http://kp3",
        "edge_predicate": "-biolink:has_phenotype->",
        "source_category": "biolink:Disease",
        "target_category": "biolink:PhenotypicFeature",
        "reverse": false
    }],
    "n1-n2n1-n2": [{
        "name": "kp2",
        "url": "http://kp2",
        "edge_predicate": "<-biolink:treats-",
        "source_category": "biolink:PhenotypicFeature",
        "target_category": "biolink:ChemicalSubstance",
        "reverse": true
    }],
    "n2-n2n0-n0": [{
        "name": "kp1",
        "url": "http://kp1",
        "edge_predicate": "-biolink:treats->",
        "source_category": "biolink:ChemicalSubstance",
        "target_category": "biolink:Disease",
        "reverse": false
    }]
}
```

Things to notice about the plan:

* Keys in the plan are subject, edge, object. This is necessary because we are allowed to traverse edges in either the forward or backwards direction. In this case, we could traverse edge n1n2 in the reverse direction by changing the predicate from `biolink:treats` to `biolink:treated_by`. 
* The plan always starts from a pinned node (one with an ID). This is because we have to send an ID to the KPs.
* The KP list associated with each edge in the plan includes categories and predicates. This is because we are allowed to change the predicate when contacting a KP. For example, if there is a KP that accepts `positively_correlated_with` as the predicate we can convert `correlated_with` to `positively_correlated_with` when we contact it. 
