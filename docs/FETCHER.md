# Fetcher

## Task
Given a question graph, we would like to generate results from multiple knowledge providers (KPs).

However, suppose we can only query each KP using a single-edge question graph (aka one-hop).

## Approach
The fetcher generates results by sequentially querying each edge of the question graph.

Here's a single step of that sequential process:

1. **Select an edge**

The fetcher selects the one-hop at each step that is estimated to produce the smallest number of results - e.g., question graphs with pinned nodes tend to produce fewer results than those with two unpinned nodes.

```
           one-hop edge
 _____          |           ___________________                   ______
|     |         v          |                   |                 |      |
| Flu | --has_phenotype--> | PhenotypicFeature | --related_to--> | Drug |
|_____|                    |___________________|                 |______|
                  
```

2. **Query the KPs**

Once the one-hop is selected, it is sent asynchronously to all relevant KPs, which each provide results. This information is stored (more on that later) to build each complete result later.

```
KP #1 Results:
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Cough |
|_____|                    |_______|
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Fever |
|_____|                    |_______|

KP #2 Results:
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Fever |
|_____|                    |_______|
 _____                      __________
|     |                    |          |
| Flu | --has_phenotype--> | Headache |
|_____|                    |__________|
```

3. **Update the question graph**

For each KP, the fetcher creates a subgraph from the question graph (of step 1) *without the selected/one-hop edge*, and all orphans (nodes without neighbors) are removed. The endpoints of the selected edge are pinned using the identities retrieved from the KP's results.

```
KP #1 Subgraph:
 ________________                   ______
|                |                 |      |
| {Cough, Fever} | --related_to--> | Drug |
|________________|                 |______|

KP #2 Subgraph
 ___________________                   ______
|                   |                 |      |
| {Fever, Headache} | --related_to--> | Drug |
|___________________|                 |______|

```

For each KP's subgraph, these three steps are run recursively until all edges have been queried. This recursion works because (1) it has a base case – a question graph without nodes/edges simply returns no results – and (2) each step makes progress toward the base case – the subgraph outputted at the end of step 3 is guaranteed to have 1 less edge than the input question graph at step 1.

Once the base case is reached and each recursive call returns, the stored one-hop results (mentioned in step 2) are accessed to yield multiple complete results. The results of the subgraph ("given" by recursion/induction) can be used to construct the results of the original question graphs. Continuing the above example:
```
Question Graph:

           one-hop edge
 _____          |           ___________________                   ______
|     |         v          |                   |                 |      |
| Flu | --has_phenotype--> | PhenotypicFeature | --related_to--> | Drug |
|_____|                    |___________________|                 |______|


Results of one-hop:
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Cough |
|_____|                    |_______|
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Fever |
|_____|                    |_______|


Subgraph (question graph without one-hop edge):
 ________________                   ______
|                |                 |      |
| {Cough, Fever} | --related_to--> | Drug |
|________________|                 |______|


Complete results of subgraph (known because recursion/induction):
 _______                   ___________
|       |                 |           |
| Fever | --related_to--> | Ibuprofen |
|_______|                 |___________|
 _______                   __________________
|       |                 |                  |
| Cough | --related_to--> | Dextromethorphan |
|_______|                 |__________________|


Complete results (combination of one-hop results and subgraph results):
 _____                      _______                   ___________
|     |                    |       |                 |           |
| Flu | --has_phenotype--> | Fever | --related_to--> | Ibuprofen |
|_____|                    |_______|                 |___________|
 _____                      _______                   __________________
|     |                    |       |                 |                  |
| Flu | --has_phenotype--> | Cough | --related_to--> | Dextromethorphan |
|_____|                    |_______|                 |__________________|
```

## Implementation

The implementation details are described well in [COMPONENTS.md](https://github.com/ranking-agent/strider/blob/ff025060c55bf4f357a44f45cfd9288fa9c6a754/docs/COMPONENTS.md). However, the process of stitching together the onehop results to form complete results is not well described.

In `generate_from_results`, the fetcher combines the results of a subgraph with the results of the prior one-hop (missing from the subgraph) to produce the complete results of the question graph. Given a subgraph result, `get_results` uses `key_fcn` (defined in `generate_from_kp`) to identify the nodes in the subgraph result (and their identifiers) that are also nodes of the one-hop. `get_results` then uses `result_map` (also defined in `generate_from_kp`) to fetch the results of the one-hop that use the same identifier.

For example, this pair of one-hop and subgraph results do share the same identifiers for nodes they have in common: 
```
One-hop result:
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Fever |
|_____|                    |_______|

Subgraph result:
 _______                   ___________
|       |                 |           |
| Fever | --related_to--> | Ibuprofen |
|_______|                 |___________|
```
while these two are not compatible.

```
One-hop result:
 _____                      _______
|     |                    |       |
| Flu | --has_phenotype--> | Cough |
|_____|                    |_______|

Subgraph result:
 _______                   ___________
|       |                 |           |
| Fever | --related_to--> | Ibuprofen |
|_______|                 |___________|
```

Following up on this example with some code:
```python
                 _____                      _______
                |     |                    |       |
onehop_result = | Flu | --has_phenotype--> | Fever |
                |_____|                    |_______|
                  n0                          n1
                   _______                   ___________
                  |       |                 |           |
subgraph_result = | Fever | --related_to--> | Ibuprofen |
                  |_______|                 |___________|
                     n1                          n2

# Question graph and subgraph result both have n1, and
# in the subgraph result, n1 has the identifier Fever
assert key_fcn(subgraph_result) == (('n1', 'Fever'),)
assert onehop_result in result_map[key_fcn(subgraph_result)]
```