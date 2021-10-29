# Query planning

## Currently...
Strider handles a complex query graph by decomposing it into its component edges, corresponding to one-hop queries, then sending each one to the appropriate KPs before re-assembling the one-hop results into complete answers. The results of each one-hop query impose constraints on the adjacent (sharing at least one endpoint) one-hop queries. So while it is possible in theory to execute the one-hop queries in any arbitrary order and afterward filtering according to the implied constraints, it is much more efficient to leverage partial results for some edges to refine the queries for adjacent edges.

For example, a query graph could ask for diseases that shared phenotypes with Ebola: `(:disease)-has_phenotype->(:phenotype)<-has_phenotype-(EBOLA)`. It is possible to solve this by querying first for the less-constrained edge, resulting in all disease-phenotype relationships, then querying for the phenotypes associated with Ebola, and computing the intersection of the two phenotype sets. It is much more efficient, however, to query for the phenotypes associated with Ebola and subsequently find all disease associated with those phenotypes.

We model the cost of each one-hop query as a function of the number of possible identities for each endpoint. The Ebola-phenotype edge has cost $1 \times N=N$, where $N$ is the number of phenotypes, whereas the disease-phenotype edge has complexity $N \times N=N^2$, where we assume there are also $N$ possible diseases, for simplicity. This model is consistent with a particular type of KP that stores a connectivity matrix and requires that we parse the intersection of the given rows and columns to construct one-hop results.

For planning purposes, we imagine that Strider makes only one KP query at a time, i.e. that it operates entirely synchronously, though in reality it's complicated. Strider operates iteratively by solving one edge, removing it from the query graph and pinning its endpoints accordingly, and then solving the resulting reduced query. For example, beginning with the query `(:drug)-treats->(:disease)-has_phenotype->(:phenotype)<-has_phenotype-(EBOLA)`, we could query KPs for `(:phenotype)<-has_phenotype-(EBOLA)`, discovering for example that phenotypes "fever" and "pain" are associated with Ebola, and then reduce the original query to `(:drug)-treats->(:disease)-has_phenotype->(FEVER or PAIN)`.

The optimal query plan is the one that has the lowest total expected cost. At any time, the cost for the immediate next step is knowable, but expected costs for future steps must be estimated. 

We have invented an heuristic to reflect the expected costs for future steps that we call the "constrainedness" of a node/edge. This is in reference to the practice of "pinning" a query-graph node to a set of CURIES, i.e. `"curies": ["..."]`. The constrainedness of a node is the expected number of unique knowledge-graph nodes that will be bound to that query-graph node in all results. This can be less that one for example in unlikely queries like `(EBOLA)-treats->(:disease)`. The constrainedness is a function of the roles of each node in the query graph. The phenotype in `(:phenotype)<-has_phenotype-(EBOLA)` is "more constrained" than in `(COVID)-has_phenotype->(:phenotype)<-has_phenotype-(EBOLA)`. We approximate this metric via a process similar to loopy belief propagation, in which the expected number of bound nodes for each position affects the expectations for all of its neighbors.

Each edge is assigned a "constrainedness" metric that is the product of the constrainedness of its endpoints. Then the total cost metric for each edge is the ratio of the (known) immediate cost and the edge's constrainedness, reflecting future costs (constrainedness is inversely correlated with future costs).

We traverse edges in the order implied by these costs, which results in a substantial reduction in computational effort relative to arbitrary ordering for most query graphs. A "greedy" approach that considers only the immediate cost also underperforms the proposed method for many branching and/or doubly-pinned queries.

## Progress notes...

Need to predict how many knodes are bound to each qnode _after_ taking a given step.

$n$ and $m$ sets of nodes are connected randomly. The probability of any given pair being connected is $p=\frac{N_{edges}}{N_{nodes}^2}$. The expected number of connected nodes from $n$ is $n (1-(1-p)^m)$ ($~nmp$ for $m >> n$ and $p << 1$). The expected number of edges is $nmp$.
If $n'$ and $m'$ are connected by $nmp$ edges, the expected number from $n'$ that are connected to subset $m''$ of $m'$ is $n' (1-(1-\frac{nmp}{n'm'})^{m''})$

$$n' = n \prod_i\left(1-\left(1-\frac{K_i}{NM_i}\right)^{m_i}\right)$$

For any node in any position, what is the probability that it is part of at least one complete result? Then we can multiply by a scalar :) Or what is the expected number of results of which this node is a part? To ask this question, pin each visited node to 1 and recurse.

The expected number of results does not depend on _which_ nodes are pinned?! These are different just because the _shapes_ of the distributions are different? On a linear query with two pinned nodes, if they are close together, the distribution is bimodal (zero and high). When they are far apart, the distribution is more compact around the expectation. But. Will change if we use the proper expression for the expected number of nodes?
