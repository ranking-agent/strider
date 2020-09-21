"""KP registry."""
import json
import logging

import httpx

from strider.util import ensure_list

LOGGER = logging.getLogger(__name__)


def message_to_list_form(message):
    """Convert *graph nodes/edges and node/edge bindings to list forms."""
    if message['results']:
        message['results'] = [
            {
                'node_bindings': [
                    {
                        'qg_id': qg_id,
                        **binding,
                    }
                    for qg_id, binding in result['node_bindings'].items()
                ],
                'edge_bindings': [
                    {
                        'qg_id': qg_id,
                        **binding,
                    }
                    for qg_id, binding in result['edge_bindings'].items()
                ],
            } for result in message.get('results', [])
        ]
    if message['knowledge_graph']['nodes']:
        message['knowledge_graph']['nodes'] = [
            {
                'id': node['id'],
                **node,
            }
            for node in message['knowledge_graph']['nodes']
        ]
    if message['knowledge_graph']['edges']:
        message['knowledge_graph']['edges'] = [
            {
                'id': edge['id'],
                **edge,
            }
            for edge in message['knowledge_graph']['edges']
        ]
    return message


def message_to_dict_form(message):
    """Convert *graph nodes/edges and node/edge bindings to dict forms."""
    if message['results']:
        if isinstance(message['results'][0]['node_bindings'], list):
            message['results'] = [
                {
                    'node_bindings': {
                        binding['qg_id']: [binding]
                        for binding in result['node_bindings']
                    },
                    'edge_bindings': {
                        binding['qg_id']: [binding]
                        for binding in result['edge_bindings']
                    },
                } for result in message.get('results', [])
            ]
        elif not isinstance(
                list(message['results'][0]['node_bindings'].values())[0],
                dict
        ):
            message['results'] = [
                {
                    'node_bindings': {
                        key: [{'kg_id': el} for el in ensure_list(bindings)]
                        for key, bindings in result['node_bindings'].items()
                    },
                    'edge_bindings': {
                        key: [{'kg_id': el} for el in ensure_list(bindings)]
                        for key, bindings in result['edge_bindings'].items()
                    },
                } for result in message.get('results', [])
            ]
    if message['knowledge_graph']['nodes']:
        message['knowledge_graph']['nodes'] = {
            node['id']: node
            for node in message['knowledge_graph']['nodes']
        }
    if message['knowledge_graph']['edges']:
        message['knowledge_graph']['edges'] = {
            edge['id']: edge
            for edge in message['knowledge_graph']['edges']
        }
    return message


def fix_qnode(qnode, curie_map):
    """Replace curie with preferred, if possible."""
    if 'curie' in qnode:
        qnode['curie'] = curie_map.get(qnode['curie'], qnode['curie'])
    return qnode


async def prefer(qgraph, prefix):
    """Translate all pinned qnodes to preferred prefix."""
    if prefix is None:
        return qgraph
    url_base = 'https://nodenormalization-sri.renci.org/get_normalized_nodes'
    curies = [
        qnode.get('curie')
        for qnode in qgraph['nodes']
        if qnode.get('curie')
    ]
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(
            url_base,
            params={'curie': curies},
        )

    curie_map = dict()
    for key, value in response.json().items():
        try:
            replacement = next(
                entity['identifier']
                for entity in value['equivalent_identifiers']
                if entity['identifier'].startswith(prefix)
            )
        except StopIteration:
            # no preferred replacement
            continue
        curie_map[key] = replacement

    qgraph['nodes'] = [
        fix_qnode(qnode, curie_map)
        for qnode in qgraph['nodes']
    ]
    return qgraph


async def call_kp(url, request):
    """Call KP.

    Input request can be a string or JSON-able object.
    Output is the raw text of the response body.
    """
    if not isinstance(request, str):
        request = json.dumps(request).encode('utf-8')
    async with httpx.AsyncClient(
            timeout=None,
            headers={'Content-Type': 'application/json'},
            verify=False,
    ) as client:
        try:
            response = await client.post(
                url,
                data=request,
                timeout=30.0,
            )
        except httpx.ReadTimeout:
            LOGGER.error(
                "ReadTimeout: endpoint: %s, JSON: %s",
                url, json.dumps(request)
            )
            return []
    if response.status_code >= 300:
        LOGGER.error(
            'KP error (%s):\n'
            'request:\n%s\n'
            'response:\n%s',
            url,
            request,
            response.text,
        )
        raise RuntimeError('KP call failed.')
    return response.text


def kp_func(
        url,
        preferred_prefix=None,
        in_transformers=None,
        out_transformers=None,
):
    """Generate KP function."""
    async def func(request):
        """Call KP with pre- and post-processing."""
        request = {'message': request}
        request['message']['query_graph'] = await prefer(
            request['message']['query_graph'],
            preferred_prefix,
        )
        request = json.dumps(request)
        if in_transformers is not None:
            for old, new in in_transformers.items():
                request = request.replace(old, new)
        response = await call_kp(url, request)
        if out_transformers is not None:
            for old, new in out_transformers.items():
                response = response.replace(old, new)
        response = json.loads(response)
        return message_to_dict_form(response)
    return func


class Registry():
    """KP registry."""

    def __init__(self, url):
        """Initialize."""
        self.url = url
        self.locals = dict()

    async def __aenter__(self):
        """Enter context."""
        return self

    async def __aexit__(self, *args):
        """Exit context."""

    async def setup(self):
        """Set up database table."""

    async def get_all(self):
        """Get all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/kps',
            )
            assert response.status_code < 300
        return response.json()

    async def get_one(self, url):
        """Get a specific KP."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/kps/{url}',
            )
            assert response.status_code < 300
        provider = response.json()
        return kp_func(
            url,
            preferred_prefix=provider[5]['details']['preferred_prefix'],
            in_transformers=provider[5]['details']['in_transformers'],
            out_transformers=provider[5]['details']['out_transformers'],
        )

    async def add(self, *kps):
        """Add KP(s)."""
        kps = {
            kp.name: await kp.get_operations()
            for kp in kps
        }
        self.locals.update({
            kp.name: kp
            for kp in kps
        })
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/kps',
                json=kps
            )
            assert response.status_code < 300

    async def delete_one(self, url):
        """Delete a specific KP."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f'{self.url}/kps/{url}',
            )
            assert response.status_code < 300

    async def search(
            self,
            source_types, edge_types, target_types,
            allowlist=None, denylist=None,
    ):
        """Search for KPs matching a pattern."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/search',
                json={
                    'source_type': source_types,
                    'target_type': target_types,
                    'edge_type': edge_types,
                }
            )
            assert response.status_code < 300
        return [
            kp_func(**details)
            for kpid, details in response.json().items()
            if (
                (allowlist is None or kpid in allowlist)
                and (denylist is None or kpid not in denylist)
            )
        ]

    async def delete_all(self):
        """Delete all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/clear',
            )
            assert response.status_code < 300
        return response.json()
