"""KP registry."""
import json
import logging
import re
import uuid

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


def fix_knode(knode, curie_map):
    """Replace curie with preferred, if possible."""
    knode['id'] = curie_map.get(knode['id'], knode['id'])
    return knode


def fix_kedge(kedge, curie_map):
    """Replace curie with preferred, if possible."""
    kedge['source_id'] = curie_map.get(kedge['source_id'], kedge['source_id'])
    kedge['target_id'] = curie_map.get(kedge['target_id'], kedge['target_id'])
    return kedge


def fix_result(result, curie_map):
    """Replace curie with preferred, if possible."""
    result['node_bindings'] = {
        qid: curie_map.get(kid, kid)
        for qid, kid in result['node_bindings'].items()
    }
    return result


async def get_curie_transformations(message, preferences):
    """Get CURIE transformations according to CURIE prefix preferences.

    preferences is a map: type -> prefix
    """
    preferences = {
        'chemical_substance': 'CHEBI',
        **preferences,
    }
    curie_map = dict()
    url_base = 'https://nodenormalization-sri.renci.org/get_normalized_nodes'
    curies = set()
    if 'query_graph' in message:
        curies |= {
            qnode.get('curie')
            for qnode in message['query_graph']['nodes']
            if qnode.get('curie', False)
        }
    if 'knowledge_graph' in message:
        curies |= {
            knode.get('id')
            for knode in message['knowledge_graph']['nodes']
        }
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(
            url_base,
            params={'curie': list(curies)},
        )

    if response.status_code >= 300:
        return curie_map
    for curie, entity in response.json().items():
        if entity is None:
            continue
        for type_ in entity['type']:
            if type_ in preferences:
                prefix = preferences[type_]
                break
        else:
            prefix = None
        if prefix is None:
            continue
        try:
            replacement = next(
                ident['identifier']
                for ident in entity['equivalent_identifiers']
                if ident['identifier'].startswith(prefix)
            )
        except StopIteration:
            # no preferred replacement
            continue
        curie_map[curie] = replacement
    return curie_map


def apply_curie_map(message, curie_map):
    """Translate all pinned qnodes to preferred prefix."""
    message['query_graph']['nodes'] = [
        fix_qnode(qnode, curie_map)
        for qnode in message['query_graph']['nodes']
    ]
    message['knowledge_graph']['nodes'] = [
        fix_knode(knode, curie_map)
        for knode in message['knowledge_graph']['nodes']
    ]
    message['knowledge_graph']['edges'] = [
        fix_kedge(kedge, curie_map)
        for kedge in message['knowledge_graph']['edges']
    ]
    message['results'] = [
        fix_result(result, curie_map)
        for result in message['results']
    ]
    return message


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


def rand_str():
    """Generate random string."""
    return '"' + str(uuid.uuid4()) + '"'


def kp_func(
        url,
        preferred_prefixes=None,
        in_transformers=None,
        out_transformers=None,
):
    """Generate KP function."""
    async def func(request):
        """Call KP with pre- and post-processing."""
        qgraph = request['query_graph']
        request = {'message': request}

        curie_map = await get_curie_transformations(
            request['message'],
            preferred_prefixes,
        )
        # request['message'] = apply_curie_map(request['message'], curie_map)
        in_transformers.update({
            '"' + old + '"': '"' + new + '"'
            for old, new in curie_map.items()
        })
        out_transformers.update({
            '"' + new + '"': '"' + old + '"'
            for old, new in curie_map.items()
        })
        request_str = json.dumps(request)
        if in_transformers is not None:
            for old, new in in_transformers.items():
                request_str = re.sub(old, new, request_str)
        response_str = await call_kp(url, request_str)
        if out_transformers is not None:
            for old, new in out_transformers.items():
                response_str = re.sub(old, new, response_str)
        try:
            response = json.loads(response_str)
        except json.JSONDecodeError:
            LOGGER.error('"OK" message is not proper JSON: %s', response_str)
            return {
                "query_graph": qgraph,
                "results": [],
            }

        curie_map = await get_curie_transformations(response, {})

        curie_map['"e0"'] = rand_str()
        curie_map['"e1"'] = rand_str()
        curie_map['"e2"'] = rand_str()

        # response = apply_curie_map(response, curie_map)
        for old, new in curie_map.items():
            response_str = re.sub(old, new, response_str)
        response = json.loads(response_str)
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
            preferred_prefixes=provider[5]['details']['preferred_prefixes'],
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
