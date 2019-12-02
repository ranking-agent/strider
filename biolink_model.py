"""Biolink model."""
import requests
import yaml


class BiolinkModel():
    """Biolink model."""

    def __init__(self, filename_url):
        """Initialize."""
        response = requests.get(filename_url)
        if response.status_code != 200:
            raise RuntimeError(f'Unable to access Biolink Model at {filename_url}')
        self.model = yaml.load(response.text, Loader=yaml.FullLoader)

    def get_children(self, concepts):
        """Get direct children of concepts."""
        return {key for key, value in self.model['classes'].items() if value.get('is_a', None) in concepts}

    def get_descendants(self, concepts):
        """Get all descendants of concepts, recursively."""
        children = self.get_children(concepts)
        if not children:
            return concepts
        return self.get_descendants(
            children
        ) | children

    def get_parents(self, concepts):
        """Get direct parent of each concept."""
        return {
            self.model['classes'][c]['is_a']
            for c in concepts
            if 'is_a' in self.model['classes'][c]
        }

    def get_ancestors(self, concepts):
        """Get all ancestors of concepts."""
        parents = self.get_parents(concepts)
        if not parents:
            return concepts
        return self.get_ancestors(
            parents
        ) | concepts

    def get_lineage(self, concepts):
        """Get all ancestors and descendants of concepts."""
        if isinstance(concepts, str):
            concepts = {concepts}
        return self.get_descendants(concepts) | self.get_ancestors(concepts)

    def compatible(self, type1, type2):
        """Determine whether type1 and type2 are a descendant/ancestor pair."""
        return type1 in self.get_lineage(type2)
