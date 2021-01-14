import itertools
import time
from strider.util import WrappedBMT
WBMT = WrappedBMT()


def create_kp(args):
    """
    Generate a KP given a tuple of input, edge, output
    """
    source, edge, target = args
    return {
        "url": "http://mykp",
        "operations": [{
            "source_type": source,
            "edge_type": f"-{edge}->",
            "target_type": target,
        }]
    }


def generate_kps(qty):
    """
    Generate a given number of KPs using permutations
    of the biolink model
    """
    node_categories = WBMT.get_descendants('biolink:NamedThing')
    edge_predicates = WBMT.get_descendants('biolink:related_to')
    kp_generator = map(
        create_kp,
        itertools.product(
            node_categories,
            edge_predicates,
            node_categories,
        )
    )

    return {str(i): kp for i, kp in enumerate(kp_generator) if i < qty}


async def time_and_display(f, msg):
    """ Time a function and print the time """
    start_time = time.time()
    await f()
    total = time.time() - start_time
    print("\n-------------------------------------------")
    print(f"Total time to {msg}: {total:.2f}s")
    print("-------------------------------------------")
