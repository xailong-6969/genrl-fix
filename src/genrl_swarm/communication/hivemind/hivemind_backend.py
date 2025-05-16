import os
import pickle
import time
from typing import Any, List, Sequence

from hivemind import DHT, get_dht_time

from genrl_swarm.communication.communication import Communication


class HivemindBackend(Communication):
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 600,
        bootstrap=False,
    ):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.timeout = timeout
        self.initial_peers = initial_peers
        if bootstrap:
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
            )
            self.initial_peers = self.dht.get_visible_maddrs()
        else:
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=self.initial_peers,
            )
        self.step_ = 0

    def all_gather_object(self, obj: Any) -> Sequence[Any]:
        # TODO(jkolehm): change pickle to something more secure before launching the code.
        key = str(self.step_)
        self.dht.store(
            key,
            subkey=str(self.dht.peer_id),
            value=pickle.dumps(obj),
            expiration_time=get_dht_time() + self.timeout,
        )
        t_ = time.monotonic()
        while True:
            output_, _ = self.dht.get(key)
            if len(output_) >= self.world_size:
                break
            else:
                if time.monotonic() - t_ > self.timeout:
                    raise RuntimeError(
                        f"Failed to obtain {self.world_size} values for {key} within timeout."
                    )
        self.step_ += 1

        tmp = sorted(
            [(key, pickle.loads(value.value)) for key, value in output_.items()],
            key=lambda x: x[0],
        )
        _, output = zip(*tmp)
        return output
