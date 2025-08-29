import os
import json
import logging
import time
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

BLOCKCHAIN_BASE = "https://blockchain.info"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class BlockchainComFetcher:
    def __init__(self, session: requests.Session | None = None, rate_limit_s: float = 0.2, timeout: int = 20):
        self.session = session or requests.Session()
        self.rate_limit_s = rate_limit_s
        self.timeout = timeout

    def _get(self, url: str, params: dict | None = None):
        try:
            res = self.session.get(url, params=params, timeout=self.timeout)
            time.sleep(self.rate_limit_s)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logger.error(f"fetch_error url={url} params={params} err={e}")
            raise

    def fetch_tx(self, tx_hash: str) -> dict:
        url = f"{BLOCKCHAIN_BASE}/rawtx/{tx_hash}"
        return self._get(url)

    def fetch_address(self, address: str, limit: int = 50) -> dict:
        url = f"{BLOCKCHAIN_BASE}/rawaddr/{address}"
        params = {"limit": limit}
        return self._get(url, params=params)

    def fetch_block(self, block_hash: str) -> dict:
        url = f"{BLOCKCHAIN_BASE}/rawblock/{block_hash}"
        return self._get(url)

    def build_graph_for_address(self, address: str, tx_limit: int = 50) -> dict:
        data = self.fetch_address(address, limit=tx_limit)
        nodes = {}
        edges = []

        def ensure_node(node_id: str, label: str, type_: str):
            if node_id not in nodes:
                nodes[node_id] = {"id": node_id, "label": label, "type": type_}
        ensure_node(address, address[:10], "address")
        for tx in data.get("txs", []):
            txid = tx.get("hash")
            if not txid:
                continue
            ensure_node(txid, txid[:10], "transaction")
            for vin in tx.get("inputs", []):
                prev_out = vin.get("prev_out") or {}
                src_addr = prev_out.get("addr")
                if not src_addr:
                    continue
                ensure_node(src_addr, src_addr[:10], "address")
                edges.append({"id": f"{src_addr}->{txid}", "source": src_addr,
                             "target": txid, "type": "SENT_FROM", "value": prev_out.get("value", 0)})
            for vout in tx.get("out", []):
                dst_addr = vout.get("addr")
                if not dst_addr:
                    continue
                ensure_node(dst_addr, dst_addr[:10], "address")
                edges.append({"id": f"{txid}->{dst_addr}", "source": txid,
                             "target": dst_addr, "type": "SENT_TO", "value": vout.get("value", 0)})
        graph = {"nodes": list(nodes.values()), "edges": edges, "meta": {
            "address": address, "tx_count": len(data.get("txs", []))}}
        return graph

    def save_graph(self, graph: dict, filename: str | None = None) -> str:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not filename:
            base = graph.get("meta", {}).get("address", "graph")
            filename = f"graph_{base[:12]}.json"
        path = DATA_DIR / filename
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False)
        os.replace(tmp, path)
        logger.info(
            f"graph_saved path={path} nodes={len(graph.get('nodes', []))} edges={len(graph.get('edges', []))}")
        return str(path)
