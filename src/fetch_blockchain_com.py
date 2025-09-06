import os
import json
import logging
import time
import requests
import re
from pathlib import Path

logger = logging.getLogger(__name__)

BLOCKCHAIN_BASE = "https://blockchain.info"
# Unified data directory - use Data/graph (case-sensitive)
DATA_DIR = Path("Data/graph")
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"BlockchainComFetcher using data directory: {DATA_DIR.resolve()}")


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
        """Build graph data for an address"""
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
                edges.append({
                    "id": f"{src_addr}->{txid}",
                    "source": src_addr,
                    "target": txid,
                    "type": "SENT_FROM",
                    "value": prev_out.get("value", 0)
                })

            for vout in tx.get("out", []):
                dst_addr = vout.get("addr")
                if not dst_addr:
                    continue

                ensure_node(dst_addr, dst_addr[:10], "address")
                edges.append({
                    "id": f"{txid}->{dst_addr}",
                    "source": txid,
                    "target": dst_addr,
                    "type": "SENT_TO",
                    "value": vout.get("value", 0)
                })

        graph = {
            "nodes": list(nodes.values()),
            "edges": edges,
            "meta": {
                "address": address,
                "tx_count": len(data.get("txs", [])),
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }

        return graph

    def save_graph(self, graph: dict, filename: str | None = None) -> str:
        """Save graph to file with sanitized filename"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if not filename:
            base = graph.get("meta", {}).get("address", "graph")
            # Sanitize filename - only allow alphanumeric, underscore, hyphen
            safe_base = re.sub(r'[^A-Za-z0-9_\-]', '_', base)
            filename = f"graph_{safe_base[:12]}.json"

        path = DATA_DIR / filename

        # Use temporary file for atomic write
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        os.replace(tmp, path)

        logger.info(
            f"graph_saved path={path} nodes={len(graph.get('nodes', []))} edges={len(graph.get('edges', []))}")
        return str(path)
