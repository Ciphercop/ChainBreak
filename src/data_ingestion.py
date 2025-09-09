"""
Data Ingestion Layer for ChainBreak
Handles BlockCypher API integration and Neo4j data storage
"""

from abc import ABC, abstractmethod
from neo4j import GraphDatabase
from blockcypher import get_address_details, get_transaction_details
import logging
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDataIngestor(ABC):
    """Abstract base class for data ingestion"""
    
    @abstractmethod
    def ingest_address_data(self, address: str, blockchain: str = 'btc') -> bool:
        """Ingest data for a specific address"""
        pass
    
    @abstractmethod
    def get_address_transactions(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transactions for an address"""
        pass
    
    @abstractmethod
    def get_transaction_details(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific transaction"""
        pass
    
    @abstractmethod
    def is_operational(self) -> bool:
        """Check if the ingestor is operational"""
        pass


class Neo4jDataIngestor(BaseDataIngestor):
    """Handles blockchain data ingestion from BlockCypher API to Neo4j"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize the data ingestor with Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self._test_connection()
            self._setup_database()
            logger.info("Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _test_connection(self):
        """Test Neo4j connection"""
        with self.driver.session() as session:
            session.run("RETURN 1")
    
    def _setup_database(self):
        """Setup database constraints and indexes"""
        with self.driver.session() as session:
            try:
                session.run("""
                    CREATE CONSTRAINT address_unique IF NOT EXISTS 
                    FOR (a:Address) REQUIRE a.address IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT transaction_unique IF NOT EXISTS 
                    FOR (t:Transaction) REQUIRE t.tx_hash IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT block_unique IF NOT EXISTS 
                    FOR (b:Block) REQUIRE b.block_hash IS UNIQUE
                """)
                
                session.run("""
                    CREATE INDEX address_balance IF NOT EXISTS 
                    FOR (a:Address) ON (a.balance)
                """)
                session.run("""
                    CREATE INDEX transaction_value IF NOT EXISTS 
                    FOR (t:Transaction) ON (t.value)
                """)
                logger.info("Database constraints and indexes set up successfully")
            except Exception as e:
                logger.warning(f"Some database setup operations failed: {e}")
    
    def ingest_address_data(self, address: str, blockchain: str = 'btc') -> bool:
        """Ingest blockchain data for an address"""
        try:
            logger.info(f"Ingesting data for address: {address}")
            
            # Get address details from BlockCypher
            address_data = get_address_details(address, coin_symbol=blockchain)
            if not address_data:
                logger.warning(f"No data found for address: {address}")
                return False
            
            # Create address node
            self._create_address_node(address, address_data)
            
            # Ingest transactions
            self._ingest_transactions(address, address_data.get('txrefs', []))
            
            logger.info(f"Successfully ingested data for address: {address}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting data for address {address}: {str(e)}")
            return False
    
    def get_address_transactions(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transactions for an address from Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Address {address: $address})-[:PARTICIPATED_IN]->(t:Transaction)
                    RETURN t ORDER BY t.timestamp DESC LIMIT $limit
                """, address=address, limit=limit)
                return [dict(record['t']) for record in result]
        except Exception as e:
            logger.error(f"Error getting transactions for {address}: {e}")
            return []
    
    def get_transaction_details(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction details from Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (t:Transaction {tx_hash: $tx_hash})
                    RETURN t
                """, tx_hash=tx_hash)
                record = result.single()
                return dict(record['t']) if record else None
        except Exception as e:
            logger.error(f"Error getting transaction {tx_hash}: {e}")
            return None
    
    def is_operational(self) -> bool:
        """Check if Neo4j connection is operational"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False
    
    def _create_address_node(self, address: str, address_data: Dict[str, Any]):
        """Create or update address node in Neo4j"""
        with self.driver.session() as session:
            session.run("""
                MERGE (a:Address {address: $address})
                SET a.balance = $balance,
                    a.total_received = $total_received,
                    a.total_sent = $total_sent,
                    a.n_tx = $n_tx,
                    a.last_updated = datetime()
            """, address=address, 
                 balance=address_data.get('balance', 0),
                 total_received=address_data.get('total_received', 0),
                 total_sent=address_data.get('total_sent', 0),
                 n_tx=address_data.get('n_tx', 0))
    
    def _ingest_transactions(self, address: str, transactions: List[Dict[str, Any]]):
        """Ingest transaction data into Neo4j"""
        for tx in transactions:
            try:
                self._create_transaction_nodes(tx, address)
            except Exception as e:
                logger.warning(f"Failed to ingest transaction {tx.get('tx_hash', 'unknown')}: {e}")
                continue
    
    def _create_transaction_nodes(self, tx_data: Dict[str, Any], source_address: str):
        """Create transaction and related nodes"""
        tx_hash = tx_data.get('tx_hash')
        if not tx_hash:
            return
        
        with self.driver.session() as session:
            # Create transaction node
            session.run("""
                MERGE (t:Transaction {tx_hash: $tx_hash})
                SET t.value = $value,
                    t.timestamp = $timestamp,
                    t.block_height = $block_height,
                    t.confirmations = $confirmations
            """, tx_hash=tx_hash,
                 value=tx_data.get('value', 0),
                 timestamp=tx_data.get('confirmed', ''),
                 block_height=tx_data.get('block_height', 0),
                 confirmations=tx_data.get('confirmations', 0))
            
            # Create relationships
            session.run("""
                MATCH (a:Address {address: $address})
                MATCH (t:Transaction {tx_hash: $tx_hash})
                MERGE (a)-[:PARTICIPATED_IN]->(t)
            """, address=source_address, tx_hash=tx_hash)
    
    def close(self):
        """Close Neo4j driver connection"""
        if hasattr(self, 'driver'):
            self.driver.close()


class JSONDataIngestor(BaseDataIngestor):
    """Lightweight JSON-based data ingestor for fallback mode"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize JSON ingestor with data directory"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache = {}
        logger.info(f"JSON ingestor initialized with data directory: {self.data_dir}")
    
    def ingest_address_data(self, address: str, blockchain: str = 'btc') -> bool:
        """JSON ingestor doesn't actually ingest - it just loads existing data"""
        logger.info(f"JSON ingestor: checking for existing data for {address}")
        return self._has_address_data(address)
    
    def get_address_transactions(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transactions from JSON files"""
        try:
            data = self._load_address_data(address)
            if data and 'transactions' in data:
                return data['transactions'][:limit]
            return []
        except Exception as e:
            logger.error(f"Error loading transactions for {address}: {e}")
            return []
    
    def get_transaction_details(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction details from JSON cache"""
        return self.cache.get(tx_hash)
    
    def is_operational(self) -> bool:
        """JSON ingestor is always operational"""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from JSON data"""
        try:
            address_count = 0
            transaction_count = 0
            
            for json_file in self.data_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if 'meta' in data and 'address' in data['meta']:
                            address_count += 1
                        if 'transactions' in data:
                            transaction_count += len(data['transactions'])
                except Exception:
                    continue
            
            return {
                'node_counts': {
                    'address_count': address_count,
                    'transaction_count': transaction_count
                },
                'relationship_count': transaction_count,  # Each transaction is a relationship
                'data_files': len(list(self.data_dir.glob("*.json")))
            }
        except Exception as e:
            logger.warning(f"Error getting JSON statistics: {e}")
            return {}
    
    def _has_address_data(self, address: str) -> bool:
        """Check if data exists for an address"""
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('meta', {}).get('address') == address:
                        return True
            except:
                continue
        return False
    
    def _load_address_data(self, address: str) -> Optional[Dict[str, Any]]:
        """Load data for an address from JSON files"""
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('meta', {}).get('address') == address:
                        return data
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue
        return None
    
    def close(self):
        """JSON ingestor doesn't need cleanup"""
        pass
