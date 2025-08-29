"""
Data Ingestion Layer for ChainBreak
Handles BlockCypher API integration and Neo4j data storage
"""

from neo4j import GraphDatabase
from blockcypher import get_address_details, get_transaction_details
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BlockchainDataIngestor:
    """Handles blockchain data ingestion from BlockCypher API to Neo4j"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize the data ingestor with Neo4j connection"""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self._setup_database()
        
    def _setup_database(self):
        """Setup database constraints and indexes"""
        with self.driver.session() as session:
            # Create constraints for uniqueness
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
            
            # Create indexes for performance
            session.run("""
                CREATE INDEX address_balance IF NOT EXISTS 
                FOR (a:Address) ON (a.balance)
            """)
            session.run("""
                CREATE INDEX transaction_value IF NOT EXISTS 
                FOR (t:Transaction) ON (t.value)
            """)
            
    def ingest_address_data(self, address: str, blockchain: str = 'btc') -> bool:
        """Ingest data for a specific address"""
        try:
            logger.info(f"Ingesting data for address: {address}")
            
            # Get address details from BlockCypher
            address_details = get_address_details(address, coin_symbol=blockchain)
            
            # Create address node in Neo4j
            with self.driver.session() as session:
                session.write_transaction(self._create_address_node, address_details)
                
            # Ingest transaction data
            self._ingest_transactions(address, address_details.get('txrefs', []), blockchain)
            
            logger.info(f"Successfully ingested data for address: {address}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting data for address {address}: {str(e)}")
            return False
            
    def _create_address_node(self, tx, address_details: Dict[str, Any]):
        """Create or update address node in Neo4j"""
        query = """
        MERGE (a:Address {address: $address})
        SET a.balance = $balance,
            a.total_received = $total_received,
            a.total_sent = $total_sent,
            a.transaction_count = $transaction_count,
            a.last_updated = datetime()
        """
        
        tx.run(query, 
               address=address_details['address'],
               balance=address_details.get('balance', 0),
               total_received=address_details.get('total_received', 0),
               total_sent=address_details.get('total_sent', 0),
               transaction_count=address_details.get('n_tx', 0))
               
    def _ingest_transactions(self, address: str, txrefs: list, blockchain: str):
        """Ingest transaction data for an address"""
        for txref in txrefs[:100]:  # Limit to 100 transactions for performance
            try:
                tx_hash = txref['tx_hash']
                tx_details = get_transaction_details(tx_hash, coin_symbol=blockchain)
                
                with self.driver.session() as session:
                    session.write_transaction(self._create_transaction_nodes, 
                                           tx_details, address)
                                           
            except Exception as e:
                logger.warning(f"Error ingesting transaction {txref.get('tx_hash', 'unknown')}: {str(e)}")
                continue
                
    def _create_transaction_nodes(self, tx, tx_details: Dict[str, Any], source_address: str):
        """Create transaction and related nodes in Neo4j"""
        # Create transaction node
        tx_query = """
        MERGE (t:Transaction {tx_hash: $tx_hash})
        SET t.value = $value,
            t.fee = $fee,
            t.timestamp = datetime($timestamp),
            t.block_height = $block_height,
            t.confirmations = $confirmations
        """
        
        tx.run(tx_query,
               tx_hash=tx_details['hash'],
               value=tx_details.get('total', 0),
               fee=tx_details.get('fees', 0),
               timestamp=tx_details.get('received', ''),
               block_height=tx_details.get('block_height', 0),
               confirmations=tx_details.get('confirmations', 0))
               
        # Create relationships for inputs and outputs
        for input_tx in tx_details.get('inputs', []):
            input_address = input_tx.get('addresses', [''])[0]
            if input_address:
                # Create input address node
                tx.run("""
                    MERGE (a:Address {address: $address})
                """, address=input_address)
                
                # Create SENT_FROM relationship
                tx.run("""
                    MATCH (a:Address {address: $address})
                    MATCH (t:Transaction {tx_hash: $tx_hash})
                    MERGE (a)-[:SENT_FROM]->(t)
                """, address=input_address, tx_hash=tx_details['hash'])
                
        for output_tx in tx_details.get('outputs', []):
            output_address = output_tx.get('addresses', [''])[0]
            if output_address:
                # Create output address node
                tx.run("""
                    MERGE (a:Address {address: $address})
                """, address=output_address)
                
                # Create SENT_TO relationship
                tx.run("""
                    MATCH (a:Address {address: $address})
                    MATCH (t:Transaction {tx_hash: $tx_hash})
                    MERGE (t)-[:SENT_TO]->(a)
                """, address=output_address, tx_hash=tx_details['hash'])
                
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
