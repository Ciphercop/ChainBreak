"""
Unit Tests for ChainBreak
Tests core functionality and components
"""

from src.utils import DataValidator, BatchProcessor, PerformanceMonitor
from src.chainbreak import ChainBreak
import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataValidator(unittest.TestCase):
    """Test data validation utilities"""

    def test_validate_bitcoin_address(self):
        """Test Bitcoin address validation"""
        # Valid addresses
        valid_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block
            "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",   # P2SH
            "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"  # Bech32
        ]

        for address in valid_addresses:
            with self.subTest(address=address):
                self.assertTrue(
                    DataValidator.validate_bitcoin_address(address))

        # Invalid addresses
        invalid_addresses = [
            "",  # Empty
            "invalid",  # Invalid format
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa123456789",  # Too long
            "1234567890123456789012345678901234567890",  # Invalid prefix
        ]

        for address in invalid_addresses:
            with self.subTest(address=address):
                self.assertFalse(
                    DataValidator.validate_bitcoin_address(address))

    def test_validate_transaction_hash(self):
        """Test transaction hash validation"""
        # Valid transaction hash
        valid_hash = "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16"
        self.assertTrue(DataValidator.validate_transaction_hash(valid_hash))

        # Invalid hashes
        invalid_hashes = [
            "",  # Empty
            "invalid",  # Invalid format
            "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e1",  # Too short
            "f4184fc596403b9d638783cf57adfe4c75c605f6356fbc91338530e9831e9e16x",  # Invalid char
        ]

        for tx_hash in invalid_hashes:
            with self.subTest(tx_hash=tx_hash):
                self.assertFalse(
                    DataValidator.validate_transaction_hash(tx_hash))

    def test_validate_numeric_value(self):
        """Test numeric value validation"""
        # Valid values
        self.assertTrue(DataValidator.validate_numeric_value(100))
        self.assertTrue(DataValidator.validate_numeric_value(0))
        self.assertTrue(DataValidator.validate_numeric_value(100.5))
        self.assertTrue(DataValidator.validate_numeric_value("100"))

        # Invalid values
        self.assertFalse(DataValidator.validate_numeric_value(-1))
        self.assertFalse(DataValidator.validate_numeric_value("invalid"))
        self.assertFalse(DataValidator.validate_numeric_value(None))


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring utilities"""

    def setUp(self):
        self.monitor = PerformanceMonitor()

    def test_timer_functionality(self):
        """Test timer start/stop functionality"""
        operation = "test_operation"

        # Start timer
        self.monitor.start_timer(operation)

        # End timer
        duration = self.monitor.end_timer(operation)

        # Check that duration is reasonable
        self.assertGreater(duration, 0)
        self.assertLess(duration, 1)  # Should be very fast

    def test_performance_summary(self):
        """Test performance summary generation"""
        # Run some operations
        for i in range(3):
            self.monitor.start_timer(f"op_{i}")
            self.monitor.end_timer(f"op_{i}")

        summary = self.monitor.get_performance_summary()

        # Check summary structure
        self.assertIn('op_0', summary)
        self.assertIn('op_1', summary)
        self.assertIn('op_2', summary)

        # Check summary data
        for op_name in ['op_0', 'op_1', 'op_2']:
            op_data = summary[op_name]
            self.assertIn('count', op_data)
            self.assertIn('average_time', op_data)
            self.assertIn('min_time', op_data)
            self.assertIn('max_time', op_data)
            self.assertEqual(op_data['count'], 1)

    def test_reset_metrics(self):
        """Test metrics reset functionality"""
        # Add some metrics
        self.monitor.start_timer("test_op")
        self.monitor.end_timer("test_op")

        # Verify metrics exist
        summary = self.monitor.get_performance_summary()
        self.assertIn('test_op', summary)

        # Reset metrics
        self.monitor.reset_metrics()

        # Verify metrics are cleared
        summary = self.monitor.get_performance_summary()
        self.assertEqual(len(summary), 0)


class TestBatchProcessor(unittest.TestCase):
    """Test batch processing utilities"""

    def setUp(self):
        # Mock Neo4j driver
        self.mock_driver = type('MockDriver', (), {})()
        self.processor = BatchProcessor(
            self.mock_driver, batch_size=3, max_workers=2)

    def test_batch_processing(self):
        """Test batch processing functionality"""
        items = list(range(10))  # 0-9

        def mock_processor(batch):
            return len(batch)

        results = self.processor.process_transactions_batch(
            items, mock_processor)

        # Check results structure
        self.assertIn('total_processed', results)
        self.assertIn('successful', results)
        self.assertIn('failed', results)
        self.assertIn('processing_time', results)

        # Check processing results
        self.assertEqual(results['total_processed'], 10)
        self.assertEqual(results['successful'], 10)
        self.assertEqual(results['failed'], 0)
        self.assertGreater(results['processing_time'], 0)

    def test_threaded_processing(self):
        """Test threaded processing functionality"""
        items = list(range(5))

        def mock_processor(item):
            return item * 2

        results = self.processor.process_with_threading(items, mock_processor)

        # Check results
        self.assertEqual(results['total_processed'], 5)
        self.assertEqual(results['successful'], 5)
        self.assertEqual(results['failed'], 0)
        self.assertGreater(results['processing_time'], 0)


class TestChainBreakIntegration(unittest.TestCase):
    """Integration tests for ChainBreak (requires Neo4j)"""

    @unittest.skip("Requires Neo4j database")
    def test_chainbreak_initialization(self):
        """Test ChainBreak initialization with valid config"""
        try:
            chainbreak = ChainBreak()
            self.assertIsNotNone(chainbreak)
            self.assertIsNotNone(chainbreak.data_ingestor)
            self.assertIsNotNone(chainbreak.risk_scorer)
            chainbreak.close()
        except Exception as e:
            self.skipTest(f"Neo4j not available: {str(e)}")

    @unittest.skip("Requires Neo4j database")
    def test_system_status(self):
        """Test system status functionality"""
        try:
            chainbreak = ChainBreak()
            status = chainbreak.get_system_status()

            # Check status structure
            self.assertIn('system_status', status)
            self.assertIn('neo4j_connection', status)
            self.assertIn('timestamp', status)

            chainbreak.close()
        except Exception as e:
            self.skipTest(f"Neo4j not available: {str(e)}")


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration structure"""
        from src.chainbreak import ChainBreak

        # Test default config structure
        default_config = ChainBreak._get_default_config(None)

        # Check required sections
        self.assertIn('neo4j', default_config)
        self.assertIn('blockcypher', default_config)
        self.assertIn('analysis', default_config)
        self.assertIn('risk_scoring', default_config)

        # Check Neo4j config
        neo4j_config = default_config['neo4j']
        self.assertIn('uri', neo4j_config)
        self.assertIn('username', neo4j_config)
        self.assertIn('password', neo4j_config)

        # Check risk scoring weights
        risk_config = default_config['risk_scoring']
        self.assertIn('volume_weight', risk_config)
        self.assertIn('layering_weight', risk_config)
        self.assertIn('smurfing_weight', risk_config)

        # Check weights sum to reasonable value
        total_weight = sum(risk_config.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDataValidator,
        TestPerformanceMonitor,
        TestBatchProcessor,
        TestChainBreakIntegration,
        TestConfiguration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
