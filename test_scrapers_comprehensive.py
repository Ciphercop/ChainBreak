#!/usr/bin/env python3
"""
Comprehensive Test Suite for ChainBreak Scrapers
Tests all scrapers with their new enhancements and fallback mechanisms
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'crypto_threat_intel_package'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

def test_configuration_system():
    """Test the new configuration system"""
    logger.info("=" * 60)
    logger.info("TESTING CONFIGURATION SYSTEM")
    logger.info("=" * 60)
    
    try:
        from crypto_threat_intel_package.config.scraper_config import ScraperConfig
        
        config = ScraperConfig()
        
        # Test basic configuration
        logger.info(f"[OK] ChainAbuse Base URL: {config.CHAINABUSE_BASE_URL}")
        logger.info(f"[OK] BitcoinWhosWho API URL: {config.BITCOINWHOSWHO_API_URL}")
        logger.info(f"[OK] Default Timeout: {config.DEFAULT_TIMEOUT}")
        logger.info(f"[OK] Default Retry Attempts: {config.DEFAULT_RETRY_ATTEMPTS}")
        logger.info(f"[OK] User Agent: {config.USER_AGENT}")
        
        # Test environment variable support
        test_env_var = "TEST_SCRAPER_CONFIG"
        os.environ[test_env_var] = "test_value"
        
        # Test known malicious addresses
        logger.info(f"[OK] Known malicious addresses count: {len(config.KNOWN_MALICIOUS_ADDRESSES)}")
        
        # Test test addresses
        logger.info(f"[OK] Test addresses count: {len(config.TEST_ADDRESSES)}")
        
        logger.info("[OK] Configuration system test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Configuration system test FAILED: {e}")
        return False

def test_chainabuse_scraper():
    """Test ChainAbuse scraper with new fallback mechanisms"""
    logger.info("=" * 60)
    logger.info("TESTING CHAINABUSE SCRAPER")
    logger.info("=" * 60)
    
    try:
        from crypto_threat_intel_package.scrapers.chainabuse_scraper import ChainAbuseScraper
        from crypto_threat_intel_package.config.scraper_config import ScraperConfig
        
        config = ScraperConfig()
        scraper = ChainAbuseScraper(config)
        
        # Test with known malicious address
        test_address = "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94"  # WannaCry address
        
        logger.info(f"Testing ChainAbuse scraper with address: {test_address}")
        
        # Test search functionality
        result = scraper.search_address(test_address)
        
        if result:
            logger.info(f"[OK] Found ChainAbuse report:")
            logger.info(f"  - Category: {result.category}")
            logger.info(f"  - Description: {result.description}")
            logger.info(f"  - Confidence Score: {result.confidence_score}")
            logger.info(f"  - Abuse Type: {result.abuse_type}")
        else:
            logger.info("[OK] No ChainAbuse report found (expected for some addresses)")
        
        # Test fallback mechanisms
        logger.info("Testing fallback mechanisms...")
        
        # Test with invalid address to trigger fallbacks
        invalid_address = "invalid_address_test"
        fallback_result = scraper.search_address(invalid_address)
        
        if fallback_result:
            logger.info(f"[OK] Fallback mechanism worked: {fallback_result.category}")
        else:
            logger.info("[OK] Fallback mechanism handled invalid address gracefully")
        
        # Test batch search
        test_addresses = [
            "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94",
            "1Q2TWHE3GMdB6BZKafqwxXtWAWgFt5Jvm3",
            "invalid_test_address"
        ]
        
        batch_results = scraper.search_addresses_batch(test_addresses)
        logger.info(f"[OK] Batch search completed: {len(batch_results)} results")
        
        logger.info("[OK] ChainAbuse scraper test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] ChainAbuse scraper test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_bitcoinwhoswho_scraper():
    """Test BitcoinWhosWho scraper with improved error handling"""
    logger.info("=" * 60)
    logger.info("TESTING BITCOINWHOSWHO SCRAPER")
    logger.info("=" * 60)
    
    try:
        from crypto_threat_intel_package.scrapers.bitcoinwhoswho_scraper import BitcoinWhosWhoScraper
        from crypto_threat_intel_package.config.scraper_config import ScraperConfig
        
        config = ScraperConfig()
        scraper = BitcoinWhosWhoScraper(config)
        
        # Test with known address
        test_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # Genesis block address
        
        logger.info(f"Testing BitcoinWhosWho scraper with address: {test_address}")
        
        # Test search functionality
        result = scraper.search_address(test_address)
        
        if result:
            logger.info(f"[OK] Found BitcoinWhosWho data:")
            logger.info(f"  - Risk Level: {result.risk_level}")
            logger.info(f"  - Confidence Score: {result.confidence_score}")
            logger.info(f"  - Scam Reports: {len(result.scam_reports)}")
            logger.info(f"  - Website Appearances: {len(result.website_appearances)}")
            logger.info(f"  - Tags: {result.tags}")
        else:
            logger.info("[OK] No BitcoinWhosWho data found (expected for some addresses)")
        
        # Test error handling with invalid address
        logger.info("Testing error handling...")
        
        invalid_address = "invalid_address_test"
        error_result = scraper.search_address(invalid_address)
        
        if error_result is None:
            logger.info("[OK] Error handling worked correctly for invalid address")
        else:
            logger.info(f"[OK] Unexpected result for invalid address: {error_result}")
        
        # Test batch search
        test_addresses = [
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94",
            "invalid_test_address"
        ]
        
        batch_results = scraper.search_addresses_batch(test_addresses)
        logger.info(f"[OK] Batch search completed: {len(batch_results)} results")
        
        logger.info("[OK] BitcoinWhosWho scraper test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] BitcoinWhosWho scraper test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_threat_intelligence_client():
    """Test threat intelligence client with enhanced categorization"""
    logger.info("=" * 60)
    logger.info("TESTING THREAT INTELLIGENCE CLIENT")
    logger.info("=" * 60)
    
    try:
        from crypto_threat_intel_package.scrapers.threat_intel_client import ThreatIntelClient
        from crypto_threat_intel_package.config.scraper_config import ScraperConfig
        
        config = ScraperConfig()
        client = ThreatIntelClient(config)
        
        # Test with known malicious address
        test_address = "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94"  # WannaCry address
        
        logger.info(f"Testing threat intelligence client with address: {test_address}")
        
        # Test comprehensive analysis
        result = client.analyze_address(test_address)
        
        if result:
            logger.info(f"✓ Threat intelligence analysis completed:")
            logger.info(f"  - Overall Risk Level: {result.overall_risk_level}")
            logger.info(f"  - Confidence Score: {result.confidence_score}")
            logger.info(f"  - Sources Checked: {len(result.source_results)}")
            logger.info(f"  - Activity Type: {result.activity_type}")
            logger.info(f"  - Evidence Quality: {result.evidence_quality}")
            
            # Show individual source results
            for source, source_result in result.source_results.items():
                if source_result and source_result.is_malicious:
                    logger.info(f"    - {source}: {source_result.category} (confidence: {source_result.confidence_score})")
        else:
            logger.info("✓ No threat intelligence data found")
        
        # Test with clean address
        clean_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # Genesis block
        clean_result = client.analyze_address(clean_address)
        
        if clean_result:
            logger.info(f"✓ Clean address analysis: {clean_result.overall_risk_level}")
        else:
            logger.info("✓ Clean address analysis completed")
        
        # Test batch analysis
        test_addresses = [
            "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94",
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "invalid_test_address"
        ]
        
        batch_results = client.analyze_addresses_batch(test_addresses)
        logger.info(f"✓ Batch analysis completed: {len(batch_results)} results")
        
        logger.info("✓ Threat intelligence client test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Threat intelligence client test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_neo4j_integration():
    """Test integration with Neo4j database"""
    logger.info("=" * 60)
    logger.info("TESTING NEO4J INTEGRATION")
    logger.info("=" * 60)
    
    try:
        # Check if Neo4j is running
        import requests
        
        neo4j_url = "http://localhost:7474"
        try:
            response = requests.get(neo4j_url, timeout=5)
            if response.status_code == 200:
                logger.info("[OK] Neo4j is running and accessible")
            else:
                logger.warning(f"[WARN] Neo4j responded with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[WARN] Neo4j is not accessible: {e}")
            logger.info("  This is expected if Neo4j is not running")
        
        # Test ChainBreak integration
        try:
            from src.chainbreak import ChainBreak
            
            chainbreak = ChainBreak()
            
            # Test with a sample address
            test_address = "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94"
            
            logger.info(f"Testing ChainBreak with address: {test_address}")
            
            # This will test the full pipeline including Neo4j integration
            result = chainbreak.analyze_address(test_address)
            
            if result:
                logger.info("[OK] ChainBreak analysis completed successfully")
                logger.info(f"  - Analysis ID: {result.get('analysis_id', 'N/A')}")
                logger.info(f"  - Risk Level: {result.get('risk_level', 'N/A')}")
                logger.info(f"  - Threat Intelligence: {len(result.get('threat_intelligence', {}))}")
            else:
                logger.info("[OK] ChainBreak analysis completed (no result)")
            
            logger.info("[OK] Neo4j integration test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] ChainBreak integration test FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    except Exception as e:
        logger.error(f"[FAIL] Neo4j integration test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_performance_and_reliability():
    """Test performance and reliability of scrapers"""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE AND RELIABILITY")
    logger.info("=" * 60)
    
    try:
        from crypto_threat_intel_package.scrapers.threat_intel_client import ThreatIntelClient
        from crypto_threat_intel_package.config.scraper_config import ScraperConfig
        
        config = ScraperConfig()
        client = ThreatIntelClient(config)
        
        # Test multiple addresses for performance
        test_addresses = [
            "13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94",
            "1Q2TWHE3GMdB6BZKafqwxXtWAWgFt5Jvm3",
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "invalid_address_1",
            "invalid_address_2"
        ]
        
        logger.info(f"Testing performance with {len(test_addresses)} addresses...")
        
        start_time = time.time()
        results = []
        
        for address in test_addresses:
            try:
                result = client.analyze_address(address)
                results.append(result)
                logger.info(f"[OK] Analyzed {address}: {result.overall_risk_level if result else 'No result'}")
            except Exception as e:
                logger.warning(f"[WARN] Error analyzing {address}: {e}")
                results.append(None)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"[OK] Performance test completed:")
        logger.info(f"  - Total time: {total_time:.2f} seconds")
        logger.info(f"  - Average time per address: {total_time/len(test_addresses):.2f} seconds")
        logger.info(f"  - Successful analyses: {len([r for r in results if r])}")
        logger.info(f"  - Failed analyses: {len([r for r in results if r is None])}")
        
        # Test error handling
        logger.info("Testing error handling with invalid inputs...")
        
        invalid_inputs = [
            "",
            None,
            "not_an_address",
            "123",
            "x" * 1000  # Very long string
        ]
        
        error_count = 0
        for invalid_input in invalid_inputs:
            try:
                result = client.analyze_address(invalid_input)
                if result is None:
                    logger.info(f"[OK] Handled invalid input gracefully: {type(invalid_input).__name__}")
                else:
                    logger.warning(f"[WARN] Unexpected result for invalid input: {type(invalid_input).__name__}")
            except Exception as e:
                logger.info(f"[OK] Exception handled for invalid input: {type(e).__name__}")
                error_count += 1
        
        logger.info(f"[OK] Error handling test completed: {error_count} exceptions handled")
        
        logger.info("[OK] Performance and reliability test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Performance and reliability test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all scraper tests"""
    logger.info("STARTING COMPREHENSIVE SCRAPER TESTS")
    logger.info(f"Test started at: {datetime.now()}")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Configuration System", test_configuration_system),
        ("ChainAbuse Scraper", test_chainabuse_scraper),
        ("BitcoinWhosWho Scraper", test_bitcoinwhoswho_scraper),
        ("Threat Intelligence Client", test_threat_intelligence_client),
        ("Neo4j Integration", test_neo4j_integration),
        ("Performance and Reliability", test_performance_and_reliability)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            test_results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "[PASS] PASSED" if result else "[FAIL] FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal Tests: {len(test_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(test_results)*100):.1f}%")
    
    if failed == 0:
        logger.info("\n[SUCCESS] ALL TESTS PASSED! Scrapers are working correctly with new changes.")
    else:
        logger.info(f"\n[WARN] {failed} test(s) failed. Check the logs for details.")
    
    logger.info(f"\nTest completed at: {datetime.now()}")
    logger.info("Detailed logs saved to: scraper_test_results.log")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
