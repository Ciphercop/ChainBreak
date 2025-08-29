#!/usr/bin/env python3
"""
Basic ChainBreak Analysis Example
Demonstrates how to use ChainBreak for blockchain analysis
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.chainbreak import ChainBreak
from src.utils import DataValidator


def main():
    """Main example function"""
    print("🔗 ChainBreak Basic Analysis Example")
    print("=" * 50)
    
    # Example Bitcoin addresses to analyze
    test_addresses = [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",   # Another example
    ]
    
    try:
        # Initialize ChainBreak
        print("Initializing ChainBreak...")
        chainbreak = ChainBreak()
        
        # Check system status
        print("\nChecking system status...")
        status = chainbreak.get_system_status()
        print(f"✅ System Status: {status['system_status']}")
        print(f"🔌 Neo4j Connection: {status['neo4j_connection']}")
        
        if status['system_status'] != 'operational':
            print("❌ System not operational. Please check Neo4j connection.")
            return
        
        # Analyze each address
        for address in test_addresses:
            print(f"\n{'='*60}")
            print(f"🔍 Analyzing Address: {address}")
            print(f"{'='*60}")
            
            # Validate address
            if not DataValidator.validate_bitcoin_address(address):
                print(f"❌ Invalid Bitcoin address: {address}")
                continue
            
            # Run analysis
            print("🚀 Starting analysis...")
            results = chainbreak.analyze_address(address, generate_visualizations=False)
            
            if 'error' in results:
                print(f"❌ Analysis failed: {results['error']}")
                continue
            
            # Display results
            print("✅ Analysis completed successfully!")
            print(f"\n📊 Results Summary:")
            print(f"  Address: {results['address']}")
            print(f"  Blockchain: {results['blockchain']}")
            print(f"  Risk Level: {results['risk_score']['risk_level']}")
            print(f"  Risk Score: {results['risk_score']['total_risk_score']:.3f}")
            print(f"  Total Anomalies: {results['summary']['total_anomalies']}")
            print(f"  Layering Patterns: {results['summary']['layering_count']}")
            print(f"  Smurfing Patterns: {results['summary']['smurfing_count']}")
            print(f"  Volume Anomalies: {results['summary']['volume_anomaly_count']}")
            
            # Display risk factors
            if 'risk_details' in results['risk_score']:
                print(f"\n🔍 Risk Factor Details:")
                risk_details = results['risk_score']['risk_details']
                for factor, score in risk_details.items():
                    if score > 0:
                        print(f"  {factor.replace('_', ' ').title()}: {score:.3f}")
            
            # Display recommendations
            if results['summary']['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in results['summary']['recommendations']:
                    print(f"  • {rec}")
        
        # Generate comparative analysis
        print(f"\n{'='*60}")
        print("📈 Comparative Analysis")
        print(f"{'='*60}")
        
        print("Running comparative analysis...")
        comparative_results = chainbreak.analyze_multiple_addresses(test_addresses)
        
        if 'error' not in comparative_results:
            risk_summary = comparative_results['risk_summary']
            print(f"\n📊 Risk Summary:")
            print(f"  Total Addresses: {risk_summary['total_addresses']}")
            print(f"  Average Risk Score: {risk_summary['average_risk_score']:.3f}")
            print(f"  High Risk Addresses: {len(risk_summary['high_risk_addresses'])}")
            
            print(f"\n📊 Risk Distribution:")
            for level, count in risk_summary['risk_distribution'].items():
                if count > 0:
                    print(f"  {level}: {count}")
        
        # Export example
        print(f"\n{'='*60}")
        print("📤 Export Example")
        print(f"{'='*60}")
        
        export_choice = input("Export network to Gephi format? (y/n): ").lower().strip()
        if export_choice in ['y', 'yes']:
            print("Exporting to Gephi...")
            export_file = chainbreak.export_network_to_gephi(test_addresses[0])
            if export_file:
                print(f"✅ Network exported to: {export_file}")
            else:
                print("❌ Export failed")
        
        print(f"\n{'='*60}")
        print("🎉 Example completed successfully!")
        print("Check the generated log files for detailed information.")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during example: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'chainbreak' in locals():
            chainbreak.close()


if __name__ == '__main__':
    main()
