#!/usr/bin/env python3
"""
MongoDB Metrics Calculator

This script calculates MERIT metrics for data stored in MongoDB collections
and generates comprehensive reports with visualizations.

Usage:
    python calculate_mongodb_metrics.py [options]

Examples:
    # Basic usage with default config
    python calculate_mongodb_metrics.py
    
    # Process specific number of documents
    python calculate_mongodb_metrics.py --limit 100
    
    # Override specific metrics
    python calculate_mongodb_metrics.py --metrics correctness,faithfulness,relevance
    
    # Custom output filename
    python calculate_mongodb_metrics.py --output my_report.html
    
    # Dry run to validate configuration
    python calculate_mongodb_metrics.py --dry-run
"""

import argparse
import sys
import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the merit package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from merit.storage.mongodb_storage import MongoDBStorage
from merit.core.models import Input, Response
from merit.metrics.rag import (
    CorrectnessMetric, FaithfulnessMetric, RelevanceMetric, 
    CoherenceMetric, FluencyMetric
)
from merit.core.logging import get_logger

# Import configuration
try:
    from examples.mongodb.mongodb_metrics_config import (
        LLM_CLIENT, MONGODB_CONFIG, COLLECTION_CONFIG, ENABLED_METRICS,
        BATCH_SIZE, RESULTS_CONFIG, REPORT_CONFIG,
        validate_document, extract_field_value
    )
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Make sure mongodb_metrics_config.py is in the same directory as this script.")
    sys.exit(1)

logger = get_logger(__name__)


class MongoDBMetricsCalculator:
    """
    Main class for calculating MERIT metrics on MongoDB data.
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config_override = config_override or {}
        self.storage = None
        self.metrics = {}
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        
        # Initialize components
        self._initialize_storage()
        self._initialize_metrics()
    
    def _initialize_storage(self):
        """Initialize MongoDB storage connection."""
        try:
            # Apply any config overrides
            mongodb_config = MONGODB_CONFIG.copy()
            if 'mongodb_config' in self.config_override:
                mongodb_config.update(self.config_override['mongodb_config'])
            
            self.storage = MongoDBStorage(mongodb_config)
            logger.info(f"Connected to MongoDB: {mongodb_config['database']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB storage: {e}")
            raise
    
    def _initialize_metrics(self):
        """Initialize the metrics to be calculated."""
        try:
            # Get enabled metrics from config or override
            enabled_metrics = self.config_override.get('metrics', ENABLED_METRICS)
            
            # Initialize metric instances
            metric_classes = {
                'correctness': CorrectnessMetric,
                'faithfulness': FaithfulnessMetric,
                'relevance': RelevanceMetric,
                'coherence': CoherenceMetric,
                'fluency': FluencyMetric
            }
            
            for metric_name in enabled_metrics:
                if metric_name in metric_classes:
                    # Initialize metric for evaluation context
                    metric_class = metric_classes[metric_name]
                    self.metrics[metric_name] = metric_class.for_evaluation()
                    
                    # Set LLM client if the metric needs it
                    if hasattr(self.metrics[metric_name], 'llm_client'):
                        self.metrics[metric_name].llm_client = LLM_CLIENT
                    
                    logger.info(f"Initialized metric: {metric_name}")
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
            
            if not self.metrics:
                raise ValueError("No valid metrics configured")
                
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            raise
    
    def validate_configuration(self) -> bool:
        """
        Validate the configuration and test connections.
        
        Returns:
            True if configuration is valid
        """
        try:
            logger.info("Validating configuration...")
            
            # Test MongoDB connection
            collections = self.storage.list_collections()
            collection_name = COLLECTION_CONFIG['collection_name']
            
            if collection_name not in collections:
                logger.error(f"Collection '{collection_name}' not found in database")
                logger.info(f"Available collections: {collections}")
                return False
            
            # Test document structure with a sample
            sample_doc = self.storage.find_one(
                COLLECTION_CONFIG.get('query_filter', {}),
                collection_name=collection_name
            )
            
            if not sample_doc:
                logger.error(f"No documents found in collection '{collection_name}'")
                return False
            
            # Validate field mapping
            field_mapping = COLLECTION_CONFIG['field_mapping']
            if not validate_document(sample_doc, field_mapping):
                logger.error("Sample document does not match field mapping configuration")
                logger.info(f"Sample document keys: {list(sample_doc.keys())}")
                return False
            
            # Test LLM client if configured
            if LLM_CLIENT:
                try:
                    test_response = LLM_CLIENT.generate_text("Test connection")
                    logger.info("LLM client connection successful")
                except Exception as e:
                    logger.warning(f"LLM client test failed: {e}")
                    logger.warning("Metrics requiring LLM evaluation may fail")
            
            logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def process_documents(self, limit: Optional[int] = None, dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Process documents from MongoDB and calculate metrics.
        
        Args:
            limit: Maximum number of documents to process
            dry_run: If True, only validate without calculating metrics
            
        Returns:
            List of results
        """
        try:
            collection_name = COLLECTION_CONFIG['collection_name']
            field_mapping = COLLECTION_CONFIG['field_mapping']
            query_filter = COLLECTION_CONFIG.get('query_filter', {})
            sort_order = COLLECTION_CONFIG.get('sort_order', [])
            
            # Apply limit from parameter or config
            doc_limit = limit or COLLECTION_CONFIG.get('limit')
            
            logger.info(f"Processing documents from collection: {collection_name}")
            if doc_limit:
                logger.info(f"Limit: {doc_limit} documents")
            
            # Get documents from MongoDB
            documents = self.storage.find(
                query=query_filter,
                limit=doc_limit or 1000,  # Default reasonable limit
                sort=sort_order,
                collection_name=collection_name
            )
            
            if not documents:
                logger.warning("No documents found matching the query")
                return []
            
            logger.info(f"Found {len(documents)} documents to process")
            
            if dry_run:
                logger.info("Dry run mode - validating document structure only")
                valid_count = 0
                for doc in documents[:10]:  # Check first 10 documents
                    if validate_document(doc, field_mapping):
                        valid_count += 1
                logger.info(f"Validated {valid_count}/10 sample documents")
                return []
            
            # Process documents in batches
            batch_size = self.config_override.get('batch_size', BATCH_SIZE)
            results = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_results = self._process_batch(batch, field_mapping)
                results.extend(batch_results)
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_results)} results")
            
            self.results = results
            logger.info(f"Processing complete. Total results: {len(results)}")
            logger.info(f"Processed: {self.processed_count}, Errors: {self.error_count}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def _process_batch(self, documents: List[Dict[str, Any]], 
                      field_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents.
        
        Args:
            documents: List of MongoDB documents
            field_mapping: Field mapping configuration
            
        Returns:
            List of results for the batch
        """
        batch_results = []
        
        for doc in documents:
            try:
                result = self._process_single_document(doc, field_mapping)
                if result:
                    batch_results.append(result)
                    self.processed_count += 1
                else:
                    self.error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                self.error_count += 1
        
        return batch_results
    
    def _process_single_document(self, doc: Dict[str, Any], 
                                field_mapping: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single document and calculate metrics.
        
        Args:
            doc: MongoDB document
            field_mapping: Field mapping configuration
            
        Returns:
            Result dictionary or None if processing failed
        """
        try:
            # Validate document
            if not validate_document(doc, field_mapping):
                logger.warning(f"Document {doc.get('_id')} failed validation")
                return None
            
            # Extract fields using mapping
            extracted_data = {}
            for merit_field, mapping in field_mapping.items():
                value = extract_field_value(doc, mapping)
                extracted_data[merit_field] = value
            
            # Create MERIT objects
            input_obj = Input(content=extracted_data.get('user_input', ''))
            
            response_obj = Response(
                content=extracted_data.get('bot_response', ''),
                documents=extracted_data.get('context_documents', [])
            )
            
            # Create reference response if available
            reference_obj = None
            if extracted_data.get('reference_answer'):
                reference_obj = Response(content=extracted_data['reference_answer'])
            
            # Calculate metrics
            metric_results = []
            for metric_name, metric in self.metrics.items():
                try:
                    result = metric.calculate_evaluation(
                        input_obj=input_obj,
                        response=response_obj,
                        reference=reference_obj,
                        llm_client=LLM_CLIENT
                    )
                    
                    # Format result for the report template
                    metric_result = {
                        "metric_name": metric_name,
                        "value": result.get("value", 0.0),
                        "explanation": result.get("explanation", ""),
                        "method": result.get("method", ""),
                        "timestamp": result.get("timestamp", datetime.now().isoformat()),
                        "raw_llm_response": result.get("raw_llm_response", ""),
                        "custom_measurements": result.get("metadata", {})
                    }
                    metric_results.append(metric_result)
                    
                except Exception as e:
                    logger.error(f"Error calculating {metric_name} for document {doc.get('_id')}: {e}")
                    metric_results.append({
                        "metric_name": metric_name,
                        "value": 0.0,
                        "explanation": f"Error: {str(e)}",
                        "method": "error",
                        "timestamp": datetime.now().isoformat(),
                        "raw_llm_response": "",
                        "custom_measurements": {}
                    })
            
            # Prepare result in format expected by the HTML template
            result = {
                "input": {
                    "id": str(doc.get('_id', '')),
                    "content": extracted_data.get('user_input', '')
                },
                "response": {
                    "content": extracted_data.get('bot_response', ''),
                    "documents": extracted_data.get('context_documents', [])
                },
                "reference": {
                    "content": extracted_data.get('reference_answer', '')
                } if extracted_data.get('reference_answer') else None,
                "metrics": metric_results,
                "metadata": {
                    "document_id": str(doc.get('_id', '')),
                    "document_content": extracted_data.get('context_documents', []),
                    **{key: extracted_data.get(key) for key in field_mapping.keys() 
                       if key not in ['user_input', 'bot_response', 'context_documents', 'reference_answer']}
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    def save_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Save results to MongoDB.
        
        Args:
            results: List of metric results
            
        Returns:
            True if successful
        """
        try:
            if not results:
                logger.warning("No results to save")
                return True
            
            collection_name = COLLECTION_CONFIG['collection_name']
            results_collection = f"{collection_name}{RESULTS_CONFIG['collection_suffix']}"
            
            logger.info(f"Saving {len(results)} results to collection: {results_collection}")
            
            # Check if we should overwrite existing results
            if not RESULTS_CONFIG.get('overwrite_existing', False):
                # Filter out documents that already have results
                existing_ids = set()
                existing_docs = self.storage.find(
                    query={},
                    collection_name=results_collection
                )
                existing_ids = {doc.get('document_id') for doc in existing_docs}
                
                original_count = len(results)
                results = [r for r in results if r.get('metadata', {}).get('document_id') not in existing_ids]
                
                if len(results) < original_count:
                    logger.info(f"Filtered out {original_count - len(results)} existing results")
            
            if results:
                # Convert results to storage format
                storage_results = []
                for result in results:
                    storage_result = {
                        "document_id": result.get('metadata', {}).get('document_id'),
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {metric['metric_name']: metric for metric in result['metrics']},
                        "extracted_data": {
                            "user_input": result['input']['content'],
                            "bot_response": result['response']['content'],
                            "context_documents_count": len(result['response'].get('documents', [])),
                        },
                        "metadata": result.get('metadata', {})
                    }
                    
                    # Include original document if configured
                    if RESULTS_CONFIG.get('include_original_doc', False):
                        storage_result['original_document'] = result.get('original_document')
                    
                    storage_results.append(storage_result)
                
                # Insert results
                inserted_ids = self.storage.insert_many(storage_results, collection_name=results_collection)
                logger.info(f"Successfully saved {len(inserted_ids)} results")
                return True
            else:
                logger.info("No new results to save")
                return True
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def generate_report(self, results: List[Dict[str, Any]], 
                       output_filename: Optional[str] = None) -> str:
        """
        Generate an HTML report from the results using the existing template.
        
        Args:
            results: List of metric results
            output_filename: Optional custom output filename
            
        Returns:
            Path to the generated report
        """
        try:
            if not results:
                logger.warning("No results to generate report from")
                return ""
            
            # Read the HTML template
            template_path = Path(__file__).parent.parent / "merit" / "templates" / "report_template.html"
            
            with open(template_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
            
            # Prepare report data
            report_data = {
                "results": results,
                "metadata": {
                    "title": REPORT_CONFIG.get('report_title', 'MongoDB Metrics Analysis Report'),
                    "timestamp": datetime.now().isoformat(),
                    "total_documents": len(results),
                    "processed_count": self.processed_count,
                    "error_count": self.error_count,
                    "collection": COLLECTION_CONFIG['collection_name'],
                    "enabled_metrics": list(self.metrics.keys()),
                    "batch_size": BATCH_SIZE
                }
            }
            
            # Embed the data into the HTML template
            data_script = f"const REPORT_DATA = {json.dumps(report_data, indent=2)};"
            
            # Replace the placeholder data script
            html_content = html_template.replace(
                "const REPORT_DATA = {};",
                data_script
            )
            
            # Update the title
            html_content = html_content.replace(
                "<title>MERIT Evaluation Report</title>",
                f"<title>{report_data['metadata']['title']}</title>"
            )
            
            # Write the report
            output_file = output_filename or REPORT_CONFIG.get('report_filename', 'metrics_report.html')
            output_path = Path(output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated: {output_path.absolute()}")
            return str(output_path.absolute())
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""


def main():
    """Main function to run the MongoDB metrics calculator."""
    parser = argparse.ArgumentParser(
        description="Calculate MERIT metrics for MongoDB data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_mongodb_metrics.py
  python calculate_mongodb_metrics.py --limit 100
  python calculate_mongodb_metrics.py --metrics correctness,faithfulness
  python calculate_mongodb_metrics.py --output my_report.html --dry-run
        """
    )
    
    parser.add_argument(
        '--limit', type=int,
        help='Maximum number of documents to process'
    )
    
    parser.add_argument(
        '--metrics',
        help='Comma-separated list of metrics to calculate (overrides config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output filename for the report'
    )
    
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate configuration without calculating metrics'
    )
    
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save results to MongoDB'
    )
    
    parser.add_argument(
        '--no-report', action='store_true',
        help='Do not generate HTML report'
    )
    
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger('merit').setLevel(logging.DEBUG)
    
    try:
        # Prepare configuration overrides
        config_override = {}
        
        if args.metrics:
            config_override['metrics'] = [m.strip() for m in args.metrics.split(',')]
        
        # Initialize calculator
        calculator = MongoDBMetricsCalculator(config_override)
        
        # Validate configuration
        if not calculator.validate_configuration():
            logger.error("Configuration validation failed. Please check your settings.")
            return 1
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0
        
        # Process documents
        results = calculator.process_documents(limit=args.limit)
        
        if not results:
            logger.warning("No results generated")
            return 1
        
        # Save results to MongoDB
        if not args.no_save:
            success = calculator.save_results(results)
            if not success:
                logger.warning("Failed to save results to MongoDB")
        
        # Generate report
        if not args.no_report:
            report_path = calculator.generate_report(results, args.output)
            if report_path:
                print(f"\nReport generated: {report_path}")
                print(f"Open the report in your browser to view the results.")
            else:
                logger.warning("Failed to generate report")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Processed: {calculator.processed_count} documents")
        print(f"  Errors: {calculator.error_count} documents")
        print(f"  Metrics calculated: {list(calculator.metrics.keys())}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
