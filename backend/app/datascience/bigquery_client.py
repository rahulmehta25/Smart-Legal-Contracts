import os
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

class BigQueryManager:
    def __init__(self, credentials_path=None):
        """
        Initialize BigQuery client with optional credentials path
        
        Args:
            credentials_path (str, optional): Path to service account JSON key
        """
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            self.client = bigquery.Client(credentials=credentials)
        else:
            self.client = bigquery.Client()

    def execute_query(self, query, params=None):
        """
        Execute a BigQuery SQL query
        
        Args:
            query (str): SQL query to execute
            params (dict, optional): Query parameters
        
        Returns:
            pandas.DataFrame: Query results
        """
        try:
            # Use query parameters for SQL injection prevention
            if params:
                job_config = bigquery.QueryJobConfig(query_parameters=params)
                query_job = self.client.query(query, job_config=job_config)
            else:
                query_job = self.client.query(query)
            
            # Convert to pandas DataFrame
            return query_job.to_dataframe()
        
        except Exception as e:
            print(f"BigQuery Query Error: {e}")
            return None

    def stream_data(self, dataset_id, table_id, rows):
        """
        Stream data into BigQuery table
        
        Args:
            dataset_id (str): BigQuery dataset ID
            table_id (str): BigQuery table ID
            rows (list): List of dictionaries containing rows to insert
        
        Returns:
            bool: Success or failure of streaming
        """
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        
        try:
            errors = self.client.insert_rows_json(table, rows)
            return len(errors) == 0
        except Exception as e:
            print(f"Streaming Error: {e}")
            return False

    def create_cost_optimized_view(self, view_query, destination_table):
        """
        Create a materialized view with cost optimization
        
        Args:
            view_query (str): SQL query for view
            destination_table (str): Fully qualified table name
        """
        job_config = bigquery.QueryJobConfig(
            destination=destination_table,
            write_disposition='WRITE_TRUNCATE'
        )
        
        query_job = self.client.query(view_query, job_config=job_config)
        query_job.result()  # Wait for the job to complete

# Example usage
if __name__ == '__main__':
    bq_manager = BigQueryManager()
    
    # Example query
    sample_query = """
    SELECT 
        contract_type, 
        AVG(contract_value) as avg_value 
    FROM `arbitration_dataset.contracts`
    GROUP BY contract_type
    """
    
    results = bq_manager.execute_query(sample_query)
    print(results)