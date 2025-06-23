"""
Milvus locustfile example showing usage of three different client types:
1. ORM Client - Maximum flexibility with PyMilvus ORM
2. V2 Client - Simplified MilvusClient v2 SDK  
3. RESTful Client - HTTP-based access

This example demonstrates various Milvus operations including:
- Vector insertions
- Vector searches
- Query operations
- Data deletion
"""

import random
import numpy as np
from locust import task, between
from locust.contrib.milvus import MilvusUser


class MilvusORMLoadTest(MilvusUser):
    """Load test using PyMilvus ORM client"""
    
    # Connection settings
    client_type = "orm"
    host = "http://localhost:19530"
    collection_name = "test_collection"
    
    # Locust settings
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize test data and parameters"""
        self.vector_dim = 128
        self.test_data_size = 100
        
    def _generate_random_vector(self):
        """Generate a random vector for testing"""
        return [random.random() for _ in range(self.vector_dim)]
    
    def _generate_test_records(self, count=10):
        """Generate test records for insertion"""
        records = []
        for i in range(count):
            records.append({
                "id": random.randint(1, 100000),
                "vector": self._generate_random_vector(),
                "text": f"Test document {i}",
                "category": random.choice(["tech", "science", "business"])
            })
        return records
    
    @task(3)
    def search_vectors(self):
        """Perform vector similarity search"""
        query_vector = self._generate_random_vector()
        result = self.search(
            data=[query_vector],
            anns_field="vector",
            top_k=10,
            param={
                "metric_type": "COSINE",
                "params": {"ef": 64}
            },
            output_fields=["id", "text", "category"]
        )
        
        if result.get("success"):
            print(f"ORM Search found {len(result.get('result', []))} results")
    
    @task(2)
    def insert_vectors(self):
        """Insert new vectors into collection"""
        test_records = self._generate_test_records(5)
        result = self.insert(test_records)
        
        if result.get("success"):
            print(f"ORM Inserted {len(test_records)} records successfully")
    
    @task(1)
    def query_by_filter(self):
        """Query vectors using filter conditions"""
        filter_expr = f'category == "tech"'
        result = self.query(
            expr=filter_expr,
            output_fields=["id", "text", "category"]
        )
        
        if result.get("success"):
            print(f"ORM Query found {len(result.get('result', []))} matching records")


class MilvusV2LoadTest(MilvusUser):
    """Load test using MilvusClient v2 SDK"""
    
    # Connection settings
    client_type = "v2"
    host = "http://localhost:19530"
    collection_name = "test_collection_v2"
    
    # Optional authentication for cloud deployment
    # token = "your_api_key_here"
    # db_name = "your_database_name"
    
    # Locust settings
    wait_time = between(1, 2)
    
    def on_start(self):
        """Initialize test data and parameters"""
        self.vector_dim = 768  # Common dimension for embedding models
        
    def _generate_embedding_vector(self):
        """Generate a normalized embedding vector"""
        # Simulate typical embedding distribution
        vector = np.random.normal(0, 0.1, self.vector_dim)
        # Normalize vector
        norm = np.linalg.norm(vector)
        return (vector / norm).tolist() if norm > 0 else vector.tolist()
    
    def _generate_documents(self, count=5):
        """Generate document-like test data"""
        topics = ["artificial intelligence", "machine learning", "deep learning", 
                 "natural language processing", "computer vision", "robotics"]
        
        documents = []
        for i in range(count):
            documents.append({
                "id": random.randint(1, 50000),
                "vector": self._generate_embedding_vector(),
                "text": f"Document about {random.choice(topics)} - entry {i}",
                "metadata": {
                    "topic": random.choice(topics),
                    "timestamp": random.randint(1640995200, 1672531200),
                    "score": round(random.uniform(0.1, 1.0), 3)
                }
            })
        return documents
    
    @task(4)
    def semantic_search(self):
        """Perform semantic search with embeddings"""
        query_vector = self._generate_embedding_vector()
        result = self.search(
            data=[query_vector],
            anns_field="vector",
            top_k=5,
            param={"metric_type": "COSINE"},
            output_fields=["text", "metadata"]
        )
        
        if result.get("success"):
            print(f"V2 Semantic search returned {len(result.get('result', []))} results")
    
    @task(2)
    def batch_insert(self):
        """Batch insert documents with embeddings"""
        documents = self._generate_documents(8)
        result = self.insert(documents)
        
        if result.get("success"):
            print(f"V2 Batch inserted {len(documents)} documents")
    
    @task(1)
    def metadata_query(self):
        """Query documents by metadata"""
        timestamp_filter = f'metadata["timestamp"] > {random.randint(1641000000, 1671000000)}'
        result = self.query(
            expr=timestamp_filter,
            output_fields=["text", "metadata"]
        )
        
        if result.get("success"):
            print(f"V2 Metadata query found {len(result.get('result', []))} documents")


class MilvusRESTLoadTest(MilvusUser):
    """Load test using RESTful API client"""
    
    # Connection settings  
    client_type = "restful"
    host = "http://localhost:19530"
    collection_name = "rest_test_collection"
    
    # REST API authentication
    token = "root:Milvus"  # Default token for local Milvus
    db_name = "default"
    timeout = 30
    
    # Locust settings
    wait_time = between(0.5, 2)
    
    def on_start(self):
        """Initialize REST client specific settings"""
        self.vector_dim = 256
        
    def _generate_feature_vector(self):
        """Generate feature vectors for REST API testing"""
        # Generate sparse-like feature vector
        vector = [0.0] * self.vector_dim
        # Fill random positions with values
        positions = random.sample(range(self.vector_dim), random.randint(10, 30))
        for pos in positions:
            vector[pos] = random.uniform(-1.0, 1.0)
        return vector
    
    def _generate_product_data(self, count=3):
        """Generate product catalog style test data"""
        categories = ["electronics", "clothing", "books", "home", "sports"]
        products = []
        
        for i in range(count):
            products.append({
                "product_id": f"PROD_{random.randint(10000, 99999)}",
                "vector": self._generate_feature_vector(),
                "name": f"Product {i} - {random.choice(['Premium', 'Standard', 'Basic'])}",
                "category": random.choice(categories),
                "price": round(random.uniform(10.0, 1000.0), 2),
                "rating": round(random.uniform(1.0, 5.0), 1)
            })
        return products
        
    @task(3)
    def product_similarity_search(self):
        """Search for similar products using REST API"""
        query_vector = self._generate_feature_vector()
        result = self.search(
            data=[query_vector],
            anns_field="vector", 
            top_k=8,
            param={
                "metric_type": "L2",
                "params": {"nprobe": 16}
            },
            output_fields=["product_id", "name", "category", "price"]
        )
        
        if result.get("success"):
            print(f"REST Product search found {len(result.get('result', []))} similar items")
    
    @task(2)
    def add_products(self):
        """Add new products via REST API"""
        products = self._generate_product_data(4)
        result = self.insert(products)
        
        if result.get("success"):
            print(f"REST Added {len(products)} products to catalog")
    
    @task(1)
    def query_by_category(self):
        """Query products by category and price range"""
        category = random.choice(["electronics", "clothing", "books"])
        price_range = random.choice([100, 200, 500])
        filter_expr = f'category == "{category}" and price < {price_range}'
        
        result = self.query(
            expr=filter_expr,
            output_fields=["product_id", "name", "price", "rating"]
        )
        
        if result.get("success"):
            print(f"REST Category query found {len(result.get('result', []))} products")
    
    @task(1)
    def cleanup_old_products(self):
        """Remove products with low ratings (simulate cleanup)"""
        cleanup_expr = 'rating < 2.0'
        result = self.delete(cleanup_expr)
        
        if result.get("success"):
            print("REST Cleanup operation completed")


# Multi-client load test combining all three approaches
class MilvusMultiClientTest(MilvusUser):
    """Test that randomly switches between different client types"""
    
    # Default to ORM, but will switch during runtime
    client_type = "orm"
    host = "http://localhost:19530"
    collection_name = "multi_client_test"
    
    wait_time = between(1, 4)
    
    def on_start(self):
        """Initialize multi-client test setup"""
        self.vector_dim = 384
        self.client_types = ["orm", "v2", "restful"]
        print(f"Starting multi-client test with {self.client_type} client")
    
    def _switch_client_type(self):
        """Simulate switching client types (for demonstration)"""
        # In real scenarios, you'd create separate user classes
        # This is just for showing the concept
        return random.choice(self.client_types)
    
    def _generate_mixed_vector(self):
        """Generate vectors suitable for different client types"""
        return [random.gauss(0, 0.3) for _ in range(self.vector_dim)]
    
    @task(5)
    def mixed_search_operations(self):
        """Perform search operations simulating different client usage patterns"""
        query_vector = self._generate_mixed_vector()
        
        # Simulate different search parameters based on client type
        if self.client_type == "orm":
            param = {"metric_type": "COSINE", "params": {"ef": 128}}
            top_k = 15
        elif self.client_type == "v2":
            param = {"metric_type": "IP"}
            top_k = 10
        else:  # restful
            param = {"metric_type": "L2", "params": {"nprobe": 32}}
            top_k = 12
            
        result = self.search(
            data=[query_vector],
            anns_field="vector",
            top_k=top_k,
            param=param,
            output_fields=["id", "data"]
        )
        
        if result.get("success"):
            client_label = self.client_type.upper()
            print(f"{client_label} Mixed search completed with {len(result.get('result', []))} results")
    
    @task(2) 
    def mixed_insert_operations(self):
        """Insert data with patterns suitable for different clients"""
        count = random.randint(2, 6)
        test_data = []
        
        for i in range(count):
            test_data.append({
                "id": random.randint(1, 100000),
                "vector": self._generate_mixed_vector(),
                "data": f"Mixed client data {i} from {self.client_type}",
                "client_type": self.client_type,
                "timestamp": random.randint(1640000000, 1672000000)
            })
        
        result = self.insert(test_data)
        
        if result.get("success"):
            print(f"{self.client_type.upper()} Mixed insert of {len(test_data)} records completed")
