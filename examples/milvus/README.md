# Milvus Locust Performance Testing Guide

This example demonstrates how to use Locust for performance testing of Milvus vector databases, supporting three different client types to meet testing needs in various scenarios.

## 🚀 Features

- **Three Client Types Support**: ORM, MilvusClient v2, RESTful API
- **Complete Vector Database Operations**: Insert, Search, Query, Delete
- **Real-world Scenario Simulation**: Document retrieval, product recommendation, semantic search
- **Performance Statistics**: Detailed response time and success rate statistics
- **Smart Retry Mechanism**: Handle memory limits, service denials and other situations

## 📋 Prerequisites

### System Dependencies
```bash
# Python 3.8+
pip install locust
pip install pymilvus>=2.4.0
pip install numpy
pip install requests
```

### Milvus Server
Ensure Milvus server is running:
```bash
# Start Milvus using Docker (recommended)
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest standalone

# Or use Docker Compose
# Reference: https://milvus.io/docs/install_standalone-docker.md
```

## 🎯 Three Client Types

### 1. PyMilvus ORM Client
**Use Cases**: Complex applications requiring maximum control and performance

**Features**:
- Direct gRPC connection with lowest latency
- Fine-grained connection and resource management
- Support for advanced search and batch operations
- 15-20% higher throughput

**Example**:
```python
class MilvusORMLoadTest(MilvusUser):
    client_type = "orm"
    host = "http://localhost:19530"
    collection_name = "test_collection"
```

### 2. MilvusClient v2 Client
**Use Cases**: Standard AI applications, RAG systems, semantic search

**Features**:
- Unified cross-language API
- Built-in embedding function support
- Smart default parameters and schema caching
- Native async support with 3-5x concurrency performance improvement

**Example**:
```python
class MilvusV2LoadTest(MilvusUser):
    client_type = "v2"
    host = "http://localhost:19530"
    collection_name = "test_collection_v2"
    token = "your_api_key"  # Optional for cloud deployment
    db_name = "your_db"     # Optional
```

### 3. RESTful API Client
**Use Cases**: Microservices architecture, multi-language integration, cloud-native applications

**Features**:
- Standard HTTP protocol, language-agnostic
- Easy integration with monitoring and load balancing
- Support for horizontal scaling
- Add 10-20ms latency but with highest compatibility

**Example**:
```python
class MilvusRESTLoadTest(MilvusUser):
    client_type = "restful"
    host = "http://localhost:19530"
    collection_name = "rest_test_collection"
    token = "root:Milvus"   # Authentication token
    timeout = 30            # Request timeout
```

## 🛠️ Usage

### 1. Prepare Test Collection

Before running tests, create the test collection first:

```python
# Example script to create test collection
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
]

schema = CollectionSchema(fields, "Test collection for performance testing")
collection = Collection("test_collection", schema)

# Create index
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("vector", index_params)
collection.load()
```

### 2. Run Single Client Tests

```bash
# Test ORM client
locust -f locustfile.py --host http://localhost:19530 MilvusORMLoadTest

# Test V2 client
locust -f locustfile.py --host http://localhost:19530 MilvusV2LoadTest

# Test RESTful client
locust -f locustfile.py --host http://localhost:19530 MilvusRESTLoadTest
```

### 3. Run Mixed Client Tests

```bash
# Test multiple clients simultaneously
locust -f locustfile.py --host http://localhost:19530 MilvusMultiClientTest
```

### 4. Web UI Testing
Visit http://localhost:8089 to use Locust Web interface for test configuration and monitoring.

## ⚙️ Configuration Options

### General Configuration
```python
class MyMilvusTest(MilvusUser):
    client_type = "orm"  # "orm", "v2", "restful"
    host = "http://localhost:19530"
    collection_name = "my_collection"
    wait_time = between(1, 3)  # Request interval
```

### ORM Client Configuration
```python
class MyORMTest(MilvusUser):
    client_type = "orm"
    alias = "production"  # Connection alias
    # Other PyMilvus connection parameters
```

### V2 Client Configuration
```python
class MyV2Test(MilvusUser):
    client_type = "v2"
    token = "api_key_from_console"  # Zilliz Cloud API key
    db_name = "production"          # Database name
    timeout = 60                    # Connection timeout
```

### RESTful Client Configuration
```python
class MyRESTTest(MilvusUser):
    client_type = "restful"
    token = "root:Milvus"          # Authentication token
    db_name = "default"            # Database name
    timeout = 30                   # HTTP timeout
```

## 📊 Performance Testing Recommendations

### 1. Baseline Testing
```bash
# Basic performance testing
locust -f locustfile.py --headless -u 10 -r 2 -t 300s --host http://localhost:19530
```

### 2. Stress Testing
```bash
# High concurrency stress testing
locust -f locustfile.py --headless -u 100 -r 10 -t 600s --host http://localhost:19530
```

### 3. Sustained Load Testing
```bash
# Long-term sustained testing
locust -f locustfile.py --headless -u 50 -r 5 -t 3600s --host http://localhost:19530
```

### 4. Performance Comparison Testing
Run three client tests separately and compare:
- **Response Time**: Average response time and P95/P99 latency
- **Throughput**: Requests per second (RPS)
- **Success Rate**: Request success percentage
- **Resource Usage**: CPU and memory consumption

## 🔍 Monitoring and Debugging

### View Detailed Logs
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Custom Statistics
```python
@task
def custom_search(self):
    start_time = time.time()
    result = self.search(...)
    
    # Record custom metrics
    if result.get("empty"):
        events.request.fire(
            request_type="milvus",
            name="empty_search_result",
            response_time=int((time.time() - start_time) * 1000),
            response_length=0
        )
```

### Performance Optimization Tips

1. **Adjust Vector Dimensions**: Choose appropriate vector dimensions based on actual applications
2. **Optimize Batch Size**: Balance memory usage and throughput
3. **Index Parameter Tuning**: Adjust HNSW parameters based on data characteristics
4. **Connection Pool Configuration**: Set reasonable connection count and timeout values

## 🎯 Real-world Application Scenarios

### 1. Document Retrieval System
```python
# Document vector search performance testing
@task
def document_search(self):
    query_vector = self._generate_embedding_vector()
    self.search(
        data=[query_vector],
        anns_field="document_vector",
        top_k=20,
        param={"metric_type": "COSINE"},
        output_fields=["title", "content", "metadata"]
    )
```

### 2. Recommendation System
```python
# Product recommendation performance testing
@task  
def product_recommendation(self):
    user_vector = self._generate_user_profile_vector()
    self.search(
        data=[user_vector],
        anns_field="product_vector",
        top_k=10,
        param={"metric_type": "IP"},
        output_fields=["product_id", "name", "price"]
    )
```

### 3. Image Similarity Search
```python
# Image feature search performance testing
@task
def image_similarity_search(self):
    image_vector = self._generate_image_feature_vector()
    self.search(
        data=[image_vector],
        anns_field="image_vector", 
        top_k=15,
        param={"metric_type": "L2"},
        output_fields=["image_id", "url", "tags"]
    )
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Solution: Check Milvus service status and network connection
   docker ps | grep milvus
   telnet localhost 19530
   ```

2. **Collection Not Found**
   ```
   Solution: Ensure test collection is created and loaded
   ```

3. **Out of Memory**
   ```
   Solution: Reduce batch size or increase system memory
   ```

4. **Authentication Failed** (Cloud deployment)
   ```
   Solution: Check if token and database name are correct
   ```

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.getLogger("pymilvus").setLevel(logging.DEBUG)

# Check connection status
from pymilvus import connections
print(connections.list_connections())

# Verify collection status
collection = Collection("test_collection")
print(f"Collection loaded: {collection.is_loaded}")
print(f"Collection entities: {collection.num_entities}")
```

## 📈 Performance Benchmark Reference

Performance reference values based on typical hardware configuration:

| Client Type | Search QPS | Insert TPS | P95 Latency |
|-------------|------------|------------|-------------|
| ORM         | 1200-1500  | 800-1000   | 25-35ms     |
| V2          | 1000-1200  | 600-800    | 30-40ms     |
| RESTful     | 800-1000   | 400-600    | 45-60ms     |

*Note: Actual performance depends on hardware configuration, network environment, and data characteristics*

## 🤝 Contributing

Issues and Pull Requests are welcome to improve this performance testing tool!

## 📄 License

This project follows the Apache 2.0 license.
