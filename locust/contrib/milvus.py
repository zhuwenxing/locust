import logging
import time
import json
import requests
from typing import Dict, Any, List, Optional
from pymilvus import Collection, connections, MilvusClient

# Abstract base class for Milvus clients
class MilvusClientBase:
    """Abstract base for all Milvus client types"""
    def insert(self, data):
        raise NotImplementedError
    def upsert(self, data):
        raise NotImplementedError
    def search(self, data, anns_field, top_k, param=None, output_fields=None, search_type=None):
        raise NotImplementedError
    def query(self, expr, output_fields=None, expr_type=None):
        raise NotImplementedError
    def delete(self, expr):
        raise NotImplementedError
    def close(self):
        raise NotImplementedError

class MilvusORMClient(MilvusClientBase):
    """Milvus ORM Client Wrapper"""
    def __init__(self, host, collection_name, **kwargs):
        self.host = host
        self.collection_name = collection_name
        self.kwargs = kwargs
        self.alias = kwargs.get("alias", "default")
        self._connect()
        self.collection = Collection(self.collection_name)
        self.logger = logging.getLogger(__name__)
        self.sleep_time = 1
    
    def _connect(self):
        connections.connect(uri=self.host, alias=self.alias)
    
    def close(self):
        connections.disconnect(alias=self.alias)
    
    def insert(self, data):
        start = time.time()
        try:
            result = self.collection.insert(data)
            self.collection.flush()  # Ensure data consistency
            total_time = (time.time() - start) * 1000
            return {"success": True, "response_time": total_time, "result": result}
        except Exception as e:
            if "memory" in str(e).lower() or "deny" in str(e).lower() or "limit" in str(e).lower():
                time.sleep(self.sleep_time)
                self.sleep_time = min(self.sleep_time * 2, 60)  # Cap at 60 seconds
                return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
            else:
                return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def upsert(self, data):
        start = time.time()
        try:
            result = self.collection.upsert(data)
            self.collection.flush()  # Ensure data consistency
            total_time = (time.time() - start) * 1000
            return {"success": True, "response_time": total_time, "result": result}
        except Exception as e:
            if "memory" in str(e).lower() or "deny" in str(e).lower() or "limit" in str(e).lower():
                time.sleep(self.sleep_time)
                self.sleep_time = min(self.sleep_time * 2, 60)
                return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
            else:
                return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def search(self, data, anns_field, top_k, param=None, output_fields=None, search_type=None):
        if search_type is None:
            search_type = "dense-search"
        if param is None:
            param = {"metric_type": "COSINE", "params": {"ef": 64}}
        if output_fields is None:
            output_fields = ["id"]
        
        start = time.time()
        try:
            # Load collection if not loaded
            if not self.collection.has_index():
                self.logger.warning(f"Collection {self.collection_name} has no index")
            
            res = self.collection.search(
                data=data,
                anns_field=anns_field,
                param=param,
                limit=top_k,
                output_fields=output_fields,
            )
            total_time = (time.time() - start) * 1000
            
            # Convert SearchResult to list and check if empty
            results = []
            if hasattr(res, '__iter__'):
                for hits in res:
                    results.append([hit for hit in hits])
            else:
                # Handle single result
                results = [[hit for hit in res]] if res else []
            
            empty = all(len(r) == 0 for r in results)
            return {"success": not empty, "response_time": total_time, "empty": empty, "result": results}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def query(self, expr, output_fields=None, expr_type=None):
        if expr_type is None:
            expr_type = "text_match"
        if output_fields is None:
            output_fields = ["id"]
        
        start = time.time()
        try:
            res = self.collection.query(expr=expr, output_fields=output_fields)
            total_time = (time.time() - start) * 1000
            empty = len(res) == 0
            return {"success": not empty, "response_time": total_time, "empty": empty, "result": res}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def delete(self, expr):
        start = time.time()
        try:
            result = self.collection.delete(expr)
            self.collection.flush()
            total_time = (time.time() - start) * 1000
            return {"success": True, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}

class MilvusV2Client(MilvusClientBase):
    """Milvus v2 Python SDK Client Wrapper"""
    def __init__(self, host, collection_name, **kwargs):
        self.host = host
        self.collection_name = collection_name
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
        # Initialize MilvusClient v2
        db_name = kwargs.get("db_name", "default")
        token = kwargs.get("token")  # Can be None
        timeout = kwargs.get("timeout", 30)
        
        client_kwargs = {
            "uri": host,
            "db_name": db_name,
            "timeout": timeout
        }
        
        # Only add token if it's provided
        if token is not None:
            client_kwargs["token"] = token
            
        self.client = MilvusClient(**client_kwargs)
        
        # Load collection to ensure it's ready
        try:
            self.client.load_collection(collection_name=self.collection_name)
        except Exception as e:
            self.logger.warning(f"Failed to load collection {self.collection_name}: {e}")
    
    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()
    
    def insert(self, data):
        start = time.time()
        try:
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            total_time = (time.time() - start) * 1000
            return {"success": True, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def upsert(self, data):
        start = time.time()
        try:
            result = self.client.upsert(
                collection_name=self.collection_name,
                data=data
            )
            total_time = (time.time() - start) * 1000
            return {"success": True, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def search(self, data, anns_field, top_k, param=None, output_fields=None, search_type=None):
        if search_type is None:
            search_type = "dense-search"
        if param is None:
            param = {"metric_type": "COSINE", "params": {"ef": 64}}
        if output_fields is None:
            output_fields = ["id"]
        
        start = time.time()
        try:
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": data,
                "limit": top_k,
                "output_fields": output_fields
            }
            
            # Add search_params if available
            if isinstance(param, dict) and "params" in param:
                search_kwargs["search_params"] = param["params"]
            
            # Add filter if provided
            filter_value = self.kwargs.get("filter")
            if filter_value:
                search_kwargs["filter"] = filter_value
            
            result = self.client.search(**search_kwargs)
            total_time = (time.time() - start) * 1000
            empty = len(result) == 0 or all(len(r) == 0 for r in result)
            return {"success": not empty, "response_time": total_time, "empty": empty, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def query(self, expr, output_fields=None, expr_type=None):
        if expr_type is None:
            expr_type = "text_match"
        if output_fields is None:
            output_fields = ["id"]
        
        start = time.time()
        try:
            result = self.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=output_fields
            )
            total_time = (time.time() - start) * 1000
            empty = len(result) == 0
            return {"success": not empty, "response_time": total_time, "empty": empty, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def delete(self, expr):
        start = time.time()
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=expr
            )
            total_time = (time.time() - start) * 1000
            return {"success": True, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}

class MilvusRestfulClient(MilvusClientBase):
    """Milvus RESTful Client Wrapper"""
    def __init__(self, host, collection_name, **kwargs):
        self.host = host.rstrip('/')  # Remove trailing slash
        self.collection_name = collection_name
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
        # Setup authentication
        self.token = kwargs.get("token", "root:Milvus")
        self.db_name = kwargs.get("db_name", "default")
        self.timeout = kwargs.get("timeout", 30)
        
        # Setup session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            self.logger.warning(f"Failed to test connection: {e}")
    
    def _test_connection(self):
        """Test REST API connectivity"""
        url = f"{self.host}/v2/vectordb/collections/list"
        data = {"dbName": self.db_name}
        response = self.session.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
    
    def close(self):
        if hasattr(self, 'session'):
            self.session.close()
    
    def insert(self, data):
        start = time.time()
        try:
            url = f"{self.host}/v2/vectordb/entities/insert"
            payload = {
                "collectionName": self.collection_name,
                "data": data
            }
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            total_time = (time.time() - start) * 1000
            
            # Check if insertion was successful
            success = result.get("code") == 0 or response.status_code == 200
            return {"success": success, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def upsert(self, data):
        start = time.time()
        try:
            url = f"{self.host}/v2/vectordb/entities/upsert"
            payload = {
                "collectionName": self.collection_name,
                "data": data
            }
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            total_time = (time.time() - start) * 1000
            
            success = result.get("code") == 0 or response.status_code == 200
            return {"success": success, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def search(self, data, anns_field, top_k, param=None, output_fields=None, search_type=None):
        if search_type is None:
            search_type = "dense-search"
        if param is None:
            param = {"metric_type": "COSINE", "params": {"ef": 64}}
        if output_fields is None:
            output_fields = ["id"]
        
        start = time.time()
        try:
            url = f"{self.host}/v2/vectordb/entities/search"
            
            # Handle single vector or multiple vectors
            search_vector = data[0] if isinstance(data, list) and len(data) > 0 else data
            
            payload = {
                "collectionName": self.collection_name,
                "vector": search_vector,
                "limit": top_k,
                "outputFields": output_fields,
                "searchParams": param
            }
            
            # Add filter if provided
            if "filter" in self.kwargs and self.kwargs["filter"]:
                payload["filter"] = self.kwargs["filter"]
            
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            total_time = (time.time() - start) * 1000
            
            # Check results
            success = result.get("code") == 0 or response.status_code == 200
            search_results = result.get("data", [])
            empty = len(search_results) == 0
            
            return {"success": success and not empty, "response_time": total_time, "empty": empty, "result": search_results}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def query(self, expr, output_fields=None, expr_type=None):
        if expr_type is None:
            expr_type = "text_match"
        if output_fields is None:
            output_fields = ["id"]
        
        start = time.time()
        try:
            url = f"{self.host}/v2/vectordb/entities/query"
            payload = {
                "collectionName": self.collection_name,
                "filter": expr,
                "outputFields": output_fields
            }
            
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            total_time = (time.time() - start) * 1000
            
            success = result.get("code") == 0 or response.status_code == 200
            query_results = result.get("data", [])
            empty = len(query_results) == 0
            
            return {"success": success and not empty, "response_time": total_time, "empty": empty, "result": query_results}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}
    
    def delete(self, expr):
        start = time.time()
        try:
            url = f"{self.host}/v2/vectordb/entities/delete"
            payload = {
                "collectionName": self.collection_name,
                "filter": expr
            }
            
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            total_time = (time.time() - start) * 1000
            
            success = result.get("code") == 0 or response.status_code == 200
            return {"success": success, "response_time": total_time, "result": result}
        except Exception as e:
            return {"success": False, "response_time": (time.time() - start) * 1000, "exception": e}


# ----------------------------------
# Locust User wrapper
# ----------------------------------
from locust import User, events


class MilvusUser(User):
    """Locust User implementation for Milvus operations.

    This class wraps a selected `MilvusClientBase` implementation and translates
    client method results into Locust request events so that performance
    statistics are collected properly.

    Parameters
    ----------
    client_type : str, optional
        Which underlying client to use. One of ``orm`` (default), ``v2``, or
        ``restful``.
    host : str
        Milvus server URI, e.g. ``"http://localhost:19530"``.
    collection_name : str
        The name of the collection to operate on.
    **client_kwargs
        Additional keyword arguments forwarded to the chosen client.
    """

    abstract = True

    def __init__(self, *args, client_type: str = "orm", host: Optional[str] = None, collection_name: Optional[str] = None, **client_kwargs):
        super().__init__(*args, **client_kwargs)

        if host is None:
            raise ValueError("'host' must be provided for MilvusUser")
        if collection_name is None:
            raise ValueError("'collection_name' must be provided for MilvusUser")

        client_map = {
            "orm": MilvusORMClient,
            "v2": MilvusV2Client,
            "restful": MilvusRestfulClient,
        }
        if client_type not in client_map:
            raise ValueError(f"Unsupported client_type '{client_type}'. Must be one of {list(client_map)}")

        self.client: MilvusClientBase = client_map[client_type](host=host, collection_name=collection_name, **client_kwargs)

    # Helper -------------------------------------------------------------
    @staticmethod
    def _fire_event(name: str, result: Dict[str, Any]):
        """Emit a Locust request event from a Milvus client result dict."""
        response_time = int(result.get("response_time", 0))
        events.request.fire(
            request_type="milvus",
            name=name,
            response_time=response_time,
            response_length=len(result.get("result", [])) if result.get("result") is not None else 0,
            exception=result.get("exception"),
        )

    # Public wrappers ----------------------------------------------------
    def insert(self, data):
        result = self.client.insert(data)
        self._fire_event("insert", result)
        return result

    def upsert(self, data):
        result = self.client.upsert(data)
        self._fire_event("upsert", result)
        return result

    def search(self, data, anns_field, top_k, param=None, output_fields=None, search_type="dense-search"):
        result = self.client.search(data, anns_field, top_k, param=param, output_fields=output_fields, search_type=search_type)
        self._fire_event("search", result)
        return result

    def query(self, expr, output_fields=None, expr_type=""):
        result = self.client.query(expr, output_fields=output_fields, expr_type=expr_type)
        self._fire_event("query", result)
        return result

    def delete(self, expr):
        result = self.client.delete(expr)
        self._fire_event("delete", result)
        return result

    # We keep on_stop for symmetry, but current clients require no cleanup.
    def on_stop(self):
        self.client.close()
