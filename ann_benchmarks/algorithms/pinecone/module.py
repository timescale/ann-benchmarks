from ..base.module import BaseANN
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Optional
import psutil
import pinecone
import numpy
from time import sleep

class Pinecone(BaseANN):
    def __init__(self, metric, api_key, environment, index_name, pods, pod_type):
        self._metric = metric
        self._api_key = api_key
        self._environment = environment
        self._index_name = index_name
        self._pods = pods
        self._pod_type = pod_type
        self._query_search_list_size = None

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X: numpy.array) -> None:
        print("initializing pinecone client...")
        pinecone.init(api_key=self._api_key, environment=self._environment)
        dimension = X.shape[1]
        for idx in pinecone.list_indexes():
            if idx == self._index_name:
                print(f"deleting existing index {self._index_name}...")
                pinecone.delete_index(self._index_name)
        print(f"creating index {self._index_name}...")
        pinecone.create_index(name=self._index_name, dimension=dimension, 
                              metric=self._metric, pods=self._pods, pod_type=self._pod_type)
        print("waiting for index to be ready...")
        ready = False
        while not ready:
            sleep(5)
            index = pinecone.describe_index(self._index_name)
            ready = index.status['ready']
        print("upserting dataset...")
        index = pinecone.Index(self._index_name)
        total = len(X)
        batch = []
        for i, v in enumerate(X):
            batch.append({"id": str(i), "values": v.astype(float).tolist()})
            if len(batch) == 100 or i == total - 1:
                print(f"{i}: upserting batch of {len(batch)} vectors")
                index.upsert(vectors=batch, batch_size=len(batch), show_progress=True)
                batch = []
        print("index loaded")

    def query(self, q: numpy.array, n: int) -> numpy.array:
        index = pinecone.Index(self._index_name)
        resp = index.query(vector=q.astype(float).tolist(), top_k=n, include_values=False)
        matches = []
        for match in resp['matches']:
            matches.append(int(match['id']))
        return numpy.array(matches)

    def batch_query(self, X: numpy.array, n: int) -> None:
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self) -> numpy.array:
        return self.res

    def set_query_arguments(self, query_search_list_size):
        self._query_search_list_size = query_search_list_size

    def __str__(self) -> str:
        return f"Pinecone(index_name={self._index_name}, pods={self._pods}, pod_type={self._pod_type}, query_search_list_size={self._query_search_list_size})"
