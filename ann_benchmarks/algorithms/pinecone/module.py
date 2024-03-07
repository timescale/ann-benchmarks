from ..base.module import BaseANN
import concurrent.futures
from typing import Optional
import psutil
import pinecone
import numpy
from time import sleep


MAX_THREADS = 16


class Pinecone(BaseANN):
    def __init__(self, metric, api_key, environment, index_name, pods, pod_type, replicas):
        if metric == "angular":
            metric = "cosine"
        self._metric = metric
        self._api_key = api_key
        self._environment = environment
        self._index_name = index_name
        self._pods = pods
        self._pod_type = pod_type
        self._replicas = replicas
        self._query_search_list_size = None

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        pass

    def get_memory_usage(self) -> Optional[float]:
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X: numpy.array) -> None:
        print("initializing pinecone client...")
        pc = pinecone.Pinecone(api_key=self._api_key)
        dimension = X.shape[1]
        print(f"dimension: {dimension}")
        for idx in pc.list_indexes():
            if idx == self._index_name:
                print(f"deleting existing index {self._index_name}...")
                pc.delete_index(self._index_name)
        print(f"creating index {self._index_name}...")
        if self._replicas > 0:
            pc.create_index(
                name=self._index_name, 
                dimension=dimension,
                metric=self._metric,
                spec=pinecone.PodSpec(
                    environment=self._environment,
                    replicas=self._replicas,
                    pod_type=self._pod_type,
                    pods=self._pods,
                ))
        else:
            pc.create_index(
                name=self._index_name, 
                dimension=dimension,
                metric=self._metric,
                spec=pinecone.PodSpec(
                    environment=self._environment,
                    pod_type=self._pod_type,
                    pods=self._pods,
                ))
        print("waiting for index to be ready...")
        ready = False
        while not ready:
            sleep(5)
            index = pc.describe_index(self._index_name)
            ready = index.status['ready']
        print("upserting dataset...")
        index = pc.Index(self._index_name)
        total = len(X)
        print(f"upserting {total} vectors...")
        batch: list[pinecone.Vector] = []
        for i, v in enumerate(X):
            batch.append(pinecone.Vector(id = str(i), values=v.astype(float).tolist()))
            if len(batch) == 100 or i == total - 1:
                print(f"{i}: upserting batch of {len(batch)} vectors")
                index.upsert(vectors=batch, batch_size=len(batch), show_progress=False)
                batch: list[pinecone.Vector] = []
        print("index loaded")
        self._index = index

    def query(self, q: numpy.array, n: int) -> numpy.array:
        resp = self._index.query(vector=q.astype(float).tolist(), top_k=n, include_values=False)
        matches = []
        for match in resp['matches']:
            matches.append(int(match['id']))
        return numpy.array(matches)

    def batch_query(self, X: numpy.array, n: int) -> None:
        results = numpy.empty((X.shape[0], n), dtype=int)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(self.query, q, n): i for i, q in enumerate(X)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as x2:
                    print(f"exception getting batch results: {x2}")
        self.res = results

    def get_batch_results(self) -> numpy.array:
        return self.res

    def set_query_arguments(self, query_search_list_size):
        self._query_search_list_size = query_search_list_size

    def __str__(self) -> str:
        return f"Pinecone(index_name={self._index_name}, pods={self._pods}, pod_type={self._pod_type}, replicas={self._replicas}, query_search_list_size={self._query_search_list_size})"
