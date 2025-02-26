from time import sleep, time
from typing import Iterable, List, Any
import asyncio
import numpy as np
import concurrent.futures
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import grpc
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    BinaryQuantization,
    BinaryQuantizationConfig,
    ScalarType,
    HnswConfigDiff,
)
import threading
from pprint import pprint

from ..base.module import BaseANN

TIMEOUT = 30
QDRANT_BATCH_SIZE = 128  # Size of batches for Qdrant requests

class Qdrant(BaseANN):
    _distances_mapping = {"dot": Distance.DOT, "angular": Distance.COSINE, "euclidean": Distance.EUCLID}

    def __init__(self, metric, quantization, m, ef_construct):
        self._ef_construct = ef_construct
        self._m = m
        self._metric = metric
        self._collection_name = "ann_benchmarks_matrix"
        self._quantization_mode = quantization
        self._grpc = True
        self._search_params = {"hnsw_ef": None, "rescore": True}
        self.batch_results = []
        self.batch_latencies = []

        # Client configuration
        self._client_config = {
            "host": "172.31.23.61",
            "port": 6333,
            "grpc_port": 6334,
            "prefer_grpc": self._grpc,
        }
        
        # Initialize synchronous client for management operations
        self._client = QdrantClient(**self._client_config)
        
        # Initialize single global async client for search operations
        self._async_client = AsyncQdrantClient(**self._client_config)
        
        # Initialize event loop for async operations
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def __del__(self):
        if hasattr(self, '_loop') and self._loop is not None:
            self._loop.close()

    def fit(self, X):
        return
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        quantization_config = None
        if self._quantization_mode == "scalar":
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    always_ram=True,
                    quantile=0.99,
                    type=ScalarType.INT8,
                )
            )
        elif self._quantization_mode == "binary":
            quantization_config = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )

        print("recreating collection...")
        # Disabling indexing during bulk upload
        # https://qdrant.tech/documentation/tutorials/bulk-upload/#disable-indexing-during-upload
        # Uploading to multiple shards
        # https://qdrant.tech/documentation/tutorials/bulk-upload/#parallel-upload-into-multiple-shards
        self._client.recreate_collection(
            collection_name=self._collection_name,
            shard_number=2,
            vectors_config=VectorParams(size=X.shape[1], distance=self._distances_mapping[self._metric], on_disk=True),
            # optimizers_config=OptimizersConfigDiff(
            #     default_segment_number=2,
            #     memmap_threshold=20000,
            #     indexing_threshold=0,
            # ),
            quantization_config=quantization_config,
            # TODO: benchmark this as well
            # hnsw_config=HnswConfigDiff(
            #     ef_construct=self._ef_construct,
            #     m=self._m,
            # ),
            timeout=TIMEOUT,
        )
        print("collection recreated")

        print("uploading vectors...")
        def upload_with_retry(ids: list[int], vectors: list[list[float]]) -> bool:
            retry_count = 0
            backoff_time = 1  # Initial backoff time in seconds
            while retry_count < 10:
                try:
                    # Attempt to upload the collection
                    self._client.upload_collection(
                        collection_name=self._collection_name,
                        vectors=vectors,
                        ids=ids,
                        batch_size=QDRANT_BATCH_SIZE,
                        parallel=1,
                    )
                    return True
                except grpc._channel._InactiveRpcError:
                    print(f"Upload failed, retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                    retry_count += 1
            print("Maximum retries reached. Upload failed.")
            return False
        
        ids = []
        vectors = []
        for i, x in enumerate(X):
            ids.append(i)
            vectors.append([float(f) for f in x])
            if i > 0 and i % QDRANT_BATCH_SIZE == 0:
                print(f"{i} uploading collection of {len(vectors)} vectors")
                upload_with_retry(ids=ids, vectors=vectors)
                ids = []
                vectors = []
        print("done uploading vectors")

        #print("uploading collection...")
        #self._client.upload_collection(
        #    collection_name=self._collection_name,
        #    vectors=X,
        #    ids=list(range(X.shape[0])),
        #    batch_size=BATCH_SIZE,
        #    parallel=1,
        #)

        print("re-enabling indexing...")
        # Re-enabling indexing
        self._client.update_collection(
            collection_name=self._collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
            timeout=TIMEOUT,
        )

        # wait for vectors to be fully indexed
        SECONDS_WAITING_FOR_INDEXING_API_CALL = 5

        while True:
            print("waiting for indexing to complete...")
            sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)
            collection_info = self._client.get_collection(self._collection_name)
            if collection_info.status != CollectionStatus.GREEN:
                continue
            sleep(SECONDS_WAITING_FOR_INDEXING_API_CALL)  # the flag is sometimes flacky, better double check
            collection_info = self._client.get_collection(self._collection_name)
            if collection_info.status == CollectionStatus.GREEN:
                print(f"Stored vectors: {collection_info.vectors_count}")
                print(f"Indexed vectors: {collection_info.indexed_vectors_count}")
                print(f"Collection status: {collection_info.indexed_vectors_count}")
                print("indexing complete.")
                break

    async def _process_qdrant_batch(self, vectors: List[np.ndarray], n: int) -> tuple[List[List[int]], List[float]]:
        """Process a batch of vectors using Qdrant's batch search"""
        search_points = [
            grpc.SearchPoints(
                collection_name=self._collection_name,
                vector=vector.tolist(),
                limit=n,
                with_payload=grpc.WithPayloadSelector(enable=False),
                with_vectors=grpc.WithVectorsSelector(enable=False),
                params=grpc.SearchParams(
                    quantization=grpc.QuantizationSearchParams(ignore=False),
                ),
            )
            for vector in vectors
        ]

        try:
            start_time = time()
            batch_request = grpc.SearchBatchPoints(
                collection_name=self._collection_name,
                search_points=search_points
            )
            response = await self._async_client.grpc_points.SearchBatch(batch_request, timeout=TIMEOUT)
            query_time = time() - start_time

            # Extract results
            batch_results = []
            for search_response in response.result:
                batch_results.append([hit.id.num for hit in search_response.result])

            return batch_results, [query_time] * len(vectors)
        except Exception as e:
            print(f"Error in batch processing: {e}")
            raise

    async def _batch_query_async(self, X: np.ndarray, n: int):
        """Process all queries in batches using async/await"""
        n_total = len(X)
        results = np.empty((n_total, n), dtype=int)
        latencies = np.empty(n_total, dtype=float)
        
        # Process batches sequentially but use async/await for each batch
        for i in range(0, n_total, QDRANT_BATCH_SIZE):
            batch = X[i:i + QDRANT_BATCH_SIZE]
            batch_results, batch_latencies = await self._process_qdrant_batch(batch, n)
            
            for j, (result, latency) in enumerate(zip(batch_results, batch_latencies)):
                results[i + j] = result
                latencies[i + j] = latency

        return results, latencies

    def set_query_arguments(self, hnsw_ef, rescore):
        self._search_params["hnsw_ef"] = hnsw_ef
        self._search_params["rescore"] = rescore

    def batch_query(self, X: np.ndarray, n: int):
        """Entry point for batch querying"""
        results, latencies = self._loop.run_until_complete(self._batch_query_async(X, n))
        self.batch_results = results
        self.batch_latencies = latencies

    def get_batch_results(self):
        return self.batch_results

    def get_batch_latencies(self):
        return self.batch_latencies

    def __str__(self):
        ef_construct = self._ef_construct
        m = self._m
        hnsw_ef = self._search_params["hnsw_ef"]
        rescore = self._search_params["rescore"]
        return f"Qdrant(quantization={self._quantization_mode}, m={m}, ef_construct={ef_construct}, hnsw_ef={hnsw_ef}, rescore={rescore})"
