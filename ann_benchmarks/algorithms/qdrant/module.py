from time import sleep, time
from typing import Iterable, List, Any

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
    VectorStruct,
)

from ..base.module import BaseANN

TIMEOUT = 30
BATCH_SIZE = 128
MAX_BATCH_QUERY_THREADS = 16


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

        qdrant_client_params = {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "prefer_grpc": self._grpc,
        }
        self._client = QdrantClient(**qdrant_client_params)
        self._async_clients = [AsyncQdrantClient(**qdrant_client_params) for _ in range(0, MAX_BATCH_QUERY_THREADS)]
        self._cur_async_client = 0

    def fit(self, X):
        return # don't rebuild index now
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
                        batch_size=BATCH_SIZE,
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
            if i > 0 and i % BATCH_SIZE == 0:
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

    def set_query_arguments(self, hnsw_ef, rescore):
        self._search_params["hnsw_ef"] = hnsw_ef
        self._search_params["rescore"] = rescore

    def query(self, q, n):
        raise NotImplementedError
        # search_request = grpc.SearchPoints(
        #     collection_name=self._collection_name,
        #     vector=q.tolist(),
        #     limit=n,
        #     with_payload=grpc.WithPayloadSelector(enable=False),
        #     with_vectors=grpc.WithVectorsSelector(enable=False),
        #     # params=grpc.SearchParams(
        #     #     hnsw_ef=self._search_params["hnsw_ef"],
        #     #     quantization=grpc.QuantizationSearchParams(
        #     #         ignore=False,
        #     #         rescore=self._search_params["rescore"],
        #     #         oversampling=3.0,
        #     #     ),
        #     # ),
        # )

        # search_result = self._client.grpc_points.Search(search_request, timeout=TIMEOUT)
        # result_ids = [point.id.num for point in search_result.result]
        # return result_ids

    def batch_query(self, X, n):
        threads = min(MAX_BATCH_QUERY_THREADS, X.size)
        quantization_search_params = grpc.QuantizationSearchParams(
            ignore=False,
        )
        
        # Track starting index for each batch
        batch_start_indices = []
        current_idx = 0
        
        def iter_queries() -> Iterable:
            for q in X:
                yield grpc.SearchPoints(
                    collection_name=self._collection_name,
                    vector=q.tolist(),
                    limit=n,
                    with_payload=grpc.WithPayloadSelector(enable=False),
                    with_vectors=grpc.WithVectorsSelector(enable=False),
                    params=grpc.SearchParams(
                        quantization=quantization_search_params,
                    ),
                )

        def iter_batches(iterable, batch_size) -> Iterable[tuple[List[Any], int]]:
            nonlocal current_idx
            batch = []
            for item in iterable:
                batch.append(item)
                if len(batch) >= batch_size:
                    start_idx = current_idx
                    current_idx += len(batch)
                    batch_start_indices.append(start_idx)
                    yield batch, start_idx
                    batch = []
            if batch:
                start_idx = current_idx
                current_idx += len(batch)
                batch_start_indices.append(start_idx)
                yield batch, start_idx

        def query(request_batch, start_idx, thread_idx):
            start_time = time()
            try:
                grpc_res: grpc.SearchBatchResponse = self._async_clients[thread_idx].grpc_points.SearchBatch(
                    grpc.SearchBatchPoints(
                        collection_name=self._collection_name,
                        search_points=request_batch,
                        read_consistency=None,
                    ),
                    timeout=TIMEOUT,
                )
                
                # Extract results from the response
                batch_results = []
                for response in grpc_res.responses:
                    batch_results.append([hit.id for hit in response.result])
                
                end_time = time() - start_time
                return batch_results, [end_time] * len(request_batch), start_idx
            except Exception as e:
                print(f"Exception in batch query: {e}")
                raise

        results = np.empty((X.shape[0], n), dtype=int)
        latencies = np.empty(X.shape[0], dtype=float)
        
        # Distribute batches across fixed threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for batch_idx, (batch, start_idx) in enumerate(iter_batches(iter_queries(), BATCH_SIZE)):
                # Assign each batch to a fixed thread using modulo
                thread_idx = batch_idx % threads
                futures.append(
                    executor.submit(query, batch, start_idx, thread_idx)
                )
                
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results, batch_latencies, start_idx = future.result()
                    for i, (result, latency) in enumerate(zip(batch_results, batch_latencies)):
                        results[start_idx + i] = result
                        latencies[start_idx + i] = latency
                except Exception as e:
                    print(f"Exception processing batch results: {e}")
                    raise

        self.batch_results = results
        self.batch_latencies = latencies
        return results, latencies

#    def batch_query(self, X, n):
#        def iter_batches(iterable, batch_size) -> Iterable[List[Any]]:
#            """Iterate over `iterable` in batches of size `batch_size`."""
#            batch = []
#            for item in iterable:
#                batch.append(item)
#                if len(batch) >= batch_size:
#                    yield batch
#                    batch = []
#            if batch:
#                yield batch
#
#        quantization_search_params = grpc.QuantizationSearchParams(
#            ignore=False,
#            rescore=self._search_params["rescore"],
#        )
#
#        search_queries = [
#            grpc.SearchPoints(
#                collection_name=self._collection_name,
#                vector=q.tolist(),
#                limit=n,
#                with_payload=grpc.WithPayloadSelector(enable=False),
#                with_vectors=grpc.WithVectorsSelector(enable=False),
#                params=grpc.SearchParams(
#                    hnsw_ef=self._search_params["hnsw_ef"],
#                    quantization=quantization_search_params,
#                ),
#            )
#            for q in X
#        ]
#
#        self.batch_results = []
#
#        for request_batch in iter_batches(search_queries, BATCH_SIZE):
#            start = time()
#            grpc_res: grpc.SearchBatchResponse = self._client.grpc_points.SearchBatch(
#                grpc.SearchBatchPoints(
#                    collection_name=self._collection_name,
#                    search_points=request_batch,
#                    read_consistency=None,
#                ),
#                timeout=TIMEOUT,
#            )
#            self.batch_latencies.extend([time() - start] * len(request_batch))
#
#            for r in grpc_res.result:
#                self.batch_results.append([hit.id.num for hit in r.result])

    def get_batch_results(self):
        return self.batch_results

#    def get_batch_latencies(self):
#        return self.batch_latencies

    def __str__(self):
        ef_construct = self._ef_construct
        m = self._m
        hnsw_ef = self._search_params["hnsw_ef"]
        rescore = self._search_params["rescore"]
        return f"Qdrant(quantization={self._quantization_mode}, m={m}, ef_construct={ef_construct}, hnsw_ef={hnsw_ef}, rescore={rescore})"
