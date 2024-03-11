begin;

set local max_parallel_workers_per_gather = 0;
set local max_parallel_workers = 8;
set local max_parallel_maintenance_workers = 7;
set local work_mem = '8GB';
set local maintenance_work_mem = '8GB';

with i as
(
insert into log (name, start) 
values ('indexing', clock_timestamp())
returning id
)
select id from i
\gset

create index on only public.items using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);

update log set stop=clock_timestamp()
where id = :id
;

commit;

