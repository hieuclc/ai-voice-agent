model=Alibaba-NLP/gte-multilingual-reranker-base
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
docker run -d --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model

docker pull unclecode/crawl4ai:latest
docker run -d -p 11235:11235 --name crawl4ai --shm-size=1g unclecode/crawl4ai:latest