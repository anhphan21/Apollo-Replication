# How to build the dockerfile
```bash
docker build . --file Dockerfile --tag anhph/apollo-replication:cuda
```

# How to run the dockerfile
```bash
docker run -dit -v $(pwd):/Apollo-Replication -w /Apollo-Replication anhph/apollo-replication:cuda bash
```

# How to build the code
```bash
```