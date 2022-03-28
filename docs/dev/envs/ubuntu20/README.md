The common way to build a docker image is:

```shell
docker build -t ubuntu20:dev .
```

Docker supports cross-platform image build. That's why there are arguments `TARGETPLATFORM` and `BUILDPLATFORM`. By
default, they are set to your current platform.

To cross-build a docker image, using [`docker buildx build`](https://docs.docker.com/engine/reference/commandline/buildx_build/):
```shell
# for Apple Silicon processors
docker buildx build --load --platform linux/arm64 -t ubuntu20:dev-arm64 .
# NB: without --load, the built image will stay in buildx cache and not accessible to docker run
```
or
```shell
# to build for x86-64 arch on Apple Silicon processors 
docker buildx build --load --platform linux/amd64 -t ubuntu20:dev-amd64 .
```
or
```shell
# multiple architecture build; up to now, you have to push the image on a repository to use them (cf --push) 
docker buildx build --platform linux/amd64,linux/arm64 -t ${DOCKERHUB_USER}/ubuntu20:dev . --push
docker buildx imagetools inspect ${DOCKERHUB_USER}/ubuntu20:dev # show that is a multi-arch images
# Then, you have to specify the expected arch on run
docker run --platform linux/amd64 ${DOCKERHUB_USER}/ubuntu20:dev
```