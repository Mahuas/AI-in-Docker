# 15 AI applications packaged by docker



I integrated 15 AI applications to Flask framework, which could be packaged into 15 separate docker images.

Please go to `./apps/`, you will see 15 applications, and each application has a `Dockerfile`. Just run the following commands to build a docker image and run the image:

```shell
docker build -t verify-code:latest .
docker run --name image-verify -d -p 3500:3500 verify-code:latest
```



Then, you can test it with this:

```shell
curl -X POST -F image=@test1.jpg 'http://localhost:3500/predict'
```



More details will be updated if I'm free. Thanks.