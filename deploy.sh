# assuming environment variables are set
# MODEL_NAME
# SERVE_PATH

# this will pull tensorflow/serving if not exists
# then will run the server
docker run --rm -it -d \
-v $SERVE_PATH/models_for_serving:/models \
-e MODEL_NAME=$MODEL_NAME -e MODEL_PATH=/models \
-p 8500:8500 -p 8501:8501 --name tf_server tensorflow/serving:latest