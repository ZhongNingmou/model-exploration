# Model evaluation utilizing multiple open-source platforms

## How to run

1. Clone(download) this repository

2. Build and run the containers with `docker-compose`

    ```bash
    docker-compose up --build
    ```

3. Set environment variables and build MLflow server

    ```bash
    export AWS_SECRET_ACCESS_KEY="minio123" 
    export AWS_ACCESS_KEY_ID="minio"
    mlflow server --backend-store-uri postgresql://user:password@127.0.0.1/mlflow --default-artifact-root http://127.0.0.1:9000/mlflow --host 0.0.0.0 --port 5000
    ```

4. Access MLflow UI with http://localhost:5000

5. Access MinIO UI with http://localhost:9000

6. Access LDAP UI with http://localhost:10004

## Set up account and run example

1. Open LDAP UI and login with 'cn=admin,dc=ningmoulocal,dc=com' and password 'admin_pass'

Create new groups and new users using the following steps:
    Under the basic root, select Create a child entry, then select Generic: Posix Group to create a new group
    Under the new group entry, press Create a child entry, then select Generic: User Account to create a new user
Each user must have a group ID, i.e, under an existing group

2. Set MinIO alias and user policy. Get into bin with

    ```bash
    docker exec -it mc_container_name /bin/sh
    ```
and set mc alias with 

    ```bash
    mc alias set miniohost http://minio:9000
    ```

Then set different policies on the users with the following command 

    ```bash
    mc admin policy set miniohost POLICY user=user_distingued_name
    ```

This policy is the default policy for the user to access all buckets and objects

3. Login to MinIO with LDAPAfter username(cn) and password. Then create a new bucket "mlflow" and modify the access policy

4. Access Grafana UI with http://localhost:3000 and login with username 'admin' and password 'admin'

5. run run_example.py to begin data transfer and check data drift dashboard on Grafana general dashboard

6. cd into inference folder and run inferece.py and check results on Mlflow UI with http://localhost:5000

## Containerization

The MLflow tracking server is composed of 4 docker containers:

* MLflow server
* MinIO object storage server
* Postgres database server

## Example

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Install MLflow with extra dependencies, inclusing scikit-learn

    ```bash
    pip install mlflow[extras]
    ```

3. Set environmental variables

    ```bash
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    ```
4. Set MinIO credentials

    ```bash
    cat <<EOF > ~/.aws/credentials
    [default]
    aws_access_key_id=minio
    aws_secret_access_key=minio123
    EOF
    ```

5. Train a sample MLflow model

    ```bash
    mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.42
    ```

    * Note: To fix ModuleNotFoundError: No module named 'boto3'

        ```bash
        #Switch to the conda env
        conda env list
        conda activate mlflow-3eee9bd7a0713cf80a17bc0a4d659bc9c549efac #replace with your own generated mlflow-environment
        pip install boto3
        ```

 6. Serve the model (replace with your model's actual path)
    ```bash
    mlflow models serve -m S3://mlflow/0/98bdf6ec158145908af39f86156c347f/artifacts/model -p 1234
    ```

 7. You can check the input with this command
    ```bash
    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations
    ```
