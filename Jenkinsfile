pipeline {
    agent any

    environment {
        MODEL_PATH = 'models/heart_pipeline.joblib'
    }

    stages {

        stage('Build') {
            steps {
                sh '''
                    docker build -t heart_dis_server:latest .
                    docker image inspect heart_dis_server:latest
                '''
            }
        }

        stage('Start Container') {
            steps {
                sh '''
                    docker rm -f heart-server 2>/dev/null || true

                    docker run -d \
                        --name heart-server \
                        -e MODEL_PATH=${MODEL_PATH} \
                        -p 5999:5999 \
                        heart_dis_server:latest
                '''
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    def status = sh(
                        script: '''
                            docker exec heart-server \
                                pytest tests/test_model.py
                        ''',
                        returnStatus: true
                    )

                    if (status != 0) {
                        echo "Model tests failed. Switching MODEL_PATH..."

                        env.MODEL_PATH = 'models/model_fold_1.joblib'
                    } else {
                        echo "Model tests passed."
                    }
                }
            }
        }

       stage('Deploy') {
    steps {
        sh '''
            set -e

            echo "=== Removing temporary Docker test container ==="
            docker rm -f heart-server 2>/dev/null || true

            echo "=== Running as user ==="
            whoami

            echo "=== Starting Minikube ==="
            minikube start

            echo "=== Loading local Docker image into Minikube ==="
            minikube image load heart_dis_server:latest

            echo "=== Checking image ==="
            minikube image ls | grep heart_dis_server

            echo "=== Applying Kubernetes configuration ==="
            kubectl apply -f deployment.yaml

            echo "=== Setting MODEL_PATH ==="
            kubectl set env \
                deployment/heart-server \
                MODEL_PATH=${MODEL_PATH}

            echo "=== Waiting for deployment ==="
            kubectl rollout status \
                deployment/heart-server \
                --timeout=120s

            echo "=== Current pods ==="
            kubectl get pods

            echo "=== Current service ==="
            kubectl get service heart-server

            echo "=== Stopping previous port-forward if present ==="
            if [ -f /tmp/heart-server-port-forward.pid ]; then
                OLD_PID=$(cat /tmp/heart-server-port-forward.pid)

                kill "$OLD_PID" 2>/dev/null || true

                rm -f /tmp/heart-server-port-forward.pid
            fi

            echo "=== Exposing heart-server on localhost:5999 ==="

            nohup kubectl port-forward \
                service/heart-server \
                5999:5999 \
                --address=127.0.0.1 \
                > /tmp/heart-server-port-forward.log 2>&1 &

            echo $! > /tmp/heart-server-port-forward.pid

            sleep 3

            echo "=== Checking port-forward ==="

            if ! kill -0 $(cat /tmp/heart-server-port-forward.pid) 2>/dev/null; then
                echo "ERROR: kubectl port-forward failed"
                cat /tmp/heart-server-port-forward.log
                exit 1
            fi

            echo "=== Deployment successful ==="
            echo "Heart server available at http://localhost:5999"
        '''
    }
}
    }
}