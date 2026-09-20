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
                    # Stop/remove temporary test container
                    docker rm -f heart-server || true
                    echo ("Running as user: $(whoami)")
                    minikube start
                    # Create/update Kubernetes resources
                    kubectl apply -f deployment.yaml

                    # Override MODEL_PATH with the model selected by Jenkins
                    kubectl set env \
                        deployment/heart-server \
                        MODEL_PATH=${MODEL_PATH}

                    # Wait for deployment
                    kubectl rollout status deployment/heart-server
                '''
            }
        }
    }
}