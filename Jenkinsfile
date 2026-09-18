pipeline {
  agent any
  stages {
    stage('Buid') {
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
                        --env-file .env \
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

                sh '''
                    docker rm -f heart-server

                    docker run -d \
                        --name heart-server \
                        --env-file .env \
                        -e MODEL_PATH= models/model_fold_1.joblib \
                        -p 5999:5999 \
                        heart_dis_server:latest
                '''
          }
        }
    }

  }
}
