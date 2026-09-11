pipeline {
  agent any
  stages {
    stage('hello') {
      steps {
       sh '''#!/bin/bash
            whoami
            docker images
         '''
        
      }
    }
  }
}
