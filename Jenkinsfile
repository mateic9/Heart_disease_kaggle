pipeline {
  agent any
  stages {
    stage('hello') {
      steps {
       sh '''#!/bin/bash
            echo $(uname -a)
         '''
        
      }
    }
  }
}
