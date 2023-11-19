// Pipeline for building, testing and deploying cdlean package.
pipeline {
    agent any

    stages {
        
        stage("Build Docker Image") {
            
        steps {
      	    sh "docker build -t cdlearn-user/cdlearn:latest ."
        }

        }
        
        stage("Test") {
            steps {
                echo "Testing ..."
            }
        }
        
        stage("Deploy") {
            steps {
                echo "Deploying ..."
            }
        }
    }
}