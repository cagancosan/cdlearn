// Pipeline for building, testing and deploying cdlean package.
pipeline {
    agent any
    stages {
        stage("Build Docker Image") {
            steps { 
                
                // Using latest tag.  
      	        sh "docker build --tag cdlearn-user/cdlearn:latest --file ./Dockerfile ."
            }
        }
        stage("Test") {
            steps {
                
                // Run all unit tests.
                sh "docker run --rm --entrypoint /bin/bash --name cdlearn-container cdlearn-user/cdlearn:latest"
                sh "docker exec cdlearn-user/cdlearn:latest pytest test"
                sh "docker stop cdlearn-user/cdlearn:latest"
            }
        }
        stage("Deploy") {
            steps {
                echo "Deploying ..."
            }
        }
    }
}