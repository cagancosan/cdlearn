// Pipeline for building, testing and deploying cdlean package.
pipeline {
    agent any
    stages {
        stage("Build Docker Image") {
            steps { 

                // Dockerfile linter.
                sh """
                docker run \
                --rm \
                --interactive \
                hadolint/hadolint < Dockerfile
                """
                
                // Using latest tag.  
      	        sh """
                docker build \
                --tag cdlearn-user/cdlearn:latest \
                --file ./Dockerfile .
                """
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