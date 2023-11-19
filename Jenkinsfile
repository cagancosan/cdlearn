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
                sh "docker exec --interactive --tty cdlearn-container pytest test"
            }
        }
        stage("Deploy") {
            steps {
                echo "Deploying ..."
            }
        }
    }
}