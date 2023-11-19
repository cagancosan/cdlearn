// Pipeline for building, testing and deploying cdlean package.
pipeline {
    agent any

    stages {
        
        stage("Build Docker Image") {
            
        // Using latest tag.    
        steps {
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