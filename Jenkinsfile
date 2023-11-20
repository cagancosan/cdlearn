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
                sh "docker run --rm --name cdlearn-container cdlearn-user/cdlearn:latest pytest /cdlearn_app/test/test_clustering.py"
            }
        }
        stage("Deploy") {
            steps {
                echo "Deploying ..."
            }
        }
    }
}