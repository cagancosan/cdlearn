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
        stage("Test Submodules") {
            steps {
                sh """
                docker run --rm --name cdlearn-container cdlearn-user/cdlearn:latest \
                pytest /cdlearn_app/test/test_clustering.py
                """
            }
        }
        stage("Code Quality Evaluation") {
            steps {
                sh """
                docker run --rm --name cdlearn-container cdlearn-user/cdlearn:latest \
                pylint --exit-zero --msg-template='{path}:{line}: [{msg_id}, {obj}] {msg} ({symbol})' /cdlearn_app/cdlearn > pylint.log
                """
            }    
        }
        stage("Deploy Module") {
            steps {
                echo "Deploying ..."
            }
        }
    }
}