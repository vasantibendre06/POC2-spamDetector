pipeline:
  agent: any

  environment:
    PROJECT_DIR: "POC2-spamDetector"
    BACKEND_DIR: "backend"
    FRONTEND_DIR: "frontend"
    VENV_DIR: "venv"
    FLASK_APP: "server.py"

  stages:
    - stage: "Install System Dependencies"
      steps:
        - script: |
            sudo apt update
            sudo apt install -y python3.12-venv nodejs npm

    - stage: "Checkout Code"
      steps:
        - script: |
            rm -rf ${PROJECT_DIR}
            git clone https://github.com/vasantibendre06/POC2-spamDetector.git ${PROJECT_DIR}

    - stage: "Setup Backend Virtual Environment & Install Dependencies"
      steps:
        - script: |
            cd ${PROJECT_DIR}/${BACKEND_DIR}
            python3 -m venv ${VENV_DIR}
            bash -c "source ${VENV_DIR}/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

    - stage: "Run Backend Tests (Optional)"
      steps:
        - script: |
            cd ${PROJECT_DIR}/${BACKEND_DIR}
            bash -c "source ${VENV_DIR}/bin/activate && if [ -d 'tests' ]; then pytest tests/; else echo 'No tests found, skipping...'; fi"

    - stage: "Deploy Flask Backend"
      steps:
        - script: |
            cd ${PROJECT_DIR}/${BACKEND_DIR}
            bash -c "source ${VENV_DIR}/bin/activate && nohup python ${FLASK_APP} > server.log 2>&1 &"

    - stage: "Install Frontend Dependencies"
      steps:
        - script: |
            cd ${PROJECT_DIR}/${FRONTEND_DIR}
            npm install

    - stage: "Build Frontend"
      steps:
        - script: |
            cd ${PROJECT_DIR}/${FRONTEND_DIR}
            npm run build

    - stage: "Deploy Frontend"
      steps:
        - script: |
            sudo mkdir -p /var/www/html/spamdetector
            sudo cp -r ${PROJECT_DIR}/${FRONTEND_DIR}/build/* /var/www/html/spamdetector/

  post:
    always:
      - echo "Pipeline execution completed."
    success:
      - echo "Build and Deployment Successful!"
    failure:
      - echo "Build or Deployment Failed."

