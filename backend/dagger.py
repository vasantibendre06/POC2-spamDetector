import dagger
import os

async def main():
    async with dagger.Connection() as client:
        
        print("Setting up Backend Container...")
        backend_container = (
            client.container()
            .from_("python:3.12-slim")
            .with_workdir("/app")
            .with_mounted_directory("/app", client.host().directory("."))
            .with_exec(["pip", "install", "-r", "backend/requirements.txt"])
        )

        print("Running Backend Tests...")
        test_result = await backend_container.with_exec(["pytest", "backend/tests"]).stdout()
        print(test_result)

        print("Building and Publishing Backend Image...")
        backend_image = (
            client.container()
            .build(".", dockerfile="Dockerfile.backend")
            .publish("your-dockerhub/backend-app")
        )

        print("Setting up Frontend Container...")
        frontend_container = (
            client.container()
            .from_("node:18")
            .with_workdir("/app")
            .with_mounted_directory("/app", client.host().directory("."))
            .with_exec(["yarn", "install"])
            .with_exec(["yarn", "build"])
        )

        print("Building and Publishing Frontend Image...")
        frontend_image = (
            client.container()
            .build(".", dockerfile="Dockerfile.frontend")
            .publish("your-dockerhub/frontend-app")
        )

        print("Deploying to AWS ECS...")
        deploy_command = [
            "aws", "ecs", "update-service",
            "--cluster", "my-cluster",
            "--service", "my-service",
            "--force-new-deployment"
        ]
        deploy_result = await client.container().with_exec(deploy_command).stdout()
        print("Deployment Output: ", deploy_result)

dagger.run(main)
