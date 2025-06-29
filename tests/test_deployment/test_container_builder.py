"""
Tests for container deployment functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# from simpulse.deployment.container_builder import ContainerBuilder


class TestContainerBuilder:
    """Test suite for ContainerBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a ContainerBuilder instance for testing."""
        return ContainerBuilder(
            base_image="leanprover/lean4:latest", push_registry=None
        )

    def test_generate_dockerfile(self, builder, temp_dir):
        """Test Dockerfile generation."""
        project_path = temp_dir
        output_path = temp_dir / "Dockerfile"

        # Create fake project structure
        (project_path / "lakefile.lean").touch()
        (project_path / "src").mkdir()
        (project_path / "src" / "Main.lean").touch()

        # Generate Dockerfile
        result = builder.generate_dockerfile(project_path, output_path)

        # Verify Dockerfile was created
        assert result
        assert output_path.exists()

        # Check Dockerfile content
        content = output_path.read_text()
        assert "FROM leanprover/lean4:latest" in content
        assert "WORKDIR /app" in content
        assert "COPY lakefile.lean" in content
        assert "RUN lake build" in content

    def test_generate_docker_compose(self, builder, temp_dir):
        """Test docker-compose.yml generation."""
        output_path = temp_dir / "docker-compose.yml"

        services = {
            "simpulse": {
                "build": ".",
                "volumes": ["./src:/app/src"],
                "command": "lake build",
            },
            "web": {
                "image": "nginx:alpine",
                "ports": ["8080:80"],
                "depends_on": ["simpulse"],
            },
        }

        # Generate docker-compose.yml
        result = builder.generate_docker_compose(services, output_path)

        # Verify file was created
        assert result
        assert output_path.exists()

        # Check content
        content = output_path.read_text()
        assert "version: '3.8'" in content
        assert "simpulse:" in content
        assert "web:" in content
        assert "ports:" in content

    def test_build_image(self, builder, temp_dir):
        """Test Docker image building."""
        # Create minimal Dockerfile
        dockerfile = temp_dir / "Dockerfile"
        dockerfile.write_text(
            """
FROM alpine:latest
RUN echo "Test build"
CMD ["echo", "Hello Simpulse"]
"""
        )

        # Mock subprocess for docker build
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="Build complete")

            # Build image
            result = builder.build_image(
                context_path=temp_dir, tag="simpulse-test:latest"
            )

            # Verify build command was called
            assert result
            mock_run.assert_called_once()

            # Check command arguments
            args = mock_run.call_args[0][0]
            assert args[0] == "docker"
            assert args[1] == "build"
            assert "-t" in args
            assert "simpulse-test:latest" in args

    def test_push_image(self, builder):
        """Test Docker image pushing."""
        builder.push_registry = "docker.io/myuser"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            # Push image
            result = builder.push_image("simpulse:latest")

            # Verify push command
            assert result
            mock_run.assert_called()

            # Should tag and push
            calls = mock_run.call_args_list
            assert len(calls) >= 1  # At least push command

    def test_generate_github_action(self, builder, temp_dir):
        """Test GitHub Action workflow generation."""
        output_path = temp_dir / ".github" / "workflows" / "simpulse.yml"

        # Generate workflow
        result = builder.generate_github_action(output_path)

        # Verify file was created
        assert result
        assert output_path.exists()

        # Check workflow content
        content = output_path.read_text()
        assert "name: Simpulse Optimization" in content
        assert "on:" in content
        assert "jobs:" in content
        assert "steps:" in content
        assert "uses: actions/checkout" in content

    def test_create_deployment_package(self, builder, temp_dir):
        """Test complete deployment package creation."""
        project_path = temp_dir / "project"
        project_path.mkdir()

        # Create minimal project
        (project_path / "lakefile.lean").write_text("package simpulse")
        (project_path / "src").mkdir()
        (project_path / "src" / "Main.lean").write_text(
            'def main : IO Unit := IO.println "Hello"'
        )

        # Create deployment package
        package_path = builder.create_deployment_package(
            project_path=project_path, include_compose=True, include_github_action=True
        )

        # Verify package structure
        assert package_path.exists()
        assert (package_path / "Dockerfile").exists()
        assert (package_path / "docker-compose.yml").exists()
        assert (package_path / ".github" / "workflows" / "simpulse.yml").exists()

    def test_dockerfile_with_dependencies(self, builder, temp_dir):
        """Test Dockerfile generation with dependencies."""
        project_path = temp_dir

        # Create lakefile with dependencies
        lakefile_content = """
package simpulse

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"
"""
        (project_path / "lakefile.lean").write_text(lakefile_content)

        # Generate Dockerfile
        output_path = temp_dir / "Dockerfile"
        builder.generate_dockerfile(project_path, output_path)

        # Check dependency handling
        content = output_path.read_text()
        assert "lake update" in content or "lake build" in content

    def test_multi_stage_dockerfile(self, builder, temp_dir):
        """Test multi-stage Dockerfile generation."""
        builder.use_multi_stage = True

        project_path = temp_dir
        (project_path / "lakefile.lean").touch()

        output_path = temp_dir / "Dockerfile"
        builder.generate_dockerfile(project_path, output_path)

        # Check multi-stage build
        content = output_path.read_text()
        assert content.count("FROM") >= 2  # At least builder and runtime stages
        assert "AS builder" in content
        assert "COPY --from=builder" in content

    def test_error_handling(self, builder, temp_dir):
        """Test error handling in container builder."""
        # Test with non-existent project
        result = builder.generate_dockerfile(
            Path("/non/existent/path"), temp_dir / "Dockerfile"
        )
        assert not result

        # Test build failure
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="Build failed")

            result = builder.build_image(temp_dir, "test:latest")
            assert not result

    def test_validate_docker_available(self, builder):
        """Test Docker availability check."""
        with patch("subprocess.run") as mock_run:
            # Docker available
            mock_run.return_value = Mock(returncode=0, stdout="Docker version 20.10.0")
            assert builder._validate_docker_available()

            # Docker not available
            mock_run.side_effect = FileNotFoundError()
            assert not builder._validate_docker_available()


class TestDockerfileTemplates:
    """Test suite for Dockerfile templates."""

    def test_lean_base_template(self):
        """Test basic Lean Dockerfile template."""
        builder = ContainerBuilder()

        template = builder._get_dockerfile_template("lean-basic")

        # Verify template structure
        assert "FROM leanprover/lean4" in template
        assert "WORKDIR" in template
        assert "COPY" in template
        assert "RUN lake" in template

    def test_optimized_template(self):
        """Test optimized Dockerfile template."""
        builder = ContainerBuilder()

        template = builder._get_dockerfile_template("lean-optimized")

        # Should include optimization steps
        assert "lake build" in template
        assert "--release" in template or "release" in template.lower()

    def test_development_template(self):
        """Test development Dockerfile template."""
        builder = ContainerBuilder()

        template = builder._get_dockerfile_template("lean-dev")

        # Should include development tools
        assert "git" in template.lower() or "dev" in template.lower()


@pytest.mark.integration
class TestContainerBuilderIntegration:
    """Integration tests for container builder."""

    @pytest.mark.requires_docker
    def test_real_docker_build(self, temp_dir):
        """Test with real Docker daemon."""
        builder = ContainerBuilder()

        # Create minimal Dockerfile
        dockerfile = temp_dir / "Dockerfile"
        dockerfile.write_text(
            """
FROM alpine:latest
RUN echo "Integration test"
"""
        )

        # Only run if Docker is available
        if builder._validate_docker_available():
            result = builder.build_image(
                context_path=temp_dir, tag="simpulse-integration-test:latest"
            )

            # Cleanup
            import subprocess

            subprocess.run(
                ["docker", "rmi", "simpulse-integration-test:latest"],
                capture_output=True,
            )

            assert result
