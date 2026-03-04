from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import paramiko

from .platformspec import PlatformSpec
from .utils import _build_module_setup

logger = logging.getLogger(__name__)


class CommandExecutor:
    """Execute commands locally or on remote systems via SSH."""

    def __init__(self, platform: PlatformSpec):
        self.platform = platform
        self._ssh_client = None
        self._tunnel_client = None
        self._local_forward = None  # (local_addr, local_port, channel) for tunnel
        self._closed = False

    def _get_ssh_client(self, verbose: bool = False) -> paramiko.SSHClient:
        """Get or create SSH client for remote execution, with tunneling if needed."""
        if self._ssh_client is None:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            ssh_key_path = None
            if self.platform.ssh_identity:
                ssh_key_path = os.path.expanduser(os.path.expandvars(self.platform.ssh_identity))
                if not os.path.exists(ssh_key_path):
                    logger.warning("SSH identity file not found: %s", ssh_key_path)
                    if verbose: print(f"WARN: SSH key not found: {ssh_key_path}")
                    ssh_key_path = None
                else:
                    if verbose: print(f"OK: SSH key found: {ssh_key_path}")
            else:
                if verbose: print("WARN: No SSH identity specified in platform config, will try SSH agent")

            try:
                if self.platform.ssh_tunnel:
                    if verbose:
                        print("\n[SSH Tunnel Setup]")
                        print(f"  Jump host: {self.platform.ssh_tunnel}:{self.platform.ssh_tunnel_port}")
                        print(f"  Target: {self.platform.machine}:{self.platform.ssh_port}")
                        print(f"  Username: {self.platform.username}")

                    self._tunnel_client = paramiko.SSHClient()
                    self._tunnel_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    try:
                        self._tunnel_client.connect(
                            self.platform.ssh_tunnel,
                            username=self.platform.username,
                            key_filename=ssh_key_path,
                            port=self.platform.ssh_tunnel_port,
                            timeout=10,
                        )
                        tunnel_transport = self._tunnel_client.get_transport()
                        if tunnel_transport is not None:
                            tunnel_transport.set_keepalive(30)
                        if verbose: print(f"OK: Connected to jump host: {self.platform.ssh_tunnel}")
                    except (paramiko.AuthenticationException, paramiko.SSHException) as e:
                        raise RuntimeError(
                            f"Failed to authenticate to tunnel host {self.platform.ssh_tunnel} "
                            f"as {self.platform.username} (port {self.platform.ssh_tunnel_port}): {e}. "
                            "Check username, ssh_identity, and ssh_tunnel_port in platform config."
                        )
                    logger.info(
                        "Connected to tunnel/jump host: %s:%s as %s",
                        self.platform.ssh_tunnel,
                        self.platform.ssh_tunnel_port,
                        self.platform.username,
                    )

                    tunnel_transport = self._tunnel_client.get_transport()
                    local_addr = "127.0.0.1"
                    local_port = self._find_free_port()

                    channel = tunnel_transport.open_channel(
                        "direct-tcpip",
                        (self.platform.machine, self.platform.ssh_port),
                        (local_addr, local_port),
                    )
                    if verbose: print(
                        f"OK: Tunnel channel opened: {local_addr}:{local_port} -> "
                        f"{self.platform.machine}:{self.platform.ssh_port}"
                    )
                    logger.info(
                        "SSH tunnel established: %s:%s -> %s:%s via %s",
                        local_addr,
                        local_port,
                        self.platform.machine,
                        self.platform.ssh_port,
                        self.platform.ssh_tunnel,
                    )

                    try:
                        self._ssh_client.connect(
                            self.platform.machine,
                            port=self.platform.ssh_port,
                            username=self.platform.username,
                            key_filename=ssh_key_path,
                            sock=channel,
                            timeout=10,
                        )
                        ssh_transport = self._ssh_client.get_transport()
                        if ssh_transport is not None:
                            ssh_transport.set_keepalive(30)
                        if verbose: print(f"OK: Connected to target via tunnel: {self.platform.machine} as {self.platform.username}")
                    except (paramiko.AuthenticationException, paramiko.SSHException) as e:
                        raise RuntimeError(
                            f"Failed to authenticate to target host {self.platform.machine} "
                            f"via tunnel {self.platform.ssh_tunnel} as {self.platform.username}: {e}. "
                            "Check username, ssh_identity, and ssh_port in platform config."
                        )
                    self._local_forward = (local_addr, local_port, channel)
                else:
                    if verbose:
                        print("\n[Direct SSH Connection]")
                        print(f"  Host: {self.platform.machine}:{self.platform.ssh_port}")
                        print(f"  Username: {self.platform.username}")

                    try:
                        self._ssh_client.connect(
                            self.platform.machine,
                            username=self.platform.username,
                            key_filename=ssh_key_path,
                            port=self.platform.ssh_port,
                            timeout=10,
                        )
                        ssh_transport = self._ssh_client.get_transport()
                        if ssh_transport is not None:
                            ssh_transport.set_keepalive(30)
                        if verbose: print(f"OK: Connected to: {self.platform.machine} as {self.platform.username}")
                    except (paramiko.AuthenticationException, paramiko.SSHException) as e:
                        raise RuntimeError(
                            f"Failed to authenticate to {self.platform.machine} "
                            f"as {self.platform.username} (port {self.platform.ssh_port}): {e}. "
                            "Check username, ssh_identity, and ssh_port in platform config."
                        )
                    logger.info(
                        "Connected directly to: %s:%s as %s",
                        self.platform.machine,
                        self.platform.ssh_port,
                        self.platform.username,
                    )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"SSH connection failed: {e}")

        return self._ssh_client

    @staticmethod
    def _find_free_port() -> int:
        """Find an available local port for tunneling."""
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def execute(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command locally or remotely."""
        if self.platform.is_local():
            return self._execute_local(command, cwd, timeout, check)
        return self._execute_remote(command, cwd, timeout, check)

    def _execute_local(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command locally."""
        try:
            env = dict(os.environ)
            python_dir = str(Path(sys.executable).parent)
            path = env.get("PATH", "")
            parts = path.split(os.pathsep) if path else []
            if python_dir not in parts:
                env["PATH"] = python_dir + (os.pathsep + path if path else "")

            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if check and result.returncode != 0:
                raise RuntimeError(f"Command failed with code {result.returncode}: {result.stderr}")
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    def _execute_remote(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute command remotely via SSH."""
        ssh = self._get_ssh_client()

        full_command = command
        
        # Ensure Python is discoverable on remote system (for GACODE wrapper scripts)
        python_dir = str(Path(sys.executable).parent)
        python_setup = f"export PATH={python_dir}:$PATH"
        
        if self.platform.modules:
            module_setup = _build_module_setup(self.platform.modules)
            if module_setup:
                full_command = f"{module_setup} && {full_command}"
        
        # Prepend Python PATH setup
        full_command = f"{python_setup} && {full_command}"
        
        if cwd:
            full_command = f"cd {cwd} && {full_command}"

        try:
            stdin, stdout, stderr = ssh.exec_command(full_command, timeout=timeout)
            stdout_str = stdout.read().decode("utf-8", errors="ignore")
            stderr_str = stderr.read().decode("utf-8", errors="ignore")
            returncode = stdout.channel.recv_exit_status()

            if check and returncode != 0:
                raise RuntimeError(f"Remote command failed with code {returncode}: {stderr_str}")

            return returncode, stdout_str, stderr_str
        except socket.timeout:
            raise TimeoutError(f"Remote command timed out after {timeout}s: {command}")

    def close(self) -> None:
        """Close SSH connection and tunnel."""
        if self._closed:
            return
        
        self._closed = True
        
        if self._local_forward:
            try:
                channel = self._local_forward[2]
                if channel:
                    channel.close()
            except Exception:
                pass
            self._local_forward = None

        if self._ssh_client:
            try:
                self._ssh_client.close()
            except Exception:
                pass
            self._ssh_client = None

        if self._tunnel_client:
            try:
                self._tunnel_client.close()
            except Exception:
                pass
            self._tunnel_client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup is called."""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Silently fail in destructor to avoid issues during shutdown
