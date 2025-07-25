#!/usr/bin/env python3
"""
Setup script to create environment-specific configurations
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def create_base_config() -> Dict[str, Any]:
    """Create base configuration template"""
    return {
        "base_paths": {
            "backup": "/media/marco/SANDISK/GCOVER",
            "verification": "/media/marco/SANDISK/Verifications",
            "increment": "/media/marco/SANDISK/Increment",
        },
        "s3": {
            "bucket": "your-gdb-bucket",
            "key_prefix": "gdb-assets",
            "storage_class": "STANDARD",
        },
        "database": {
            "path": "gdb_metadata.duckdb",
            "backup": {"enabled": False, "frequency": "daily", "retention_days": 30},
        },
        "temp_dir": "/tmp/gdb_zips",
        "processing": {
            "compression_level": 6,
            "max_workers": 4,
            "max_retries": 3,
            "retry_delay": 5,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/gdb_manager.log",
            "max_size": "10MB",
            "backup_count": 5,
        },
        "monitoring": {
            "enabled": False,
            "webhook_url": "",
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
            },
        },
        "integrity_check": {"enabled": True, "timeout": 300},
        "cleanup": {"temp_files": True, "old_backups": True, "old_logs": 30},
    }


def create_development_config() -> Dict[str, Any]:
    """Create development-specific overrides"""
    return {
        "s3": {"bucket": "gdb-dev-bucket", "profile": "development"},
        "database": {"path": "dev_gdb_metadata.duckdb"},
        "temp_dir": "/tmp/gdb_dev",
        "processing": {"max_workers": 2},
        "logging": {"level": "DEBUG", "file": "logs/gdb_manager_dev.log"},
        "monitoring": {"enabled": True},
    }


def create_production_config() -> Dict[str, Any]:
    """Create production-specific overrides"""
    return {
        "s3": {
            "bucket": "gdb-production-bucket",
            "profile": "production",
            "storage_class": "STANDARD_IA",
        },
        "database": {
            "path": "/var/lib/gdb/metadata.duckdb",
            "backup": {"enabled": True, "frequency": "daily", "retention_days": 90},
        },
        "temp_dir": "/var/tmp/gdb",
        "processing": {"max_workers": 8, "compression_level": 9},
        "logging": {"level": "INFO", "file": "/var/log/gdb/manager.log"},
        "monitoring": {
            "enabled": True,
            "webhook_url": "https://your-monitoring-service.com/webhook",
            "email": {"enabled": True, "username": "alerts@yourcompany.com"},
        },
        "cleanup": {"old_logs": 90},
    }


def create_config_file(output_path: Path, interactive: bool = True):
    """Create configuration file with user input"""

    config_data = create_base_config()
    dev_config = create_development_config()
    prod_config = create_production_config()

    if interactive:
        print("🔧 GDB Asset Management Configuration Setup")
        print("=" * 50)

        # Basic settings
        print("\n📁 Base Paths:")
        for key, default_path in config_data["base_paths"].items():
            new_path = input(f"  {key.title()} path [{default_path}]: ").strip()
            if new_path:
                config_data["base_paths"][key] = new_path

        # AWS S3 settings
        print("\n☁️  AWS S3 Settings:")

        dev_bucket = input(
            f"  Development S3 bucket [{dev_config['s3']['bucket']}]: "
        ).strip()
        if dev_bucket:
            dev_config["s3"]["bucket"] = dev_bucket

        prod_bucket = input(
            f"  Production S3 bucket [{prod_config['s3']['bucket']}]: "
        ).strip()
        if prod_bucket:
            prod_config["s3"]["bucket"] = prod_bucket

        aws_profile = input("  AWS Profile (optional): ").strip()
        if aws_profile:
            dev_config["s3"]["profile"] = aws_profile
            prod_config["s3"]["profile"] = aws_profile

        # Database settings
        print("\n🗄️  Database Settings:")

        dev_db = input(
            f"  Development DB path [{dev_config['database']['path']}]: "
        ).strip()
        if dev_db:
            dev_config["database"]["path"] = dev_db

        prod_db = input(
            f"  Production DB path [{prod_config['database']['path']}]: "
        ).strip()
        if prod_db:
            prod_config["database"]["path"] = prod_db

        # Monitoring
        print("\n📊 Monitoring (Production):")
        webhook = input("  Webhook URL (optional): ").strip()
        if webhook:
            prod_config["monitoring"]["webhook_url"] = webhook

        email = input("  Alert email (optional): ").strip()
        if email:
            prod_config["monitoring"]["email"]["username"] = email
            prod_config["monitoring"]["email"]["enabled"] = True

    # Write configuration file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write base config
        yaml.dump(config_data, f, default_flow_style=False)

        # Write development overrides
        f.write("\n---\n# Development environment overrides\n")
        yaml.dump(dev_config, f, default_flow_style=False)

        # Write production overrides
        f.write("\n---\n# Production environment overrides\n")
        yaml.dump(prod_config, f, default_flow_style=False)

    print(f"\n✅ Configuration created: {output_path}")
    print("\n🚀 Next steps:")
    print("  1. Review and customize the configuration file")
    print("  2. Set up AWS credentials if not already done")
    print("  3. Run: gdb-manage --env dev init")
    print("  4. Test with: gdb-manage --env dev scan")


def create_env_file(output_path: Path, environment: str = "development"):
    """Create .env file template"""

    env_vars = {
        "development": {
            "GDB_ENV": "development",
            "GDB_S3_BUCKET": "gdb-dev-bucket",
            "GDB_LOG_LEVEL": "DEBUG",
            "GDB_MAX_WORKERS": "2",
        },
        "production": {
            "GDB_ENV": "production",
            "GDB_S3_BUCKET": "gdb-production-bucket",
            "GDB_LOG_LEVEL": "INFO",
            "GDB_MAX_WORKERS": "8",
        },
    }

    with open(output_path, "w") as f:
        f.write(f"# GDB Asset Management - {environment.title()} Environment\n")
        f.write(f"# Generated on {os.getenv('USER', 'user')}@{os.uname().nodename}\n\n")

        for key, value in env_vars.get(environment, env_vars["development"]).items():
            f.write(f"{key}={value}\n")

        f.write("\n# Optional AWS settings\n")
        f.write("# AWS_PROFILE=your-profile\n")
        f.write("# AWS_DEFAULT_REGION=us-east-1\n")

        f.write("\n# Optional overrides\n")
        f.write("# GDB_DB_PATH=/custom/path/to/db.duckdb\n")
        f.write("# GDB_TEMP_DIR=/custom/temp/dir\n")

    print(f"📄 Environment file created: {output_path}")
    print(f"   Source it with: source {output_path}")


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup GDB Asset Management configuration"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Configuration directory",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Create default configuration without prompts",
    )
    parser.add_argument(
        "--create-env", choices=["dev", "prod"], help="Create .env file for environment"
    )

    args = parser.parse_args()

    if args.create_env:
        env_map = {"dev": "development", "prod": "production"}
        env_file = Path(f".env.{args.create_env}")
        create_env_file(env_file, env_map[args.create_env])
        return

    # Create main configuration
    config_file = args.config_dir / "gdb_config.yaml"
    create_config_file(config_file, interactive=not args.non_interactive)

    # Create environment-specific .env files
    print("\n📄 Creating environment files...")
    create_env_file(Path(".env.dev"), "development")
    create_env_file(Path(".env.prod"), "production")

    print("\n🎯 Usage examples:")
    print("  # Development:")
    print("  source .env.dev && gdb-manage scan")
    print("  # or")
    print("  gdb-manage --env dev scan")
    print()
    print("  # Production:")
    print("  source .env.prod && gdb-manage sync")
    print("  # or")
    print("  gdb-manage --env prod sync")


if __name__ == "__main__":
    main()
