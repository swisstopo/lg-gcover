# Test script: test_unified_config.py
from gcover.config import load_config




from gcover.config.loader import debug_config_loading



def test_debug_config():
    try:
        # Debug development loading
        debug_config_loading(environment="development")

        # Debug production loading
        debug_config_loading(environment="production")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_config():
    """Test the unified configuration system"""
    try:
        # Test loading
        app_config = load_config(environment="development")

        print("✅ Configuration 'development' loaded successfully")
        print(f"   S3 Bucket: {app_config.global_.s3.bucket}")
        print(f"   Log Level: {app_config.global_.log_level}")
        print(f"   GDB Database: {app_config.gdb.db_path}")

        # Test S3 access via helper methods
        s3_bucket = app_config.gdb.get_s3_bucket(app_config.global_)
        print(f"   GDB can access S3: {s3_bucket}")

        # Test validation
        print(f"   Max Workers: {app_config.gdb.processing.max_workers}")
        print(f"   Compression: {app_config.gdb.processing.compression_level}")

        app_config = load_config(environment="production")
        print("✅ Configuration 'production' loaded successfully")
        print(f"   Log Level: {app_config.global_.log_level}")
        print(f"   GDB Database: {app_config.gdb.db_path}")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


if __name__ == "__main__":
    test_debug_config()
    test_config()