#!/usr/bin/env python3
"""
API Configuration Loader Tool

For unified management and loading of OpenAI API configuration
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional


def load_api_config(config_name: str = "default") -> Dict:
    """
    Load API configuration

    Args:
        config_name: Configuration name, e.g., "default", "generation", "validation"

    Returns:
        API configuration dictionary
    """
    # Find api_config.yaml
    config_file = Path(__file__).parent.parent / "api_config.yaml"

    if not config_file.exists():
        # Use default configuration
        return get_default_api_config()

    # Read configuration file
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    # Get specified configuration
    if config_name in config_data:
        api_config = config_data[config_name]
    else:
        api_config = config_data.get('default', {})

    # Handle api_key priority: environment variable > config file > default value
    fallback_key = api_config.get('api_key', 'your-api-key')  # Key from config file or default value

    if 'api_key_env' in api_config:
        env_var = api_config['api_key_env']
        api_config['api_key'] = os.environ.get(env_var, fallback_key)
    elif 'api_key' not in api_config:
        api_config['api_key'] = os.environ.get('OPENAI_API_KEY', 'your-api-key')

    return api_config


def get_default_api_config() -> Dict:
    """
    Get default API configuration (used when config file doesn't exist)
    """
    return {
        'provider': 'custom',
        'base_url': 'https://api2.aigcbest.top/v1',
        'api_key': os.environ.get('OPENAI_API_KEY', 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c'),
        'model': 'gpt-4o',
        'timeout': 120,
        'max_retries': 3
    }


def get_openai_client_code(config_name: str = "default") -> str:
    """
    Generate OpenAI client initialization code (for script generation)

    Args:
        config_name: Configuration name

    Returns:
        Python code string
    """
    api_config = load_api_config(config_name)

    base_url = api_config.get('base_url', 'https://api.openai.com/v1')
    timeout = api_config.get('timeout', 120)
    default_api_key = api_config.get('api_key', 'your-api-key')

    code = f'''# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY", "{default_api_key}")
API_BASE = os.environ.get("OPENAI_API_BASE", "{base_url}")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout={timeout}
)'''

    return code


def format_api_config_comment(config_name: str = "default") -> str:
    """
    Generate API configuration description comment

    Args:
        config_name: Configuration name

    Returns:
        Comment string
    """
    api_config = load_api_config(config_name)

    provider = api_config.get('provider', 'custom')
    base_url = api_config.get('base_url', 'N/A')

    comment = f'''# API Configuration Description:
# Provider: {provider}
# Base URL: {base_url}
#
# Usage:
# 1. Set environment variable: export OPENAI_API_KEY="your-key"
# 2. (Optional) Override base_url: export OPENAI_API_BASE="https://your-api.com/v1"
# 3. Run script'''

    return comment


if __name__ == "__main__":
    # Test
    print("=== Default Configuration ===")
    default_config = load_api_config("default")
    print(f"Base URL: {default_config['base_url']}")
    print(f"Model: {default_config['model']}")

    print("\n=== Generate OpenAI Client Code ===")
    print(get_openai_client_code("generation"))

    print("\n=== API Configuration Comment ===")
    print(format_api_config_comment())
