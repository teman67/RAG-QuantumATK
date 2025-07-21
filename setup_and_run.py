import os
import sys
import subprocess
import json
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        'flask',
        'requests',
        'beautifulsoup4',
        'sentence-transformers',
        'faiss-cpu',
        'openai',
        'numpy',
        'tiktoken',
        'lxml',
    ]
    
    print("Installing required packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")
            return False
    
    return True

def create_config():
    """Create configuration file"""
    config = {
        "openai_api_key": "",
        "max_pages_to_scrape": 200,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gpt-3.5-turbo",
        "base_urls": [
            "https://docs.quantumatk.com/",
            
        ]
    }
    
    config_path = Path("config.json")
    
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created config.json - please add your OpenAI API key")
        return False
    
    return True

def check_config():
    """Check if configuration is valid"""
    config_path = Path("config.json")
    
    if not config_path.exists():
        print("Config file not found. Creating template...")
        create_config()
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if not config.get('openai_api_key'):
        print("Please add your OpenAI API key to config.json")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🔬 QuantumATK RAG System Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Exiting.")
        sys.exit(1)
    
    # Create/check config
    if not create_config():
        print("\n📝 Please edit config.json with your OpenAI API key, then run again.")
        sys.exit(1)
    
    if not check_config():
        sys.exit(1)
    
    # Create directories
    os.makedirs("quantumatk_kb", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Use the /setup endpoint to initialize the system")
    print("4. Use the /build_kb endpoint to build knowledge base")
    print("5. Start querying!")
    
    # Ask if user wants to run the app
    if input("\nWould you like to start the web app now? (y/n): ").lower() == 'y':
        try:
            subprocess.run([sys.executable, "app.py"])
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    main()
