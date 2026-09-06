import os
import sys
import argparse
import subprocess
from huggingface_hub import HfApi, create_repo, get_token

def main():
    parser = argparse.ArgumentParser(description="Publish Project Norn V15 to Hugging Face")
    parser.add_argument("--repo_id", type=str, required=True, help="Target repository ID, e.g., 'osakra-research/norn-v15' or 'your-username/norn-v15'")
    parser.add_argument("--github_url", type=str, default=None, help="Target GitHub repository URL, e.g., 'https://github.com/Osakra-Research/Norn.git'")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (with write permission). Defaults to stored token.")
    parser.add_argument("--private", action="store_true", help="Set repository to private (default is public)")
    args = parser.parse_args()

    token = args.token or get_token()
    if not token:
        print("\n[!] Error: No Hugging Face token detected.")
        print("    Please pass your token with --token <YOUR_HF_WRITE_TOKEN> or run 'huggingface-cli login'.")
        print("    You can create a Write token at: https://huggingface.co/settings/tokens\n")
        sys.exit(1)

    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_norn_v15")
    if not os.path.exists(folder_path):
        folder_path = os.path.dirname(os.path.abspath(__file__))

    print("=" * 75)
    print("  PROJECT NORN V15: HUGGING FACE ONE-CLICK PUBLISHER")
    print("=" * 75)
    print(f"[+] Local Package Path : {folder_path}")
    print(f"[+] Target Repository  : {args.repo_id}")
    print(f"[+] Visibility         : {'Private' if args.private else 'Public'}")

    api = HfApi(token=token)

    try:
        print("\n[1/2] Verifying / Creating repository on Hugging Face Hub...")
        create_repo(
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            repo_type="model",
            exist_ok=True
        )
        print(f"      Repository '{args.repo_id}' ready.")

        print("\n[2/2] Uploading weights, adapters, code, and documentation...")
        print("      (Streaming adapter_model.safetensors, biology weights, charts, and README)...")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=["*.pyc", "__pycache__/*", "*.tmp", ".github/*", ".git/*", ".git", "*.log", ".DS_Store", "Thumbs.db"]
        )

        print("\n" + "=" * 75)
        print("  SUCCESS! Project Norn V15 is officially published on Hugging Face!")
        print(f"  Repository URL: https://huggingface.co/{args.repo_id}")
        print("=" * 75 + "\n")

        if args.github_url:
            print("\n[3/3] Uploading to GitHub...")
            try:
                # Run git commands in folder_path
                if not os.path.exists(os.path.join(folder_path, ".git")):
                    subprocess.run(["git", "init"], cwd=folder_path, check=True)
                
                # Check if remote exists
                remotes = subprocess.run(["git", "remote"], cwd=folder_path, capture_output=True, text=True).stdout
                if "origin" not in remotes:
                    subprocess.run(["git", "remote", "add", "origin", args.github_url], cwd=folder_path, check=True)
                else:
                    subprocess.run(["git", "remote", "set-url", "origin", args.github_url], cwd=folder_path, check=True)
                
                subprocess.run(["git", "add", "."], cwd=folder_path, check=True)
                subprocess.run(["git", "commit", "-m", "Initial Norn V15 Release with training scripts"], cwd=folder_path, check=False)
                
                print("      Pushing to GitHub...")
                subprocess.run(["git", "branch", "-M", "main"], cwd=folder_path, check=True)
                subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=folder_path, check=True)
                print(f"  SUCCESS! Project Norn V15 is published on GitHub at {args.github_url}")
                print("=" * 75 + "\n")
            except subprocess.CalledProcessError as e:
                print(f"\n[!] GitHub Upload encountered an error: {e}")

    except Exception as e:
        print(f"\n[!] Upload encountered an error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
