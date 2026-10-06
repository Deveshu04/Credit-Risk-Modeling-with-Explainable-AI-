from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "deveshu/credit-risk-xai-artifacts"


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "app" / "artifacts"), repo_id=REPO, repo_type="model", commit_message="Upload dashboard artifacts")
    print(f"uploaded {len(list((ROOT / 'app' / 'artifacts').iterdir()))} files to private repo {REPO}")


if __name__ == "__main__":
    main()
