def push_space():
    from huggingface_hub import HfApi
    api = HfApi()

    api.upload_folder(
        folder_path=".",
        repo_id="Aayush5665/FlakeForge",
        repo_type="space",
    )

# after training finishes
push_space()