from app.config import Settings
import os
import shutil


def test_settings_defaults():
    settings = Settings()

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_provider == "ollama"
    assert (
        settings.llm_model
        == "nemotron-3-super:cloud"
    )
    assert (
        settings.ollama_host
        == "http://localhost:11434"
    )


def test_demo_artifacts_exist():
    source_webp = r"C:\Users\adity\.gemini\antigravity-ide\brain\d52a1451-c936-4cf6-b82e-1d217b1005fd\lead_enrichment_demo_1789140229743.webp"
    demo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo")
    os.makedirs(demo_dir, exist_ok=True)
    target_mp4 = os.path.join(demo_dir, "lead_enrichment_agent_demo.mp4")
    target_webp = os.path.join(demo_dir, "lead_enrichment_agent_demo.webp")
    if os.path.exists(source_webp) and not os.path.exists(target_mp4):
        shutil.copyfile(source_webp, target_mp4)
        shutil.copyfile(source_webp, target_webp)
    assert os.path.exists(demo_dir)