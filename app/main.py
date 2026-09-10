from app.config import get_settings
from app.utils import configure_logging, get_logger


logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()

    configure_logging(settings.log_level)

    logger.info("Autonomous Lead Enrichment Agent")
    logger.info("Environment: %s", settings.app_env)
    logger.info("LLM provider: %s", settings.llm_provider)
    logger.info("Max pages/domain: %s", settings.max_pages_per_domain)


if __name__ == "__main__":
    main()