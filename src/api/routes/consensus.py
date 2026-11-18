"""Consensus API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_consensus_service
from src.core.exceptions import ConsensusError, LLMProviderError, ValidationError
from src.core.logging import get_logger
from src.core.services import ConsensusService
from src.models.request import ConsensusRequest
from src.models.response import ConsensusResponse

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/consensus",
    response_model=ConsensusResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze consensus",
    description="Send a question to multiple LLM models and get consensus answer.",
)
async def analyze_consensus(
    request: ConsensusRequest,
    service: ConsensusService = Depends(get_consensus_service),
) -> ConsensusResponse:
    """Analyze consensus across multiple LLM models.

    Sends the same question to multiple models, collects their answers,
    and determines the consensus using configurable strategies.
    """
    logger.info(
        "consensus_request_received",
        question_length=len(request.question),
        models=request.models,
    )

    try:
        return await service.analyze(request)

    except ValidationError as e:
        logger.warning("consensus_validation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": str(e)},
        ) from e

    except LLMProviderError as e:
        logger.warning("consensus_llm_error", provider=e.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "LLM_UNAVAILABLE", "message": str(e)},
        ) from e

    except ConsensusError as e:
        logger.warning("consensus_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "CONSENSUS_FAILED", "message": str(e)},
        ) from e

    except Exception as e:
        logger.error("consensus_unexpected_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
        ) from e
