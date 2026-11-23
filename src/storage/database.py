"""SQLAlchemy database models and connection."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from src.core.config import get_settings


class Base(DeclarativeBase):
    pass


class BenchmarkRunModel(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String(100), unique=True, nullable=False, index=True)
    dataset = Column(String(50), nullable=False)
    models = Column(JSON, nullable=False)
    status = Column(String(20), nullable=False, default="running")
    total_tasks = Column(Integer, default=0)
    unanimous_agreement = Column(Integer, default=0)
    majority_agreement = Column(Integer, default=0)
    disagreement = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    model_stats = relationship("ModelStatsModel", back_populates="benchmark_run", cascade="all, delete-orphan")
    task_results = relationship("TaskResultModel", back_populates="benchmark_run", cascade="all, delete-orphan")


class ModelStatsModel(Base):
    __tablename__ = "model_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, ForeignKey("benchmark_runs.id"), nullable=False)
    model = Column(String(100), nullable=False)
    total_tasks = Column(Integer, default=0)
    correct = Column(Integer, default=0)
    incorrect = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    total_latency_ms = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    avg_latency_ms = Column(Float, default=0.0)
    avg_tokens = Column(Float, default=0.0)

    # Relationships
    benchmark_run = relationship("BenchmarkRunModel", back_populates="model_stats")


class TaskResultModel(Base):
    __tablename__ = "task_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_run_id = Column(Integer, ForeignKey("benchmark_runs.id"), nullable=False)
    model = Column(String(100), nullable=False)
    task_id = Column(String(50), nullable=False)
    question = Column(Text, nullable=False)
    correct_answer = Column(Float, nullable=False)
    model_answer = Column(Float, nullable=True)
    is_correct = Column(Boolean, default=False)
    latency_ms = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    benchmark_run = relationship("BenchmarkRunModel", back_populates="task_results")


class ConsensusRequestModel(Base):
    __tablename__ = "consensus_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    models = Column(JSON, nullable=False)
    consensus_answer = Column(Float, nullable=True)
    confidence = Column(Float, default=0.0)
    agreement = Column(String(20), nullable=True)
    total_latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


_engine = None
_session_factory = None


async def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=settings.debug)
    return _engine


async def get_session_factory():
    global _session_factory
    if _session_factory is None:
        engine = await get_engine()
        _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncSession:
    factory = await get_session_factory()
    async with factory() as session:
        yield session
