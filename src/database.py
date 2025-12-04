from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Create database engine
# For POC, using SQLite. For Enterprise, switch to PostgreSQL via env var.
DB_URL = os.getenv("DATABASE_URL", "sqlite:///agrovoltaico.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Site(Base):
    __tablename__ = "sites"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    capacity_kwp = Column(Float, default=100.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, index=True) # Foreign key logic can be added
    timestamp = Column(DateTime, default=datetime.utcnow)
    data = Column(JSON, nullable=False) # Storing forecast JSON for caching

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
