from app.routers.jobs import router as jobs_router
from app.routers.auth import router as auth_router
from app.routers.projects import router as projects_router
from app.routers.agent import router as agent_router
from app.routers.search import router as search_router

__all__ = ["jobs_router", "auth_router", "projects_router", "agent_router", "search_router"]
