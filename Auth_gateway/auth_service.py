from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import logging
import os

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

origins = [
    "http://localhost:4200",
    "http://localhost:8081",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

KEYCLOAK_INTERNAL_URL = os.getenv("KEYCLOAK_INTERNAL_URL")
KEYCLOAK_PUBLIC_URL = os.getenv("KEYCLOAK_PUBLIC_URL")
REALM = os.getenv("KEYCLOAK_REALM", "authentication")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "public-client")
GATEWAY_CALLBACK = os.getenv("GATEWAY_CALLBACK")
FRONTEND_CALLBACK = os.getenv("FRONTEND_CALLBACK")


@app.get("/api/auth/login")
def login():
    keycloak_login_url = (
        f"{KEYCLOAK_PUBLIC_URL}/realms/{REALM}/protocol/openid-connect/auth"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={GATEWAY_CALLBACK}" 
    )
    return RedirectResponse(keycloak_login_url)


@app.get("/api/auth/logout")
def logout():
    logout_url = (
        f"{KEYCLOAK_PUBLIC_URL}/realms/{REALM}/protocol/openid-connect/logout"
        f"?client_id={CLIENT_ID}"
        f"&post_logout_redirect_uri=http://localhost:8081/api/auth/post-logout"
    )
    response = RedirectResponse(url=logout_url)

    for cookie_name in ["access_token", "user_id", "username", "roles", "scope"]:
        response.delete_cookie(cookie_name, path="/", domain="localhost")

    return response

@app.get("/api/auth/post-logout")
def post_logout():
    return RedirectResponse(url="http://localhost:4200")

@app.get("/api/auth/callback")
def callback(code: str = None):
    if not code:
        logger.error("No code received from Keycloak")
        return RedirectResponse("http://localhost:4200/login-error")

    try:
        payload = {
            "client_id": CLIENT_ID,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "http://localhost:8081/api/auth/callback" # MUST MATCH LOGIN
        }
        
        token_response = requests.post(
            f"{KEYCLOAK_INTERNAL_URL}/realms/{REALM}/protocol/openid-connect/token",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        if token_response.status_code != 200:
            logger.error(f"Keycloak exchange failed: {token_response.text}")
            raise HTTPException(status_code=400, detail=token_response.text)

        token_data = token_response.json()
        access_token = token_data.get("access_token")

        response = RedirectResponse(url="http://localhost:4200/login-callback")
        
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=False, 
            samesite="lax",
            path="/"
        )
        return response

    except Exception as e:
        logger.error(f"Callback Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/me")
async def me(request: Request):
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        return {"authenticated": False}

    return {
        "authenticated": True,
        "user_id": user_id,
        "username": request.headers.get("X-Username"),
        "roles": request.headers.get("X-User-Roles", "").split(",") if request.headers.get("X-User-Roles") else [],
        "scope": request.headers.get("X-User-Scope", "")
    }
