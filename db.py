from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import os
from datetime import datetime
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to chatbot frontend domain
    allow_methods=["*"],
    allow_headers=["*"]
)
 
DB_PATH = r"chatbot.db"
 
 
@app.post("/log-user")
async def log_user(request: Request):
    try:
        body = await request.json()
        user_id = body.get("userId")
        user_name = body.get("userName")
        timestamp = datetime.now().isoformat()
 
        print("Logging:", user_id, user_name, timestamp)
 
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO chatbot_user_logins (user_id, user_name, timestamp) VALUES (?, ?, ?)",
                (user_id, user_name, timestamp)
            )
            conn.commit()
 
        return {"message": "User login recorded"}
 
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {e}"}, status_code=500)
 
@app.get("/get-logins")
def get_logins():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM chatbot_user_logins ORDER BY timestamp DESC")
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        return JSONResponse(content={"message": f"Error: {e}"}, status_code=500)
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="10.103.180.125",
        port=8057,
        ssl_certfile="mpl_cert.pem",
        ssl_keyfile="mpl_privkey.pem"
    )