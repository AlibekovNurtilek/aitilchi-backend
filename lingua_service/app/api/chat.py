from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.gemini_client import chat_once

router = APIRouter()
#ws://localhost:8000/ws/chat

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_input = await websocket.receive_text()
            reply = chat_once(user_input)
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        print("🔌 Клиент отключился")