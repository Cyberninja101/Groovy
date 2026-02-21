import asyncio, json
import websockets

async def handler(ws):
    print("Client connected")
    async for msg in ws:
        data = json.loads(msg)

        # Save latest JSON (or trigger your Pi logic here)
        with open("latest.json", "w") as f:
            json.dump(data, f, indent=2)

        await ws.send(json.dumps({"ok": True, "saved": "latest.json"}))

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=50 * 1024 * 1024):
        print("Listening on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())