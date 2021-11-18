import uvicorn

if __name__ == "__main__":
    uvicorn.run("strider.server:APP", host="0.0.0.0", port=5781, reload=True)
