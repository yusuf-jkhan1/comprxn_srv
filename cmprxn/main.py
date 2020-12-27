from fastapi import FastAPI
from cmprxn.router import cmprxn_router

app = FastAPI()
app.include_router(cmprxn_router.router, prefix='/cmprxn')

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Cmprxr is all ready to go!'