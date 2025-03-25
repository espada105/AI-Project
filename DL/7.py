import asyncio

async def fetch_data():
    print('데이터 가져오는중')
    await asyncio.sleep(2)
    print('finish')

asyncio.run(fetch_data())