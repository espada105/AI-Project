import asyncio
import time

async def fetch_data():
    print("데이터 가져오는중")
    await asyncio.sleep(2)
    print("데이터 로드 완료")
    return {"data":"example data"}

async def main():
    start_time = time.time()

    data1 = asyncio.create_task(fetch_data())
    data2 = asyncio.create_task(fetch_data())

    task1 = await data1
    task2 = await data2

    end_time = time.time()

    print(f"data load time: {end_time - start_time}초")

asyncio.run(main())