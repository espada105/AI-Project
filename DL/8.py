import asyncio

async def task1():
    print('데이터 가져오는중')
    await asyncio.sleep(1)
    print('task1 finish')

async def task2():
    print('데이터 가져오는중')
    await asyncio.sleep(2)
    print('task2 finish')

async def main():
    await asyncio.gather(task1(), task2())

asyncio.run(main())