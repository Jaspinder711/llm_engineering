import asyncio

async def square(number, delay):
    await asyncio.sleep(delay)
    print(f"square of {number} is {number**2}, calculated with delay {delay}s")
    return number**2

async def main():
    results = await asyncio.gather(
        square(2, 1),
        square(4, 2),
        square(5, 5)
    )
    print(results)

asyncio.run(main())