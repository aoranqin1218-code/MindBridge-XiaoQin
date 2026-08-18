from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.bootstrap import create_schema, seed_data
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.services.tool_queue import get_tool_queue_worker


def create_app() -> FastAPI:
    # 把一个 async def 生成器函数变成异步上下文管理器。简单说就是让这个函数可以用 async with 来调用
    @asynccontextmanager
    async def lifespan(app: FastAPI):                               # lifespan 本质上是 FastAPI 的"应用生命周期管理器"
        # startup
        create_schema()                                             # 建表

        # 从连接池借一条数据库连接，返回一个会话对象
        # 对db 做查询、插入、提交的操作，都是通过这条连接跟数据库通信
        db = SessionLocal()                                         
        try:
            seed_data(db)                                           # 写初始数据
        finally:
            db.close()
        worker = get_tool_queue_worker(get_settings())      
        worker.start()                                              # 启动后台 worker
        app.state.tool_queue_worker = worker
        yield                                                       # 代码执行到这一行就停下来，应用开始正常接收和处理请求
        # shutdown
        worker = getattr(app.state, "tool_queue_worker", None)
        if worker is not None:
            worker.stop()

    app = FastAPI(title="MindBridge Python", version="0.1.0", lifespan=lifespan)

    # 中间件
    @app.middleware("http")
    async def no_cache_frontend_assets(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store"                  # no-store 告诉浏览器"不要缓存这个文件，每次都必须重新请求服务器"
        return response

    # 挂载路由
    app.include_router(router)

    # 前端页面的挂载，可以不用看
    static_dir = Path(__file__).resolve().parent / "static"                 # 算出 static 目录的绝对路径
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")         # 把这个静态目录挂载到根路径 / 上

    return app


app = create_app()
