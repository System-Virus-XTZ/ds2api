# ds2api - DeepSeek API 代理

> 基于 [System-Virus-XTZ/ds2api](https://github.com/System-Virus-XTZ/ds2api) 的 DeepSeek API 代理，支持 OpenAI 兼容接口、多账号轮询、负载均衡。

---

## 功能特性

- **OpenAI 兼容接口** — 对接 `/v1/chat/completions`，现有应用无缝迁移
- **多账号轮询** — 支持配置多个 DeepSeek 账号，自动负载均衡
- **PoW 反爬保护** — 内置 PoW (工作量证明) 解算器
- **聊天历史管理** — 支持会话历史存储、查询与删除
- **JWT 管理员认证** — 独立后台管理面板
- **流式/非流式输出** — 支持 SSE 流式响应
- **模型映射** — 支持 thinking / no-thinking / search / vision 等多种模式

---

## 支持模型

| 模型 ID | 思考 | 搜索 | 视觉 |
|---|---|---|---|
| `deepseek-v4-flash` | ✅ | ❌ | ❌ |
| `deepseek-v4-flash-nothinking` | ❌ | ❌ | ❌ |
| `deepseek-v4-pro` | ✅ | ❌ | ❌ |
| `deepseek-v4-pro-nothinking` | ❌ | ❌ | ❌ |
| `deepseek-v4-flash-search` | ✅ | ✅ | ❌ |
| `deepseek-v4-flash-search-nothinking` | ❌ | ✅ | ❌ |
| `deepseek-v4-pro-search` | ✅ | ✅ | ❌ |
| `deepseek-v4-pro-search-nothinking` | ❌ | ✅ | ❌ |
| `deepseek-v4-vision` | ✅ | ❌ | ✅ |
| `deepseek-v4-vision-nothinking` | ❌ | ❌ | ✅ |

---

## 项目结构

```
DeepSeekAPI/
├── main.py                    # 程序入口
├── config.json               # 配置文件
├── requirements.txt          # Python 依赖
│
├── server/                   # ASGI 服务层
│   └── router.py             # 路由 & 请求处理
│
├── httpapi/                  # HTTP API 层
│   └── openai/chat/
│       └── handler_chat.py    # Chat Completions 处理器
│
├── account/                  # 账号池管理
│   ├── pool_core.py          # 账号池核心
│   ├── pool_acquire.py       # 账号获取
│   └── pool_limits.py        # 限流控制
│
├── auth/                     # 认证模块
│   ├── request.py           # 请求认证 & 账号解析
│   └── admin.py             # JWT 管理员认证
│
├── deepseek/                 # DeepSeek 客户端
│   ├── client/              # 底层 API 封装
│   │   ├── client_core.py       # 核心客户端
│   │   ├── client_auth.py       # 登录认证
│   │   ├── client_completion.py # 对话完成
│   │   ├── client_session.py    # 会话管理
│   │   ├── client_upload.py     # 文件上传
│   │   └── client_file_status.py# 文件状态
│   ├── protocol/            # 协议定义
│   │   ├── constants.py     # 常量
│   │   └── sse.py           # SSE 格式
│   └── client/pow.py        # PoW 解算器
│
├── completionruntime/        # 完成运行时（流式/非流式）
│   ├── stream_retry.py      # 流式 + 重试
│   └── nonstream.py         # 非流式
│
├── format/                   # 响应格式化
│   └── openai/
│       ├── render_chat.py        # OpenAI Chat 格式
│       └── render_stream_events.py # SSE 事件格式
│
├── chathistory/              # 聊天历史存储
│   └── store.py             # 历史记录读写
│
├── assistantturn/            # 助手轮次数据结构
│   └── turn.py             # Turn 模型 & Token 统计
│
└── config/                  # 配置管理
    ├── store.py             # 配置存储
    ├── account.py           # 账号配置模型
    ├── models.py            # 模型配置
    ├── credentials.py       # 凭证管理
    └── logger.py            # 日志配置
```

---

## API 接口

### 基础信息

- **Base URL**: `http://<host>:8000`
- **认证方式**: `Authorization: Bearer <api_key>`
- **Content-Type**: `application/json`

### 接口列表

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/` | 服务信息 |
| `POST` | `/v1/chat/completions` | 对话补全（核心接口） |
| `GET` | `/v1/models` | 列出可用模型 |
| `GET` | `/v1/models/{model}` | 获取模型信息 |
| `GET` | `/v1/history/list` | 聊天历史列表 |
| `POST` | `/v1/history/delete` | 删除聊天历史 |
| `POST` | `/v1/admin/login` | 管理员登录 |
| `GET/POST` | `/v1/admin/config` | 查看/更新配置 |
| `GET` | `/admin/pool/stats` | 账号池状态统计 |

### 调用示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-Sys…-XTZ" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

### Python 调用示例

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8000",
    headers={"Authorization": "Bearer sk-Sys…-XTZ"}
)

resp = client.post("/v1/chat/completions", json={
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "解释什么是量子计算"}],
    "stream": False
})
print(resp.json())
```

---

## 配置说明

### config.json

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "debug": false,
  "log_level": "info",
  "global_max_inflight": 10,
  "account_max_inflight": 5,
  "wait_timeout": 300,
  "accounts": [
    {
      "id": "account-1",
      "email": "your-email@email.com",
      "password": "your-password",
      "enabled": true,
      "priority": 10,
      "max_inflight": 3
    }
  ],
  "api_keys": ["your-api-key"],
  "models": [
    {"model": "deepseek-v4-flash", "thinking": true, "search": false, "vision": false, "enabled": true}
  ],
  "proxies": [],
  "chat_history_path": ".chat_history.json",
  "allowed_origins": ["*"]
}
```

### 关键字段说明

| 字段 | 说明 | 默认值 |
|---|---|---|
| `host` | 监听地址 | `0.0.0.0` |
| `port` | 监听端口 | `8000` |
| `global_max_inflight` | 全局最大并发请求数 | `10` |
| `account_max_inflight` | 单账号最大并发 | `5` |
| `wait_timeout` | 账号等待超时（秒） | `300` |
| `accounts` | DeepSeek 账号列表 | `[]` |
| `api_keys` | API Key 白名单 | `[]` |
| `models` | 启用的模型列表 | - |
| `proxies` | 代理服务器列表 | `[]` |

### 环境变量

| 变量 | 说明 |
|---|---|
| `CONFIG_PATH` | 配置文件路径 |
| `DS2API_CONFIG_PATH` | 同上，优先级更高 |
| `HOST` | 监听地址（覆盖 config） |
| `PORT` | 监听端口（覆盖 config） |
| `DEBUG` | 调试模式 (`1`/`true`) |
| `LOG_LEVEL` | 日志级别 |
| `DS2API_JWT_SECRET` | JWT 密钥（覆盖 config） |

---

## 安装与运行

### 1. 安装依赖

```bash
cd /root/DeepSeekAPI
pip install -r requirements.txt
```

### 2. 配置账号

编辑 `config.json`，填入 DeepSeek 账号信息（邮箱 + 密码）。

### 3. 启动服务

```bash
# 默认配置
python3 main.py

# 指定端口
python3 main.py --port 8080

# 开发模式（自动重载）
python3 main.py --reload

# 生产模式（多进程）
python3 main.py --workers 4 --log-level info
```

### 4. 使用 Docker / systemctl（可选）

可使用 `screen` / `systemd` / `pm2` 等方式保持后台运行。

---

## 安全建议

- **JWT Secret**: 生产环境务必修改 `jwt_secret`，不要使用默认值
- **API Key**: 通过环境变量 `DS2API_JWT_SECRET` 注入，不要硬编码在配置文件中
- **CORS**: `allowed_origins` 生产环境建议限制为具体域名
- **端口暴露**: 仅在必要时暴露公网访问，建议配合 Nginx 反向代理 + HTTPS

---

## 故障排查

**端口被占用**
```bash
lsof -i :8000
# 找到 PID 后 kill 或更换端口
```

**账号登录失败**
- 检查 `email` / `password` 是否正确
- DeepSeek 可能触发了验证码，需手动登录确认

**请求返回 401**
- 检查 `Authorization` header 是否正确
- 确认 `api_keys` 白名单中包含你的 key

**PoW 解算失败**
- 检查网络连接
- 确认 `deepseekpowsolver` 依赖已正确安装

---

## 版本

- **ds2api**: v2.0.4
- **Python**: >= 3.9
- **依赖**: uvicorn>=0.20.0, starlette>=0.30.0, httpx>=0.25.0, deepseekpowsolver>=0.1.3
