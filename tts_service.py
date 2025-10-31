#!/usr/bin/env python3
"""
Qwen TTS服务模块（改进版）
支持并发控制、自动重试、异常保护
"""

import asyncio
import logging
import base64
from typing import Optional, AsyncGenerator
import aiohttp

import dashscope

# ===== 日志配置 =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TTS")

# ===== 并发控制 =====
MAX_CONCURRENT_TTS_REQUESTS = 3  # 降低并发数以避免QPS限制
# 注意：信号量将在函数内部动态创建，避免事件循环绑定问题

# ===== 节流控制 =====
import time
from collections import deque

class RateLimiter:
    """RPM节流器 - 针对Qwen TTS API的10 RPM限制"""
    def __init__(self, max_requests_per_minute: int = 8):  # 设置为8，留出2个缓冲
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """获取请求许可"""
        async with self.lock:
            now = time.time()
            # 清理60秒前的请求记录
            while self.requests and self.requests[0] <= now - 60.0:
                self.requests.popleft()
            
            # 如果达到RPM限制，等待
            if len(self.requests) >= self.max_requests:
                sleep_time = 60.0 - (now - self.requests[0])
                if sleep_time > 0:
                    logger.info(f"[节流] RPM限制，等待 {sleep_time:.2f} 秒")
                    await asyncio.sleep(sleep_time)
                    now = time.time()
                    # 重新清理过期的请求
                    while self.requests and self.requests[0] <= now - 60.0:
                        self.requests.popleft()
            
            self.requests.append(now)
            logger.info(f"[节流] 获取许可，当前RPM: {len(self.requests)}/{self.max_requests}")

# 全局节流器 - 限制每分钟最多8个请求（为API的10 RPM留出缓冲）
rate_limiter = RateLimiter(max_requests_per_minute=8)

# ===== Qwen TTS 配置 =====
TTS_CONFIG = {
    "api_key": "sk-8c526fe03364421fbf8b4c47cf3e25c7",
    "model": "qwen3-tts-flash",
    "language_type": "Chinese"
}

# ===== 声音选项 =====
QWEN_VOICE_TYPES = {
    "Nofish": "Nofish",
    "Elias": "Elias",
    "Kiki": "Kiki"
}


class TTSService:
    """Qwen TTS 异步服务类"""

    def __init__(self):
        dashscope.api_key = TTS_CONFIG["api_key"]
        self.model = TTS_CONFIG["model"]
        self.language_type = TTS_CONFIG["language_type"]
        self._semaphore = None  # 将在运行时动态创建
    
    def _get_semaphore(self):
        """获取当前事件循环的信号量"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_TTS_REQUESTS)
        return self._semaphore

    async def _fetch_audio(self, url: str) -> bytes:
        """异步下载音频文件"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()

    async def _qwen_tts_request(self, text: str, voice_type: str, retry: int = 3) -> Optional[bytes]:
        """发起Qwen TTS请求（自动重试 + 节流保护）"""
        # 先获取节流许可
        await rate_limiter.acquire()
        
        for attempt in range(1, retry + 1):
            try:
                logger.info(f"[TTS] 请求中... 尝试 {attempt}/{retry} 文本: {text[:30]}...")
                response = dashscope.MultiModalConversation.call(
                    model=self.model,
                    text=text,
                    voice=voice_type,
                    language_type=self.language_type,
                    stream=False
                )

                audio_url = getattr(response.output.audio, "url", None)
                if not audio_url:
                    raise ValueError("未返回音频URL")

                audio_data = await self._fetch_audio(audio_url)
                logger.info(f"[TTS] 获取成功: {len(audio_data)} bytes")
                return audio_data

            except Exception as e:
                logger.warning(f"[TTS] 第{attempt}次失败: {e}")
                # 重试时也要节流
                if attempt < retry:
                    await rate_limiter.acquire()
                await asyncio.sleep(1.5 * attempt)

        logger.error("[TTS] 重试三次后仍失败")
        return None

    async def text_to_speech_stream(self, text: str, voice_type: str) -> AsyncGenerator[bytes, None]:
        """异步生成音频流"""
        if not text.strip():
            return

        # 使用动态创建的信号量避免事件循环绑定问题
        semaphore = self._get_semaphore()
        async with semaphore:  # 控制最大并发
            logger.info(f"[TTS调试] 获取到信号量, 当前并发数 <= {MAX_CONCURRENT_TTS_REQUESTS}")
            audio_data = await self._qwen_tts_request(text, voice_type)
            if audio_data:
                yield audio_data
            else:
                logger.error(f"[TTS] 文本转换失败: {text[:50]}")


# ===== 全局实例 =====
tts_service = TTSService()


# ===== 对外主函数 =====
async def text_to_speech_base64(text: str, voice_type: str) -> Optional[str]:
    """将文本转换为Base64编码的音频（供API返回）"""
    try:
        audio_bytes = b""
        async for chunk in tts_service.text_to_speech_stream(text, voice_type):
            audio_bytes += chunk

        if not audio_bytes:
            return None

        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info(f"[TTS] 转换完成，音频大小: {len(audio_bytes)} bytes")
        return b64_audio

    except Exception as e:
        logger.error(f"[TTS] 转换异常: {e}")
        return None
