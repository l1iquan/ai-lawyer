from flask import Flask, render_template, request, jsonify, Response
from rag_agent_stream import main_agent, get_legal_advice
from langgraph.checkpoint.memory import InMemorySaver
from tts_service import text_to_speech_base64
import json
import asyncio
import threading
from queue import Queue
import uuid
import ssl
import os

app = Flask(__name__)

# 存储用户会话
user_sessions = {}

class UserSession:
    def __init__(self):
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.consultation_complete = False
        self.last_case_summary = ""
        self.last_advice = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回空内容，避免404错误

@app.route('/test')
def test():
    """网络连接测试页面"""
    try:
        with open('test_connection.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "测试页面未找到", 404

# ✅ 限制 TTS 最大并发数
tts_semaphore = asyncio.Semaphore(7)

@app.route('/tts', methods=['POST'])
def tts():
    """TTS语音合成接口 - 支持最大并发7"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        voice_type = data.get('voice_type', 'ICL_zh_female_lingdongxinxin_cs_tob')

        print(f"[TTS请求] text={text}, voice_type={voice_type}")
        if not text:
            return jsonify({'error': '文本不能为空'}), 400

        async def run_tts_with_limit():
            async with tts_semaphore:
                print(f"[TTS调试] 获取到信号量, 当前并发数 <= 7")
                return await text_to_speech_base64(text, voice_type)

        # ✅ 每次请求独立事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_base64 = loop.run_until_complete(run_tts_with_limit())
        finally:
            loop.close()

        if not audio_base64:
            print("[TTS错误] 未生成音频数据")
            return jsonify({'error': '语音合成失败'}), 500

        print(f"[TTS成功] base64长度: {len(audio_base64)}")
        return jsonify({'audio_data': audio_base64})

    except Exception as e:
        print(f"[TTS异常] {e}")
        return jsonify({'error': f'TTS服务错误: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id', '')
    
    # 获取或创建用户会话
    if session_id not in user_sessions:
        user_sessions[session_id] = UserSession()
    
    session = user_sessions[session_id]
    
    def generate():
        try:
            # 使用流式输出调用主agent
            assistant_response = ""
            
            # 使用 stream_mode="messages" 来流式输出LLM tokens
            for token, metadata in main_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=session.config,
                stream_mode="messages"
            ):
                # 只处理来自model节点的文本内容
                if metadata.get('langgraph_node') == 'model':
                    for content_block in token.content_blocks:
                        if content_block.get('type') == 'text':
                            text = content_block.get('text', '')
                            assistant_response += text
                            # 发送流式数据
                            yield f"data: {json.dumps({'type': 'message', 'content': text})}\n\n"
            
            # 检测并保存案件摘要，然后自动调用工具
            if "{案件摘要：" in assistant_response and not session.consultation_complete:
                session.last_case_summary = assistant_response
                
                # 自动调用工具获取建议
                try:
                    # 发送开始获取法律建议的提示
                    intro_text = "\n\n基于您的案件情况，我为您提供了以下专业的法律建议：\n\n"
                    yield f"data: {json.dumps({'type': 'message', 'content': intro_text})}\n\n"
                    
                    # 流式获取法律建议
                    full_advice = ""
                    if "案件摘要：" in session.last_case_summary:
                        clean_summary = session.last_case_summary.split("案件摘要：")[1].strip()
                    else:
                        clean_summary = session.last_case_summary
                    
                    # 直接调用流式API获取法律建议
                    import requests
                    
                    payload = {"query": clean_summary}
                    headers = {"Content-Type": "application/json"}
                    
                    response = requests.post(
                        "http://localhost:8000/api/stream-text",
                        data=json.dumps(payload),
                        headers=headers,
                        stream=True,
                        timeout=30
                    )
                    
                    response.raise_for_status()
                    
                    # 流式输出法律建议
                    for line in response.iter_lines(chunk_size=1):
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                
                                if data_str == '[DONE]':
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    if 'content' in data:
                                        content = data['content']
                                        full_advice += content
                                        # 实时流式输出
                                        yield f"data: {json.dumps({'type': 'message', 'content': content})}\n\n"
                                except json.JSONDecodeError:
                                    full_advice += data_str
                                    yield f"data: {json.dumps({'type': 'message', 'content': data_str})}\n\n"
                    
                    session.last_advice = full_advice
                    session.consultation_complete = True
                    
                    outro_text = "\n\n如果您对任何建议有疑问，可以随时问我。"
                    yield f"data: {json.dumps({'type': 'message', 'content': outro_text})}\n\n"
                    
                    # 将法律建议添加到Agent的对话历史中（不显示给用户）
                    _ = main_agent.invoke(
                        {"messages": [
                            {"role": "assistant", "content": f"基于您的案件情况，我为您提供了以下专业的法律建议：\n\n{full_advice}\n\n如果您对任何建议有疑问，可以随时问我。"}
                        ]},
                        config=session.config
                    )
                    
                except Exception as e:
                    error_msg = f"调用知识库失败: {str(e)}"
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            error_msg = f"Agent调用失败: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

def create_self_signed_cert():
    """创建自签名证书"""
    cert_dir = "ssl_certs"
    os.makedirs(cert_dir, exist_ok=True)
    
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"证书文件已存在：\n  {cert_file}\n  {key_file}")
        return cert_file, key_file
    
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # 创建证书
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Organization"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.DNSName("192.168.19.139"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # 写入文件
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        print(f"SSL证书生成成功：\n  证书文件: {cert_file}\n  私钥文件: {key_file}")
        return cert_file, key_file
        
    except ImportError:
        print("需要安装 cryptography 库: pip install cryptography")
        return None, None
    except Exception as e:
        print(f"生成证书失败: {e}")
        return None, None

if __name__ == '__main__':
    # 尝试创建SSL证书
    cert_file, key_file = create_self_signed_cert()
    
    if cert_file and key_file:
        print("启动HTTPS服务器...")
        print("访问地址:")
        print("  https://localhost:5000")
        print("  https://127.0.0.1:5000") 
        print("  https://192.168.19.139:5000")
        print("\n注意：浏览器会提示安全警告，请点击'高级'->'继续访问'")
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        
        app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=context)
    else:
        print("启动HTTP服务器...")
        print("访问地址: http://localhost:5000")
        print("注意：HTTP模式下语音识别功能可能受限")
        app.run(debug=True, host='0.0.0.0', port=5000)
