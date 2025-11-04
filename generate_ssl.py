#!/usr/bin/env python3
"""
生成自签名SSL证书用于内网HTTPS访问
"""

import os
import subprocess
import sys

def generate_self_signed_cert():
    """生成自签名SSL证书"""
    cert_dir = "ssl_certs"
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    
    # 创建证书目录
    os.makedirs(cert_dir, exist_ok=True)
    
    # 检查是否已存在证书
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"证书文件已存在：\n  {cert_file}\n  {key_file}")
        return cert_file, key_file
    
    try:
        # 生成私钥
        subprocess.run([
            "openssl", "genrsa", "-out", key_file, "2048"
        ], check=True, capture_output=True)
        
        # 生成证书
        subprocess.run([
            "openssl", "req", "-new", "-x509", "-key", key_file,
            "-out", cert_file, "-days", "365",
            "-subj", "/C=CN/ST=State/L=City/O=Organization/CN=localhost"
        ], check=True, capture_output=True)
        
        print(f"SSL证书生成成功：\n  证书文件: {cert_file}\n  私钥文件: {key_file}")
        print("\n使用方法：")
        print("1. 将以下代码添加到 app.py 末尾：")
        print("""
if __name__ == '__main__':
    context = ('ssl_certs/cert.pem', 'ssl_certs/key.pem')
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=context)
""")
        print("2. 访问 https://192.168.19.139:5000 (浏览器会提示安全警告，点击'高级'继续)")
        
        return cert_file, key_file
        
    except subprocess.CalledProcessError as e:
        print(f"生成证书失败: {e}")
        print("请确保已安装 OpenSSL")
        return None, None
    except FileNotFoundError:
        print("未找到 OpenSSL 命令")
        print("Windows 用户可以：")
        print("1. 安装 Git for Windows (包含 OpenSSL)")
        print("2. 或者下载 OpenSSL for Windows")
        print("3. 或者使用在线工具生成证书")
        return None, None

if __name__ == "__main__":
    generate_self_signed_cert()
