#!/usr/bin/env python3
"""
간단한 Python 웹 애플리케이션 예제
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from datetime import datetime

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>도커 Python 앱</title>
                <meta charset="utf-8">
            </head>
            <body>
                <h1>🐳 도커로 실행되는 Python 앱</h1>
                <p>현재 시간: {}</p>
                <p>환경변수 APP_NAME: {}</p>
                <p><a href="/api/info">/api/info</a> - API 정보</p>
                <p><a href="/api/health">/api/health</a> - 헬스체크</p>
            </body>
            </html>
            """.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.getenv('APP_NAME', '설정되지 않음')
            )
            
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/api/info':
            self.send_json_response({
                'app': 'Simple Docker Python App',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'environment': {
                    'APP_NAME': os.getenv('APP_NAME', 'default'),
                    'PORT': os.getenv('PORT', '8000')
                }
            })
            
        elif self.path == '/api/health':
            self.send_json_response({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            })
            
        else:
            self.send_error(404, '페이지를 찾을 수 없습니다')
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))

def main():
    port = int(os.getenv('PORT', 8000))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    
    print(f"🚀 서버가 포트 {port}에서 시작되었습니다")
    print(f"📱 브라우저에서 http://localhost:{port} 접속하세요")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  서버를 종료합니다")
        server.shutdown()

if __name__ == '__main__':
    main()
