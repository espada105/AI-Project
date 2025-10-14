# 방법 1: 새로운 문자열 생성 (대문자 변환)
a = 'python'
a = a.capitalize()  # 첫 글자를 대문자로 변환
print("방법 1:", a)

# 방법 2: 문자열 슬라이싱 사용
a = 'python'
a = 'P' + a[1:]  # 첫 글자를 'P'로 바꾸고 나머지 부분과 연결
print("방법 2:", a)

# 방법 3: replace 메서드 사용
a = 'python'
a = a.replace('p', 'P', 1)  # 첫 번째 'p'만 'P'로 교체
print("방법 3:", a)

# 방법 4: 리스트로 변환 후 수정
a = 'python'
a_list = list(a)  # 문자열을 리스트로 변환
a_list[0] = 'P'   # 리스트는 수정 가능
a = ''.join(a_list)  # 다시 문자열로 변환
print("방법 4:", a)