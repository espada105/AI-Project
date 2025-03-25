class Singleton:
    _instance = None

    @staticmethod
    def get_instance():
        if  Singleton._instance is None:
            Singleton._instance = Singleton()
        return Singleton._instance


s1 = Singleton.get_instance()
s2 = Singleton.get_instance()
print(s1 is s2)

s1.data = 10
print(s1.data == s2.data)

