animal = AnimalFactory.create_animal('dog')
print(animal.sound())

animal = AnimalFactory.create_animal('cat')
print(animal.sound())

from abc import abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass

class dog(Animal):
    def sound(self):
        return 'bark'

class cat(Animal):
    def sound(self):
        return 'meow'

class Animalfactory:
    @staticmethod

    def create_animal(animal_type):
        if animal_type == 'dog':
            return Dog()
        elif animal_type == 'cat':
            return Cat()
        else:
            raise ValueError("unknown animal type")
