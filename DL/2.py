from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass
    def move(self):
        print('This animal moves')

class Dog(Animal):
    def sound(self):
        print("woof")

class Cat(Animal):
    def sound(self):
        print('meow')

def make_animal_sound(animal: Animal):
    print(animal.sound())

dog = Dog()
cat = Cat()

make_animal_sound(cat)
make_animal_sound(cat)