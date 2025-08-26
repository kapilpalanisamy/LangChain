from typing import TypedDict

class Person(TypedDict):
    name:str
    age:str

new_person: Person ={'name':'Kapil','age':'20'}

print(new_person)