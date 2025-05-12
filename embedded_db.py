import numpy as np

from db import Session, Student

class EmbeddedDb:
    def __init__(self):
        self.embedded_db = {}
        self.session = Session()
        self.populate_db()

    def get(self):
        return self.embedded_db
    
    def add_to_embedded_db(self, name, group, embedding):
        if name not in self.embedded_db:
            self.embedded_db[name] = {
                "group": group,
                "embeddings": [embedding]
            }
        else:
            self.embedded_db[name]["embeddings"].append(embedding)

    def populate_db(self):
        students = self.session.query(Student).all()
        for student in students:
            self.embedded_db[student.name] = {
                "group": student.group,
                "embeddings": [
                    np.frombuffer(embed.embedding, dtype=np.float32)
                    for embed in student.embeds
                ]
            }