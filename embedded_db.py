import numpy as np

from db import Session, Student

class EmbeddedDb:
    embedded_db = {}

    @classmethod
    def get(cls):
        return cls.embedded_db

    @classmethod
    def add_to_embedded_db(cls, name, group, embedding):
        if name not in cls.embedded_db:
            cls.embedded_db[name] = {
                "group": group,
                "embeddings": [embedding]
            }
        else:
            cls.embedded_db[name]["embeddings"].append(embedding)

    @classmethod
    def populate_db(cls):
        session = Session()
        students = session.query(Student).all()
        for student in students:
            cls.embedded_db[student.name] = {
                "group": student.group,
                "embeddings": [
                    np.frombuffer(embed.embedding, dtype=np.float32)
                    for embed in student.embeds
                ]
            }
        session.close()