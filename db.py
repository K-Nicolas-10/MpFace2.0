from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()   

class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    group = Column(String, nullable=False)
    embeds = relationship("Embed", back_populates="student", cascade="all, delete-orphan")

class Embed(Base):
    __tablename__ = 'embeds'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'))
    embedding = Column(LargeBinary, nullable=False)
    student = relationship("Student", back_populates="embeds")

engine = create_engine('sqlite:///db.sqlite')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_all_students(session):
    return session.query(Student).all()

def get_student_by_id(session, student_id):
    return session.query(Student).filter(Student.id == student_id).first()

def get_student_by_embed_id(session, embed_id):
    return session.query(Student).join(Embed).filter(Embed.id == embed_id).first()

def add_embed(session, student, embedding_bytes):
    embed = Embed(student=student, embedding=embedding_bytes)
    session.add(embed)
    session.commit()
    return embed

def add_student(session, name, group):
    student = Student(name=name, group=group)
    session.add(student)
    session.commit()
    return student