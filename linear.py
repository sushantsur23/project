from pydantic import BaseModel

class Linearassgn(BaseModel):
    PT: float
    RS: float
    Torque: float
    TW: float
    TWF: float
    HDF: float
    PWF: float
    OSF: float
    RNF: float 