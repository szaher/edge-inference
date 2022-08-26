import os.path

import requests
from pydantic import BaseModel
from typing import Union
import numpy as np


class ModelHelper(BaseModel):
    layer_index: int
    layer_name: str
    start_index: int
    end_index: int
    weight_type: str
    data: list


class Conv2DHelper(BaseModel):
    layer_index: int
    layer_name: str
    weight_type: str
    kernel_index: list
    strides: list
    padding: str
    data_format: str
    dilations: list
    data: list


class AgentHost(object):

    def __init__(self, host, protocol="http", username="", password="", path="", model="", method="POST"):
        self.host = host
        self.protocol = protocol
        self.username = username
        self.password = password
        self.path = path
        self.model = model
        self.method = method

    def get_supported_ops(self):
        return ["mat_mul", "convolution"]

    def agent_status(self):
        session = requests.Session()
        uri = os.path.join(self.host, "status")
        response = session.get(uri)
        print(response.json())
        if not response.ok:
            return False
        return True

    def convolution(self, data: Conv2DHelper):
        if self.protocol not in ["http"]:
            raise Exception("Protocol not supported")
        session = requests.Session()
        uri = os.path.join(self.host, self.path, self.model, "convolv")
        response = session.request(method=self.method, url=uri, json=data.dict())
        if not response.ok:
            return []
        result = np.array(response.json())
        return result

    # This is host supported operations
    def mat_mul(self, data: ModelHelper):
        if self.protocol not in ["http"]:
            raise Exception("Protocol not supported")
        session = requests.Session()
        uri = os.path.join(self.host, self.path, self.model, "matmul")
        # print("Calling ..... ", uri, "..........................")
        response = session.request(method=self.method, url=uri, json=data.dict())
        if not response.ok:
            return []
        # print(response.json())
        result = np.array(response.json())
        return result
