# -*- coding: utf-8 -*-
""" main.py """
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def run():
    """Builds model, loads data, trains and evaluates"""
    print("model olusturulmadı")
    """
    model = UNet(CFG)
    model.load_data()
    model.build()
    model.train()
    model.evaluate()
    """


if __name__ == '__main__':
    run()
