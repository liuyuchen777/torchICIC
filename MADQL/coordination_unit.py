'''
Author: Liu Yuchen
Date: 2021-06-22 15:29:47
LastEditors: Liu Yuchen
LastEditTime: 2021-06-22 15:49:49
Description: 
FilePath: /torch_ICIC/MADQL/coordination_unit.py
GitHub: https://github.com/liuyuchen777
'''


class CoordinationUnit:
    def __init__(self, pos):
        self.pos = pos
        self._get_sectors_pos_()

    def _get_sectors_pos_(self):
        